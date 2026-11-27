"""批量评测：并行跑一组恒定速度指令，输出指标 CSV。

每个 episode 无窗口运行，只统计量化指标（位移、平均速度、最低高度、
最大姿态角、是否跌倒），用于参数扫描和回归对比。

开启域随机化后，每条（指令, trial）会按 `batch/seed_XXXX` 会话独立采样一组
物理参数，并把关键参数写进结果 CSV，便于按参数分组统计鲁棒性。
"""

import csv
import itertools
from concurrent.futures import ProcessPoolExecutor

import mujoco
import numpy as np

from .config import RobotConfig
from .control.policy import RlController
from .model import build_model
from .paths import CONFIG_ROOT
from .randomization import (
    SensorModel,
    apply_sample,
    body_shapes,
    derive_rng,
    sample,
)
from .robot import Robot

RAND_FIELDS = (
    "episode", "randomized",
    "mass_scale", "ground_friction", "robot_friction_scale",
    "solref_timeconst_scale", "joint_damping_scale", "torque_limit_scale",
    "kp_scale", "kd_scale",
)
FIELDS = (
    "vx", "vy", "yaw", *RAND_FIELDS,
    "dx", "dy", "mean_vx", "h_min", "roll_max", "pitch_max", "fell",
)


def _rand_values(rand_sample, episode: int = 0) -> dict:
    """把采样到的关键标量参数展开成 CSV 列；未随机化时留空。"""
    values = {name: "" for name in RAND_FIELDS}
    values["episode"] = int(episode)
    values["randomized"] = False
    if rand_sample is None:
        return values

    params = rand_sample["params"]
    dyn = params.get("dynamics") or {}
    con = params.get("contact") or {}
    act = params.get("actuator") or {}
    # 逐 body 质量缩放是向量，CSV 单列用均值代表（便于按参数分组）。
    mass = dyn.get("mass_scale", "")
    if isinstance(mass, (list, tuple)):
        mass = round(sum(float(v) for v in mass) / len(mass), 6) if mass else ""
    values.update({
        "episode": int(rand_sample["meta"]["episode"]),
        "randomized": True,
        "mass_scale": mass,
        "ground_friction": con.get("ground_friction", ""),
        "robot_friction_scale": con.get("robot_friction_scale", ""),
        "solref_timeconst_scale": con.get("solref_timeconst_scale", ""),
        "joint_damping_scale": act.get("joint_damping_scale", ""),
        "torque_limit_scale": act.get("torque_limit_scale", ""),
        "kp_scale": act.get("kp_scale", ""),
        "kd_scale": act.get("kd_scale", ""),
    })
    return values


def rollout(vx: float, vy: float, yaw: float, duration: float = 5.0, *,
            rand_plan=None, rand_session: str = "batch",
            episode: int = 0) -> dict:
    """无窗口跑一段恒定速度指令，返回该 episode 的指标。"""
    cfg = RobotConfig.load()
    model = build_model()

    rand_sample = None
    applied = None
    dt_ctrl = cfg.dt_ctrl
    sensor = None
    if rand_plan is not None:
        rand_sample = sample(
            rand_plan, rand_session, episode,
            shapes=body_shapes(model, rand_plan),
        )
        applied = apply_sample(
            model, rand_sample, dt_ctrl=cfg.dt_ctrl,
            rng=derive_rng(rand_plan, f"{rand_session}:apply", episode),
        )
        dt_ctrl = applied.runtime.control_dt or cfg.dt_ctrl
        sensor = SensorModel(
            applied.runtime.sensor,
            dt=dt_ctrl,
            rng=derive_rng(rand_plan, f"{rand_session}:sensor", episode),
        )

    data = mujoco.MjData(model)
    robot = Robot(model, data)
    kp, kd = (
        applied.runtime.scaled_gains(cfg.kp, cfg.kd)
        if applied is not None else (cfg.kp, cfg.kd)
    )
    ctrl = RlController(
        cfg, robot, CONFIG_ROOT / cfg.policy_file,
        kp=kp, kd=kd, sensor=sensor,
        motor_delay_steps=applied.runtime.motor_delay_steps if applied is not None else 0,
    )

    robot.set_joint_pose(cfg.default_q)
    robot.forward()
    robot.place_on_ground()

    n_sub = max(1, round(dt_ctrl / model.opt.timestep))
    n_steps = max(1, round(duration / dt_ctrl))
    cmd = np.array([vx, vy, yaw], dtype=float)

    x0, y0 = float(data.qpos[0]), float(data.qpos[1])
    h_min = float(data.qpos[2])
    roll_max = pitch_max = 0.0

    for _ in range(n_steps):
        data.ctrl[:] = ctrl.act(cmd)
        for _ in range(n_sub):
            mujoco.mj_step(model, data)
        h_min = min(h_min, float(data.qpos[2]))
        roll, pitch, _ = robot.rpy()
        roll_max = max(roll_max, abs(float(roll)))
        pitch_max = max(pitch_max, abs(float(pitch)))

    dx = float(data.qpos[0] - x0)
    dy = float(data.qpos[1] - y0)
    row = {
        "vx": vx,
        "vy": vy,
        "yaw": yaw,
        "dx": dx,
        "dy": dy,
        "mean_vx": dx / duration,
        "h_min": h_min,
        "roll_max": roll_max,
        "pitch_max": pitch_max,
        "fell": bool(h_min < 0.6 or roll_max > 45.0 or pitch_max > 45.0),
    }
    row.update(_rand_values(rand_sample, episode))
    return row


def _rollout_job(args) -> dict:
    vx, vy, yaw, duration, rand_plan, session, episode = args
    return rollout(
        vx, vy, yaw, duration,
        rand_plan=rand_plan, rand_session=session, episode=episode,
    )


def run_grid(vx_values, vy_values, yaw_values, duration: float = 5.0,
             workers: int = 1, out: str = "batch_results.csv", *,
             rand_plan=None, episodes: int = 1, seed: int = 0):
    """笛卡尔积展开速度网格，逐组评测并写 CSV。"""
    trials = max(1, int(episodes)) if rand_plan is not None else 1
    session = f"batch/seed_{int(seed):04d}"
    combos = [
        (vx, vy, yaw, duration, rand_plan, session, episode)
        for vx, vy, yaw in itertools.product(vx_values, vy_values, yaw_values)
        for episode in range(trials)
    ]

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_rollout_job, combos))
    else:
        rows = [_rollout_job(combo) for combo in combos]

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    if rand_plan is not None:
        print(f"随机化: {len(rows)} 个 trial，session={session}，episodes={trials}")
    print(f"{'vx':>5} {'vy':>5} {'yaw':>5} {'ep':>3} {'dx':>7} {'mean_vx':>8} "
          f"{'h_min':>6} {'roll':>6} {'pitch':>6}  fell")
    for r in rows:
        print(f"{r['vx']:5.2f} {r['vy']:5.2f} {r['yaw']:5.2f} {r['episode']:3d} "
              f"{r['dx']:7.2f} {r['mean_vx']:8.2f} {r['h_min']:6.3f} "
              f"{r['roll_max']:6.1f} {r['pitch_max']:6.1f}  "
              f"{'yes' if r['fell'] else 'no'}")

    ok = sum(not r["fell"] for r in rows)
    print(f"\n{ok}/{len(rows)} 个 trial 未跌倒，结果已写入 {out}")
    return rows
