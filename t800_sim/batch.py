"""批量评测：并行跑一组恒定速度指令，输出指标 CSV。

每个 episode 无窗口运行，只统计用量化指标（位移、平均速度、最低高度、
最大姿态角、是否跌倒），用于参数扫描和回归对比。
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
from .robot import Robot

FIELDS = ("vx", "vy", "yaw", "dx", "dy", "mean_vx", "h_min", "roll_max", "pitch_max", "fell")


def rollout(vx: float, vy: float, yaw: float, duration: float = 5.0) -> dict:
    """无窗口跑一段恒定速度指令，返回该 episode 的指标。"""
    cfg = RobotConfig.load()
    model = build_model()
    data = mujoco.MjData(model)
    robot = Robot(model, data)
    ctrl = RlController(cfg, robot, CONFIG_ROOT / cfg.policy_file)

    robot.set_joint_pose(cfg.default_q)
    robot.forward()
    robot.place_on_ground()

    n_sub = max(1, round(cfg.dt_ctrl / model.opt.timestep))
    n_steps = max(1, round(duration / cfg.dt_ctrl))
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
    return {
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


def _rollout_job(args) -> dict:
    return rollout(*args)


def run_grid(vx_values, vy_values, yaw_values, duration: float = 5.0,
             workers: int = 1, out: str = "batch_results.csv"):
    """笛卡尔积展开速度网格，逐组评测并写 CSV。"""
    combos = [
        (vx, vy, yaw, duration)
        for vx, vy, yaw in itertools.product(vx_values, vy_values, yaw_values)
    ]

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_rollout_job, combos))
    else:
        rows = [rollout(*combo) for combo in combos]

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'vx':>5} {'vy':>5} {'yaw':>5} {'dx':>7} {'mean_vx':>8} "
          f"{'h_min':>6} {'roll':>6} {'pitch':>6}  fell")
    for r in rows:
        print(f"{r['vx']:5.2f} {r['vy']:5.2f} {r['yaw']:5.2f} {r['dx']:7.2f} "
              f"{r['mean_vx']:8.2f} {r['h_min']:6.3f} {r['roll_max']:6.1f} "
              f"{r['pitch_max']:6.1f}  {'yes' if r['fell'] else 'no'}")

    ok = sum(not r["fell"] for r in rows)
    print(f"\n{ok}/{len(rows)} 组未跌倒，结果已写入 {out}")
    return rows