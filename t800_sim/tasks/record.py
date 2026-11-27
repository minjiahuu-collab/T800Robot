"""数据采集：头部 + 左右手腕三路相机逐帧落盘，带物体真值与元数据。

固定速度指令驱动 RL 行走策略，按 fps 渲染三台相机并写成 PNG，同时记录
本体状态、相机位姿、物体/末端真值与指标（meta.json）。
默认 headless，不打开 MuJoCo 窗口，适合批量采集。

目录结构（task/scene/seed/ep 分层，便于检索与复现）:
    <out>/<task>/<scene>/seed_XXXX/ep_YYYY/
        meta.json  states.csv  cameras.csv  ground_truth.csv  frames/<cam>/*.png

用法:
    python3 -m t800_sim.cli record
    python3 -m t800_sim.cli record --duration 10 --fps 30 --width 640 --height 480
    python3 -m t800_sim.cli record --cmd 0 0 0.4 --scene empty --seed 3
    python3 -m t800_sim.cli record --randomize --seed 3   # 域随机化采集
    python3 -m t800_sim.cli record-grid --vx 0.3 0.6 --seeds 0 1 2
"""

import itertools
import time
from pathlib import Path

import mujoco
import numpy as np

from .. import loop
from ..capture import (
    CAMERA_NAMES,
    MultiCamRecorder,
    add_cameras,
    available_force_sensors,
    sha256_file,
)
from ..config import RobotConfig
from ..control.policy import RlController
from ..model import build_model
from ..paths import CONFIG_ROOT
from ..randomization import SensorModel, apply_sample, body_shapes, sample
from ..robot import Robot
from ..scenes import add_box

LOG_DT = 1.0
DEFAULT_CMD = (0.6, 0.0, 0.0)
BOX_NAME = "wood_box"
BOX_SIZE = 0.10
BOX_POS = (1.8, 0.8, BOX_SIZE / 2)
SUCCESS_VX_RATIO = 0.5
DEFAULT_NOISE = (0.0, 0.0, 0.0)
SCENES = ("empty", "box")


def make_scene_hook(scene: str):
    """返回往场景里加真值物体的 MjSpec hook；empty 场景返回 None。"""
    if scene == "empty":
        return None
    if scene == "box":
        def hook(spec: mujoco.MjSpec) -> None:
            add_box(spec, BOX_NAME, BOX_SIZE, pos=BOX_POS, free=True)
        return hook
    raise ValueError(f"未知场景: {scene}（可选 {', '.join(SCENES)}）")


def _session_dir(out_root, task_name, scene, seed, episode):
    return (
        out_root / task_name / scene
        / f"seed_{int(seed):04d}" / f"ep_{int(episode):04d}"
    )


def apply_init_noise(robot: Robot, rng: np.random.Generator, noise) -> None:
    """按 seed 给基座 x/y 位置和 yaw 加高斯扰动。"""
    std_x, std_y, std_yaw = (float(v) for v in noise)
    qpos = robot.data.qpos
    if std_x:
        qpos[0] += rng.normal(0.0, std_x)
    if std_y:
        qpos[1] += rng.normal(0.0, std_y)
    if std_yaw:
        dyaw = rng.normal(0.0, std_yaw)
        yaw_quat = np.array([np.cos(dyaw / 2), 0.0, 0.0, np.sin(dyaw / 2)])
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, yaw_quat, qpos[3:7])
        qpos[3:7] = out
    if std_x or std_y or std_yaw:
        robot.forward()


def run(
    duration: float = 5.0,
    fps: float = 30.0,
    width: int = 320,
    height: int = 240,
    cmd=DEFAULT_CMD,
    out_root="recordings",
    *,
    scene: str = "box",
    seed: int = 0,
    episode: int = 0,
    task_name: str = "walk",
    init_noise=DEFAULT_NOISE,
    rand_plan=None,
) -> Path:
    """采集一条 episode，返回 session 目录。"""
    cfg = RobotConfig.load()
    hooks = [add_cameras]
    scene_hook = make_scene_hook(scene)
    if scene_hook is not None:
        hooks.append(scene_hook)
    model = build_model(hooks)

    rand_sample = None
    applied = None
    rand_session = None
    dt_ctrl = cfg.dt_ctrl
    if rand_plan is not None:
        # 同一 task/scene/seed 归为一个 session：per_session 模式可固定物理参数，
        # per_episode 模式仍然每条 episode 重新采样。
        rand_session = f"{task_name}/{scene}/seed_{int(seed):04d}"
        rand_sample = sample(
            rand_plan, rand_session, episode,
            shapes=body_shapes(model, rand_plan),
        )
        apply_rng = np.random.default_rng([int(seed), int(episode), 0x0C0A])
        applied = apply_sample(
            model, rand_sample, dt_ctrl=cfg.dt_ctrl, rng=apply_rng
        )
        dt_ctrl = applied.runtime.control_dt or cfg.dt_ctrl
        if applied.runtime.control_dt is not None:
            print(f"随机化控制周期: {dt_ctrl * 1000:.1f} ms"
                  f"（配置默认 {cfg.dt_ctrl * 1000:.1f} ms）")

    data = mujoco.MjData(model)
    robot = Robot(model, data)  # ctrlrange 已在 apply_sample 中缩放

    kp, kd = (
        applied.runtime.scaled_gains(cfg.kp, cfg.kd)
        if applied is not None else (cfg.kp, cfg.kd)
    )
    sensor = (
        SensorModel(
            applied.runtime.sensor,
            dt=dt_ctrl,
            rng=np.random.default_rng([int(seed), int(episode), 0x5E05]),
        )
        if applied is not None else None
    )
    ctrl = RlController(
        cfg, robot, CONFIG_ROOT / cfg.policy_file,
        kp=kp, kd=kd, sensor=sensor,
        motor_delay_steps=applied.runtime.motor_delay_steps if applied is not None else 0,
    )

    # 力噪声用独立 RNG：采集频率与策略频率不同，不能和观测噪声共用随机流。
    force_rng = np.random.default_rng([int(seed), int(episode), 0xF0])

    def force_noise(_name, values):
        return sensor.force(values, rng=force_rng) if sensor is not None else values

    robot.set_joint_pose(cfg.default_q)
    robot.forward()

    gt_object = None
    skip_bodies = ()
    if scene == "box":
        box_bid = robot.body_id(BOX_NAME)
        skip_bodies = (box_bid,)
        gt_object = BOX_NAME
    robot.place_on_ground(skip_bodies=skip_bodies)

    rng = np.random.default_rng([int(seed), int(episode)])
    apply_init_noise(robot, rng, init_noise)

    for name in CAMERA_NAMES:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) < 0:
            raise RuntimeError(f"相机注入失败: {name}")

    cam_dropout = 0.0
    cam_latency_ms = 0.0
    cam_rng = None
    if applied is not None:
        cam_dropout = applied.runtime.sensor.dropout_prob
        cam_latency_ms = applied.runtime.sensor.latency_ms
        # 独立于本体观测（0x5E05）与力噪声（0xF0）的相机链路随机流。
        cam_rng = np.random.default_rng([int(seed), int(episode), 0xC472])

    session = _session_dir(Path(out_root), task_name, scene, seed, episode)
    cmd = np.asarray(cmd, dtype=float)
    policy_path = CONFIG_ROOT / cfg.policy_file
    rec = MultiCamRecorder(
        model, data, session,
        joint_names=robot.actuator_names,
        joint_qpos_idx=robot.qpos_idx,
        width=width, height=height, fps=fps,
        task=task_name,
        scene_id=scene,
        seed=seed,
        policy={"file": cfg.policy_file, "sha256": sha256_file(policy_path)},
        gt_object=gt_object,
        force_sensors=available_force_sensors(model),
        force_noise=force_noise,
        success_vx_ratio=SUCCESS_VX_RATIO,
        camera_dropout_prob=cam_dropout,
        camera_latency_ms=cam_latency_ms,
        camera_rng=cam_rng,
        extra_meta={
            "cmd": [float(c) for c in cmd],
            "episode": int(episode),
            "init_noise": [float(v) for v in init_noise],
            "control_dt": float(dt_ctrl),
            "randomization": (
                {
                    "enabled": True,
                    "plan": rand_sample["meta"],
                    "params": rand_sample["params"],
                    "applied": applied.as_dict(),
                }
                if applied is not None else {"enabled": False}
            ),
        },
    )
    rec.set_command(cmd)

    n_sub = max(1, round(dt_ctrl / model.opt.timestep))
    n_steps = max(1, round(duration / dt_ctrl))
    throttle = loop.LogThrottle(LOG_DT)

    print(f"场景: {scene}（真值物体: {gt_object or '无'}）, seed={seed} ep={episode}")
    print(f"相机: {', '.join(CAMERA_NAMES)} (挂在头部/左右手腕，fovy 见 meta)")
    print(f"采集: {duration:.1f} s @ {fps:.0f} fps, {width}x{height}, "
          f"指令 cmd=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f})")
    print(f"输出: {session}")
    if applied is not None:
        print(f"随机化: hash={rand_sample['meta']['plan_hash']} "
              f"session={rand_session} ep={episode} "
              f"指令延迟={applied.runtime.motor_delay_steps} 步")

    t0 = time.time()
    try:
        rec.capture_if_due(data.time)  # 起始站姿
        for _ in range(n_steps):
            data.ctrl[:] = ctrl.act(cmd)
            for _ in range(n_sub):
                mujoco.mj_step(model, data)
            rec.capture_if_due(data.time)

            if throttle.due(data.time):
                print(f"[t={data.time:6.2f}s] frame={rec.frame:4d} "
                      f"pos=({data.qpos[0]:+.2f},{data.qpos[1]:+.2f},{data.qpos[2]:.3f})m "
                      f"推理={ctrl.last_ms:.2f}ms")
    finally:
        rec.close()

    metrics = rec.meta["metrics"]
    print(f"完成: {rec.frame} 帧 x {len(CAMERA_NAMES)} 路，"
          f"用时 {time.time() - t0:.1f}s -> {session}")
    print(f"指标: mean_vx={metrics['mean_vx']:+.2f} m/s, "
          f"dx={metrics['final_dx']:+.2f} m, "
          f"fell={'yes' if metrics['fell'] else 'no'}, "
          f"success={'yes' if metrics['success'] else 'no'}")
    return session


def run_grid(
    vx_values=(0.3, 0.6),
    vy_values=(0.0,),
    yaw_values=(0.0,),
    seeds=(0, 1, 2),
    episodes: int = 1,
    duration: float = 5.0,
    fps: float = 30.0,
    width: int = 320,
    height: int = 240,
    scene: str = "box",
    init_noise=(0.1, 0.05, 0.1),
    out_root="recordings",
    rand_plan=None,
):
    """速度指令 × seed × episode 的网格采集，按 cmd 分 task 目录。"""
    sessions = []
    combos = list(itertools.product(vx_values, vy_values, yaw_values))
    for vx, vy, yaw in combos:
        task_name = f"walk_vx{vx:.2f}_vy{vy:.2f}_yaw{yaw:.2f}"
        for seed in seeds:
            for ep in range(int(episodes)):
                print(f"\n=== {task_name} seed={seed} ep={ep} "
                      f"({len(sessions) + 1}/{len(combos) * len(seeds) * int(episodes)}) ===")
                sessions.append(run(
                    duration, fps, width, height, (vx, vy, yaw), out_root,
                    scene=scene, seed=int(seed), episode=ep,
                    task_name=task_name, init_noise=init_noise,
                    rand_plan=rand_plan,
                ))
    print(f"\n网格采集完成: {len(sessions)} 条 session -> {out_root}")
    return sessions
