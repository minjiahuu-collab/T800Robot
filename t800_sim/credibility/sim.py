"""可信度检查共用的确定性 rollout（无渲染，逐控制步记录状态）。

与采集任务共用同一套初始化（默认关节角、贴地、seed 初始位姿噪声），
但只做物理步进，便于复跑比对和接触检查。
"""

import mujoco
import numpy as np

from ..config import RobotConfig
from ..control.policy import RlController
from ..model import build_model
from ..paths import CONFIG_ROOT
from ..robot import Robot
from ..tasks.record import BOX_NAME, DEFAULT_CMD, apply_init_noise, make_scene_hook

SLIP_SPEED = 0.3  # 足端水平滑移速度阈值 (m/s)


def rollout(
    duration: float = 2.0,
    cmd=DEFAULT_CMD,
    scene: str = "box",
    seed: int = 0,
    episode: int = 0,
    init_noise=(0.0, 0.0, 0.0),
    timestep=None,
    contacts: bool = False,
) -> dict:
    """跑一段固定指令，返回逐控制步的 qpos/qvel/ctrl 与可选接触统计。"""
    cfg = RobotConfig.load()
    hooks = []
    scene_hook = make_scene_hook(scene)
    if scene_hook is not None:
        hooks.append(scene_hook)
    model = build_model(hooks)
    if timestep is not None:
        model.opt.timestep = float(timestep)

    data = mujoco.MjData(model)
    robot = Robot(model, data)
    ctrl = RlController(cfg, robot, CONFIG_ROOT / cfg.policy_file)

    robot.set_joint_pose(cfg.default_q)
    robot.forward()
    skip_bodies = ()
    if scene == "box":
        skip_bodies = (robot.body_id(BOX_NAME),)
    robot.place_on_ground(skip_bodies=skip_bodies)

    rng = np.random.default_rng([int(seed), int(episode)])
    apply_init_noise(robot, rng, init_noise)

    n_sub = max(1, round(cfg.dt_ctrl / model.opt.timestep))
    n_steps = max(1, round(duration / cfg.dt_ctrl))
    cmd = np.asarray(cmd, dtype=float)

    foot_bids = [
        bid for bid in (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "LINK_FOOT_L"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "LINK_FOOT_R"),
        ) if bid >= 0
    ]
    prev_feet = [data.xpos[bid].copy() for bid in foot_bids]

    qpos, qvel, ctrl_hist = [], [], []
    stats = {
        "max_penetration": 0.0,
        "max_penetration_pair": "",
        "self_collisions": 0,
        "foot_contact_steps": 0,
        "foot_slip_steps": 0,
    }

    for _ in range(n_steps):
        data.ctrl[:] = ctrl.act(cmd)
        for _ in range(n_sub):
            mujoco.mj_step(model, data)
        qpos.append(data.qpos.copy())
        qvel.append(data.qvel.copy())
        ctrl_hist.append(np.asarray(data.ctrl).copy())

        if contacts:
            touching = set()
            for ci in range(data.ncon):
                con = data.contact[ci]
                b1 = int(model.geom_bodyid[con.geom1])
                b2 = int(model.geom_bodyid[con.geom2])
                stats["max_penetration"] = min(stats["max_penetration"], float(con.dist))
                if float(con.dist) == stats["max_penetration"]:
                    stats["max_penetration_pair"] = "{} / {}".format(
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                                          int(con.geom1)) or f"geom{con.geom1}",
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                                          int(con.geom2)) or f"geom{con.geom2}",
                    )
                if b1 > 0 and b2 > 0 and b1 != b2:
                    stats["self_collisions"] += 1
                for bid in foot_bids:
                    if bid in (b1, b2):
                        touching.add(bid)
            for i, bid in enumerate(foot_bids):
                if bid in touching:
                    stats["foot_contact_steps"] += 1
                    moved = data.xpos[bid][:2] - prev_feet[i][:2]
                    speed = float(np.linalg.norm(moved)) / cfg.dt_ctrl
                    if speed > SLIP_SPEED:
                        stats["foot_slip_steps"] += 1
                prev_feet[i] = data.xpos[bid].copy()

    return {
        "model": model,
        "data": data,
        "robot": robot,
        "cfg": cfg,
        "cmd": cmd,
        "duration": float(duration),
        "timestep": float(model.opt.timestep),
        "n_sub": n_sub,
        "qpos": np.asarray(qpos),
        "qvel": np.asarray(qvel),
        "ctrl": np.asarray(ctrl_hist),
        "contacts": stats,
    }


def base_metrics(run: dict) -> dict:
    """从 rollout 结果提取用于收敛比对的标量指标。"""
    qpos = run["qpos"]
    dx = float(qpos[-1, 0] - qpos[0, 0])
    dy = float(qpos[-1, 1] - qpos[0, 1])
    duration = run["duration"]
    return {
        "dx": dx,
        "dy": dy,
        "mean_vx": dx / duration if duration > 0 else 0.0,
        "mean_vy": dy / duration if duration > 0 else 0.0,
        "h_min": float(qpos[:, 2].min()),
    }
