"""自动追着木块走：世界系 +x 走廊规划 + 脚靠近时给木块踢击冲量。"""

import math

import mujoco
import numpy as np

from .. import loop
from ..config import RobotConfig
from ..control.policy import RlController
from ..model import build_model
from ..paths import CONFIG_ROOT
from ..robot import Robot
from ..scenes import add_box
from ..viewer import ViewerBackend

LOG_DT = 0.1
CMD_LPF_TAU = 0.1
SPEED_MIN = 0.15
SPEED_MAX = 0.6
FOOT_KICK_DIST = 0.22
KICK_DV = 1.5
KICK_COOLDOWN = 0.8

BOX_SIZE = 0.10
BOX_POS = (1.0, 0.12, BOX_SIZE / 2)


def _add_box(spec: mujoco.MjSpec) -> None:
    add_box(spec, "wood_box", BOX_SIZE, pos=BOX_POS, free=True, friction=(0.4, 0.01, 0.01))


def chase_command(base_xy, yaw_rad, box_xy):
    """只沿 +x 走廊追赶木块：前进速度随间距收敛，横向速度对齐木块。

    指令以世界系规划、再旋到机体系，这样机头有偏航时仍能沿世界 +x 走，
    避免绕圈；木块被踢飞后机器人会稳定地跟上去。
    """
    delta = box_xy - base_xy
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        return np.zeros(3), 0.0

    x_gap, y_err = delta[0], delta[1]

    vx_world = float(np.clip(0.9 * x_gap, SPEED_MIN, SPEED_MAX))
    vy_world = float(np.clip(1.2 * y_err, -0.4, 0.4))

    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    vx_body = c * vx_world + s * vy_world
    vy_body = -s * vx_world + c * vy_world

    yaw_err = math.atan2(math.sin(yaw_rad), math.cos(yaw_rad))
    yaw_cmd = float(np.clip(-1.2 * yaw_err, -0.6, 0.6))

    return np.array([vx_body, vy_body, yaw_cmd]), dist


def run() -> None:
    cfg = RobotConfig.load()
    model = build_model([_add_box])
    data = mujoco.MjData(model)
    robot = Robot(model, data)
    ctrl = RlController(cfg, robot, CONFIG_ROOT / cfg.policy_file)

    box_bid = robot.body_id("wood_box")
    box_jid = model.body_jntadr[box_bid]
    box_qadr = model.jnt_qposadr[box_jid]
    box_dof = model.jnt_dofadr[box_jid]
    box_mass = model.body_mass[box_bid]

    robot.set_joint_pose(cfg.default_q)
    robot.forward()
    robot.place_on_ground(skip_bodies=(box_bid,))

    n_sub = max(1, round(cfg.dt_ctrl / model.opt.timestep))
    alpha = min(1.0, cfg.dt_ctrl / CMD_LPF_TAU)
    throttle = loop.LogThrottle(LOG_DT)

    foot_bids = [robot.body_id("LINK_FOOT_L"), robot.body_id("LINK_FOOT_R")]
    cmd = np.zeros(3)
    cmd_target = np.zeros(3)
    last_kick = -1.0
    kick_count = 0
    dist = 0.0

    print(f"策略: {cfg.policy_file}")
    print(f"输入: {cfg.num_obs * cfg.n_hist + 3}, 动作: {len(cfg.active_joint_names)}, "
          f"控制: {1 / cfg.dt_ctrl:.0f} Hz")
    print(f"木块: 边长 {BOX_SIZE * 100:.0f} cm, 质量 {box_mass:.3f} kg, "
          f"初始位置 {np.round(data.qpos[box_qadr:box_qadr + 3], 3)}")
    print("自动跟随：机器人朝木块方向行走并持续推踢；关闭窗口退出")
    print()

    def step(view: ViewerBackend) -> None:
        nonlocal cmd, last_kick, kick_count, dist

        base_xy = data.qpos[0:2]
        box_xy = data.qpos[box_qadr:box_qadr + 2]
        rot = robot.rotation()
        yaw_rad = math.atan2(rot[1, 0], rot[0, 0])
        cmd_target[:], dist = chase_command(base_xy, yaw_rad, box_xy)

        # 脚靠近木块且木块在前方时，给木块一个向前的速度增量（模拟踢击）
        if data.time - last_kick > KICK_COOLDOWN:
            x_gap = box_xy[0] - base_xy[0]
            foot_dist = min(
                float(np.linalg.norm(data.xpos[b][:2] - box_xy)) for b in foot_bids
            )
            if foot_dist < FOOT_KICK_DIST and x_gap > -0.05:
                data.qvel[box_dof] += KICK_DV
                data.qvel[box_dof + 2] += 0.3
                last_kick = data.time
                kick_count += 1

        cmd += (cmd_target - cmd) * alpha
        data.ctrl[:] = ctrl.act(cmd)
        for _ in range(n_sub):
            mujoco.mj_step(model, data)

        if throttle.due(data.time):
            pos = data.qpos[0:3]
            rpy = robot.rpy()
            print(f"[t={data.time:7.3f}s] "
                  f"cmd=({cmd[0]:+.2f},{cmd[1]:+.2f}) dist={dist:.2f}m "
                  f"pos=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.3f})m "
                  f"rpy=({rpy[0]:+5.1f},{rpy[1]:+5.1f},{rpy[2]:+6.1f})deg "
                  f"v=({data.qvel[0]:+.2f},{data.qvel[1]:+.2f})m/s "
                  f"box_x={box_xy[0]:+.2f}m kicks={kick_count} "
                  f"推理={ctrl.last_ms:.2f}ms")

            view.set_texts([
                (mujoco.mjtFontScale.mjFONTSCALE_150,
                 mujoco.mjtGridPos.mjGRID_TOPLEFT,
                 "T800 chase & kick the box",
                 f"cmd vx={cmd[0]:+.2f} vy={cmd[1]:+.2f}"),
                (mujoco.mjtFontScale.mjFONTSCALE_100,
                 mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                 f"dist={dist:.2f}m  box_x={box_xy[0]:+.2f}m",
                 f"h={data.qpos[2]:.2f}m  vx={data.qvel[0]:+.2f}m/s"),
            ])

    with ViewerBackend(model, data) as view:
        view.set_camera(distance=4.5, azimuth=135.0, elevation=-10.0, lookat=[0.5, 0.0, 0.7])
        loop.run(step, view, cfg.dt_ctrl)