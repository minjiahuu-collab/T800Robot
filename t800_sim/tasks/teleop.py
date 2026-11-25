"""WASD 锁存遥控：方向指令锁存 + 一阶低通 + RL 行走策略。

按键:
    W 前进 / S 后退 / A 左移 / D 右移（锁存）
    R 指令清零，原地站立
"""

import mujoco
import numpy as np

from .. import loop
from ..config import RobotConfig
from ..control.policy import RlController
from ..model import build_model
from ..paths import CONFIG_ROOT
from ..robot import Robot
from ..viewer import ViewerBackend

LOG_DT = 0.1
CMD_LPF_TAU = 0.1
VX_FWD = 0.6
VX_BACK = 0.4
VY_SIDE = 0.4

KEY_W, KEY_S, KEY_A, KEY_D = 87, 83, 65, 68
KEY_R = 82


def run(spec_hooks=(), title: str = "T800 WASD teleop", box_size=None) -> None:
    """spec_hooks 用于在编译前改模型（例如把木块挂到手腕上）。"""
    cfg = RobotConfig.load()
    model = build_model(spec_hooks)
    data = mujoco.MjData(model)
    robot = Robot(model, data)
    ctrl = RlController(cfg, robot, CONFIG_ROOT / cfg.policy_file)

    robot.set_joint_pose(cfg.default_q)
    robot.forward()
    robot.place_on_ground()

    box_bid = robot.body_id("wood_box")  # 没有则 -1
    n_sub = max(1, round(cfg.dt_ctrl / model.opt.timestep))
    alpha = min(1.0, cfg.dt_ctrl / CMD_LPF_TAU)
    throttle = loop.LogThrottle(LOG_DT)

    cmd_target = np.zeros(3)
    cmd = cmd_target.copy()

    print(f"策略: {cfg.policy_file}")
    print(f"输入: {cfg.num_obs * cfg.n_hist + 3}, 动作: {len(cfg.active_joint_names)}, "
          f"控制: {1 / cfg.dt_ctrl:.0f} Hz")
    if box_bid >= 0 and box_size is not None:
        parent = model.body(model.body_parentid[box_bid]).name
        print(f"木块: 边长 {box_size * 100:.0f} cm, 质量 {model.body_mass[box_bid]:.3f} kg, "
              f"挂在 {parent} 上")
    print("W: 前进  S: 后退  A: 左移  D: 右移  R: 停下（点一下窗口获取焦点）")
    print()

    x0 = data.qpos[0]

    def step(view: ViewerBackend) -> None:
        nonlocal cmd

        while (key := view.poll_key()) is not None:
            if key == KEY_W:
                cmd_target[:] = [VX_FWD, 0.0, 0.0]
            elif key == KEY_S:
                cmd_target[:] = [-VX_BACK, 0.0, 0.0]
            elif key == KEY_A:
                cmd_target[:] = [0.0, VY_SIDE, 0.0]
            elif key == KEY_D:
                cmd_target[:] = [0.0, -VY_SIDE, 0.0]
            elif key == KEY_R:
                cmd_target[:] = 0.0

        cmd += (cmd_target - cmd) * alpha
        data.ctrl[:] = ctrl.act(cmd)
        for _ in range(n_sub):
            mujoco.mj_step(model, data)

        if throttle.due(data.time):
            pos = data.qpos[0:3]
            rpy = robot.rpy()
            extra = ""
            if box_bid >= 0:
                box_pos = data.xpos[box_bid]
                extra = f"box=({box_pos[0]:+.2f},{box_pos[1]:+.2f},{box_pos[2]:+.2f})m "
            print(f"[t={data.time:7.3f}s] "
                  f"cmd=({cmd[0]:+.2f},{cmd[1]:+.2f}) "
                  f"pos=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.3f})m "
                  f"rpy=({rpy[0]:+5.1f},{rpy[1]:+5.1f},{rpy[2]:+6.1f})deg "
                  f"v=({data.qvel[0]:+.2f},{data.qvel[1]:+.2f})m/s "
                  f"{extra}"
                  f"推理={ctrl.last_ms:.2f}ms")

            view.set_texts([
                (mujoco.mjtFontScale.mjFONTSCALE_150,
                 mujoco.mjtGridPos.mjGRID_TOPLEFT,
                 title,
                 f"cmd vx={cmd[0]:+.2f} vy={cmd[1]:+.2f}"),
                (mujoco.mjtFontScale.mjFONTSCALE_100,
                 mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                 "W fwd  S back  A left  D right  R stop",
                 f"h={data.qpos[2]:.2f}m  x={data.qpos[0] - x0:+.2f}m"),
            ])

    with ViewerBackend(model, data) as view:
        view.set_camera(distance=4.0, azimuth=135.0, elevation=-10.0, lookat=[0.0, 0.0, 0.8])
        loop.run(step, view, cfg.dt_ctrl)
