"""T800 键盘关节控制：在 MuJoCo viewer 窗口内调节 25 个关节。

操作（焦点放在 MuJoCo 窗口上）:
    A / D   选择上一个 / 下一个关节
    W / S   选中的关节目标角 +0.05 / -0.05 rad（按住可连发）
    R       所有关节回到初始角
    Ctrl + 鼠标右键拖动   直接对连杆施加外力（viewer 内置扰动）

关节用 PD 力矩跟踪目标角；浮动基座另加 PD 把它吊在原地，
否则机器人会直接摔到地上，不便观察关节动作。

用法:
    python3 15_t800_keyboard_joints.py
"""

import collections
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

XML_PATH = (
    Path(__file__).resolve().parent
    / "engineai_robotics_native_sdk/assets/resource/t800.xml"
)

# 关节 PD 与基座 PD
KP, KD = 200.0, 2.0
BASE_KP_POS, BASE_KD_POS = 4000.0, 400.0
BASE_KP_ROT, BASE_KD_ROT = 800.0, 80.0

STEP = 0.05  # 每次按键的关节增量（rad）

# GLFW 键码
KEY_A, KEY_D, KEY_W, KEY_S, KEY_R = 65, 68, 87, 83, 82


def quat_error(q_des, q_cur):
    neg = np.zeros(4)
    mujoco.mju_negQuat(neg, q_cur)
    err = np.zeros(4)
    mujoco.mju_mulQuat(err, q_des, neg)
    return err if err[0] >= 0 else -err


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    jnt = model.actuator_trnid[:, 0]
    qadr = model.jnt_qposadr[jnt]
    dadr = model.jnt_dofadr[jnt]
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j)) or f"joint{j}"
        for j in jnt
    ]

    q_init = data.qpos[qadr].copy()
    q_cmd = q_init.copy()
    sel = 0

    # 关节软限位，避免按键把目标角推到限位外
    lo = np.where(model.jnt_limited[jnt] > 0, model.jnt_range[jnt, 0], -np.inf)
    hi = np.where(model.jnt_limited[jnt] > 0, model.jnt_range[jnt, 1], np.inf)

    # 浮动基座保持原位
    base_bid = int(model.jnt_bodyid[0])
    p_des = data.qpos[0:3].copy()
    quat_des = data.qpos[3:7].copy()
    total_mass = float(model.body_mass.sum())

    # viewer 的回调在渲染线程，按键先入队，主循环再消费
    pending = collections.deque()

    def on_key(keycode):
        pending.append(keycode)

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -12.0
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]

        next_ui = 0.0
        while viewer.is_running():
            step_start = time.time()

            # 处理按键
            while pending:
                key = pending.popleft()
                if key == KEY_A:
                    sel = (sel - 1) % model.nu
                elif key == KEY_D:
                    sel = (sel + 1) % model.nu
                elif key == KEY_W:
                    q_cmd[sel] += STEP
                elif key == KEY_S:
                    q_cmd[sel] -= STEP
                elif key == KEY_R:
                    q_cmd[:] = q_init
                q_cmd[:] = np.clip(q_cmd, lo, hi)

            # 关节 PD
            tau = KP * (q_cmd - data.qpos[qadr]) - KD * data.qvel[dadr]
            data.ctrl[:] = np.clip(
                tau, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
            )

            # 基座 PD（抵消重力，悬停在初始位姿）
            p, v = data.qpos[0:3], data.qvel[0:3]
            force = BASE_KP_POS * (p_des - p) - BASE_KD_POS * v
            force[2] += total_mass * 9.81
            qe = quat_error(quat_des, data.qpos[3:7])
            torque = 2 * BASE_KP_ROT * qe[1:] - BASE_KD_ROT * data.qvel[3:6]
            data.xfrc_applied[base_bid] = np.concatenate([force, torque])

            mujoco.mj_step(model, data)
            viewer.sync()

            # 窗口内 HUD
            if data.time + 1e-9 >= next_ui:
                hud = [
                    (mujoco.mjtFontScale.mjFONTSCALE_150,
                     mujoco.mjtGridPos.mjGRID_TOPLEFT,
                     "T800 joint control", "A/D: select  W/S: move  R: reset"),
                    (mujoco.mjtFontScale.mjFONTSCALE_150,
                     mujoco.mjtGridPos.mjGRID_TOPLEFT,
                     f"[{sel:02d}] {names[sel]}",
                     f"cmd {q_cmd[sel]:+.2f} rad  /  cur "
                     f"{data.qpos[qadr[sel]]:+.2f} rad"),
                ]
                hud += [
                    (mujoco.mjtFontScale.mjFONTSCALE_100,
                     mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                     f"{'>' if i == sel else ' '} {i:02d} {names[i]}",
                     f"{data.qpos[qadr[i]]:+.2f}")
                    for i in range(model.nu)
                ]
                viewer.set_texts(hud)
                next_ui += 0.1

            remain = model.opt.timestep - (time.time() - step_start)
            if remain > 0:
                time.sleep(remain)


if __name__ == "__main__":
    main()
