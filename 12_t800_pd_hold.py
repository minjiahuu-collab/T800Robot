"""T800 PD 控制示例：所有关节保持初始角度。

两个 PD：
1) 关节 PD（本题目标）：ctrl = kp * (q_des - q) - kd * qd，把 25 个关节锁在初始角。
2) 基座 PD（仅演示用）：机器人是浮动基座，没有平衡控制器会直接摔倒，
   这里用外力和力矩把它“吊”在初始位姿，便于观察关节 PD 的效果。

关闭窗口或跑满 20 s 退出。

用法:
    python3 12_t800_pd_hold.py
"""

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

XML_PATH = (
    Path(__file__).resolve().parent
    / "engineai_robotics_native_sdk/assets/resource/t800.xml"
)

# 关节 PD 增益（kd 不能太大：肘部偏航关节惯量小，过大会显式积分发散）
KP, KD = 5.0, 2.0

# 基座 PD 增益与仿真时长
BASE_KP_POS, BASE_KD_POS = 4000.0, 400.0
BASE_KP_ROT, BASE_KD_ROT = 800.0, 80.0
DURATION = 20.0


def quat_error(q_des, q_cur):
    """返回四元数姿态误差 q_des * q_cur^-1（w, x, y, z）。"""
    neg = np.zeros(4)
    mujoco.mju_negQuat(neg, q_cur)
    err = np.zeros(4)
    mujoco.mju_mulQuat(err, q_des, neg)
    return err if err[0] >= 0 else -err


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    # 电机 -> 关节 -> qpos / qvel 地址的映射
    jnt = model.actuator_trnid[:, 0]
    qadr = model.jnt_qposadr[jnt]
    dadr = model.jnt_dofadr[jnt]
    q_des = data.qpos[qadr].copy()  # 目标：初始关节角（全 0）

    # 浮动基座
    base_bid = int(model.jnt_bodyid[0])
    p_des = data.qpos[0:3].copy()
    quat_des = data.qpos[3:7].copy()
    total_mass = float(model.body_mass.sum())

    print(f"电机数量: {model.nu}, 关节 PD: kp={KP}, kd={KD}")
    print(f"{'仿真时间(s)':>10} {'基座高度(m)':>12} {'最大关节误差(rad)':>18}")

    next_print = 0.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < DURATION:
            step_start = time.time()

            # 1) 关节 PD：输出力矩并限幅到电机的 ctrlrange
            q = data.qpos[qadr]
            qd = data.qvel[dadr]
            tau = KP * (q_des - q) - KD * qd
            data.ctrl[:] = np.clip(
                tau, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
            )

            # 2) 基座 PD：抵消总重力 + 位姿反馈，让机器人悬停在初始位置
            p, v = data.qpos[0:3], data.qvel[0:3]
            force = BASE_KP_POS * (p_des - p) - BASE_KD_POS * v
            force[2] += total_mass * 9.81
            qe = quat_error(quat_des, data.qpos[3:7])
            torque = 2 * BASE_KP_ROT * qe[1:] - BASE_KD_ROT * data.qvel[3:6]
            data.xfrc_applied[base_bid] = np.concatenate([force, torque])

            mujoco.mj_step(model, data)
            viewer.sync()

            if data.time + 1e-9 >= next_print:
                max_err = np.max(np.abs(data.qpos[qadr] - q_des))
                print(f"{data.time:10.2f} {data.qpos[2]:12.4f} {max_err:18.5f}")
                next_print += 1.0

            # 按真实时间节奏推进仿真
            remain = model.opt.timestep - (time.time() - step_start)
            if remain > 0:
                time.sleep(remain)

    print(f"结束，最大关节误差: {np.max(np.abs(data.qpos[qadr] - q_des)):.5f} rad")


if __name__ == "__main__":
    main()
