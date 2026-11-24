"""T800 仿真过程监控：viewer 展示 + 每 100 ms 打印基座位姿（pos / rpy）。

用法:
    python3 11_t800_viewer_state.py
关闭窗口或跑满 10 s 后退出。
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

PRINT_INTERVAL = 0.1  # 打印间隔（秒）
DURATION = 10.0       # 总打印时长（秒，仿真时间）


def quat_to_rpy(quat):
    """四元数 (w, x, y, z) 转 ZYX 欧拉角 (roll, pitch, yaw)，单位 rad。"""
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    r = mat.reshape(3, 3)
    roll = np.arctan2(r[2, 1], r[2, 2])
    pitch = np.arcsin(np.clip(-r[2, 0], -1.0, 1.0))
    yaw = np.arctan2(r[1, 0], r[0, 0])
    return roll, pitch, yaw


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    # 第 0 个关节是浮动基座 freejoint，qpos 里前 7 维是位姿
    adr = model.jnt_qposadr[0]

    print(f"{'仿真时间(s)':>10} "
          f"{'pos x':>8} {'pos y':>8} {'pos z':>8} "
          f"{'roll(deg)':>10} {'pitch(deg)':>11} {'yaw(deg)':>9}")

    def log_state():
        pos = data.qpos[adr:adr + 3]
        roll, pitch, yaw = quat_to_rpy(data.qpos[adr + 3:adr + 7])
        print(
            f"{data.time:10.3f} "
            f"{pos[0]:8.4f} {pos[1]:8.4f} {pos[2]:8.4f} "
            f"{np.degrees(roll):10.2f} {np.degrees(pitch):11.2f} "
            f"{np.degrees(yaw):9.2f}"
        )

    log_state()  # t = 0 的初始位姿

    with mujoco.viewer.launch_passive(model, data) as viewer:
        next_print_time = PRINT_INTERVAL
        while viewer.is_running() and data.time < DURATION:
            step_start = time.time()
            mujoco.mj_step(model, data)

            if data.time + 1e-9 >= next_print_time:
                log_state()
                next_print_time += PRINT_INTERVAL

            viewer.sync()

            # 按真实时间节奏推进仿真
            remain = model.opt.timestep - (time.time() - step_start)
            if remain > 0:
                time.sleep(remain)


if __name__ == "__main__":
    main()
