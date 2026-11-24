"""T800 关键状态参数打印：nq / nv / nu 及 qpos / qvel / ctrl 的初始内容。

用法:
    python3 10_t800_state.py
"""

from pathlib import Path

import mujoco
import numpy as np

XML_PATH = (
    Path(__file__).resolve().parent
    / "engineai_robotics_native_sdk/assets/resource/t800.xml"
)


def show(name, arr):
    print(f"{name} (数量 {len(arr)}):")
    print(np.array2string(arr, precision=4, suppress_small=False, max_line_width=100))
    print()


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)  # 此时 qpos = model.qpos0, qvel/ctrl 全 0

    print(f"nq (广义位置维度): {model.nq}")
    print(f"nv (广义速度维度): {model.nv}")
    print(f"nu (电机数量):     {model.nu}")
    print()

    show("qpos 初始内容", data.qpos)
    show("qvel 初始内容", data.qvel)
    show("ctrl 初始内容", data.ctrl)


if __name__ == "__main__":
    main()
