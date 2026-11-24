"""T800 自由度信息打印：nq / nv / nu。

用法:
    python3 03_t800_dof.py
"""

from pathlib import Path

import mujoco

XML_PATH = (
    Path(__file__).resolve().parent
    / "engineai_robotics_native_sdk/assets/resource/t800.xml"
)


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))

    print(f"nq (广义位置维度): {model.nq}")
    print(f"nv (广义速度维度): {model.nv}")
    print(f"nu (电机数量):     {model.nu}")


if __name__ == "__main__":
    main()
