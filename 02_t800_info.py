"""T800 模型信息打印：用 MuJoCo 加载 XML 并统计模型规模。

用法:
    python3 02_t800_info.py
"""

from pathlib import Path

import mujoco

# 相对脚本所在目录定位 XML，避免依赖运行时的工作目录
XML_PATH = (
    Path(__file__).resolve().parent
    / "engineai_robotics_native_sdk/assets/resource/t800.xml"
)


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))

    print(f"模型文件: {XML_PATH}")
    print(f"身体部件数量 (nbody):  {model.nbody}")
    print(f"关节数量 (njnt):       {model.njnt}")
    print(f"电机数量 (nu):         {model.nu}")


if __name__ == "__main__":
    main()
