"""T800 可视化 Demo：用 MuJoCo viewer 在界面中展示机器人。

用法:
    python3 04_t800_viewer.py
关闭窗口或按 Esc 退出。
"""

from pathlib import Path

import mujoco
import mujoco.viewer

XML_PATH = (
    Path(__file__).resolve().parent
    / "engineai_robotics_native_sdk/assets/resource/t800.xml"
)


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    # launch 会阻塞运行，自动推进物理仿真并实时渲染
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
