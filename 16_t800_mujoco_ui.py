"""T800 + MuJoCo 自带 viewer/simulate 界面（无任何外部 GUI / 键盘控制）。

界面分为左右两块内置面板：
    右侧面板（Janus UI）包含 Control / Joint / Equality / Tendon / Actuator /
    Sensor / Physics / Rendering / Visualization / Group 等折叠分组，
    其中 Control 分组就是 25 个电机的默认控制滑块，拖动即可给力矩指令；
    Joint 分组是关节角滑块，Equality 分组是约束相关信息。
    左侧面板（Simulate UI）是 Pause / Reset / 仿真速度等运行控制。

鼠标：左键旋转、右键平移、滚轮缩放，双击可选中物体。
关闭窗口或按 Esc 退出。

用法:
    python3 16_t800_mujoco_ui.py
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

    # launch 阻塞运行：内部自带仿真循环、渲染和两块内置控制面板
    # show_left_ui  = 左侧 Simulate 面板（暂停/重置/速度）
    # show_right_ui = 右侧 Janus 面板（Control / Joint / Equality 等滑块分组）
    mujoco.viewer.launch(model, data, show_left_ui=True, show_right_ui=True)


if __name__ == "__main__":
    main()
