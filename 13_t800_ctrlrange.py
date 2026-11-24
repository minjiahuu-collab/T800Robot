"""T800 电机 ctrlrange 打印：每个执行器的控制范围及其驱动关节。

用法:
    python3 13_t800_ctrlrange.py
"""

import unicodedata
from pathlib import Path

import mujoco

XML_PATH = (
    Path(__file__).resolve().parent
    / "engineai_robotics_native_sdk/assets/resource/t800.xml"
)


def pad(text, width):
    """按终端显示宽度补空格（中文占 2 列），保证表头与数据对齐。"""
    shown = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(0, width - shown)


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))

    print(f"电机数量: {model.nu}")
    print(
        f"{pad('索引', 6)} {pad('电机名称', 30)} {pad('驱动关节', 24)} "
        f"{pad('ctrlrange', 20)} 限位"
    )

    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "(unnamed)"
        jnt = model.actuator_trnid[i, 0]
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt) or "(unnamed)"
        lo, hi = model.actuator_ctrlrange[i]
        limit = "有限位" if model.actuator_ctrllimited[i] else "无限位"
        print(f"{i:>4} {name:<30} {jname:<24} [{lo:7.1f}, {hi:6.1f}] {limit}")


if __name__ == "__main__":
    main()
