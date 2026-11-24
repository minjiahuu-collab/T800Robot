"""T800 机器人信息打印：整体名称/质量/尺寸 + 各连杆明细。

用法:
    python3 06_t800_links.py
"""

import unicodedata
from pathlib import Path
from xml.etree import ElementTree

import mujoco
import numpy as np

XML_PATH = (
    Path(__file__).resolve().parent
    / "engineai_robotics_native_sdk/assets/resource/t800.xml"
)


def pad(text, width):
    """按终端显示宽度补空格（中文字符占 2 列），保证中文列对齐。"""
    shown = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(0, width - shown)


def body_name(model, bid):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or "(unnamed)"


def robot_size(model, data):
    """用所有机器人 geom 的 AABB 包围盒估算整体尺寸（跳过 world 中的地面）。"""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0:  # 地面属于 world，不计入机器人
            continue
        center, half = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
        rot = data.geom_xmat[g].reshape(3, 3)
        # geom 局部 AABB 的 8 个角点变换到世界系
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    corner = center + np.array([sx, sy, sz]) * half
                    world = data.geom_xpos[g] + rot @ corner
                    lo = np.minimum(lo, world)
                    hi = np.maximum(hi, world)
    return hi - lo


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    model_name = ElementTree.parse(XML_PATH).getroot().get("model")
    size = robot_size(model, data)

    print("=" * 52)
    print("T800 机器人信息")
    print("-" * 52)
    print(f"{pad('名称', 10)}: {model_name}")
    print(f"{pad('模型文件', 10)}: {XML_PATH.name}")
    print(f"{pad('连杆数量', 10)}: {model.nbody} (含 world)")
    print(f"{pad('关节数量', 10)}: {model.njnt}")
    print(f"{pad('总质量', 10)}: {model.body_mass.sum():.3f} kg")
    print(
        f"{pad('外形尺寸', 10)}: 长 {size[0]:.3f} m × 宽 {size[1]:.3f} m "
        f"× 高 {size[2]:.3f} m"
    )
    print("=" * 52)

    header = (
        f"{pad('索引', 6)} {pad('名称', 24)} {pad('父连杆', 18)} "
        f"{pad('质量(kg)', 10)} {pad('相对父系位置(m)', 26)}"
    )
    print(f"\n{header.rstrip()}")
    for i in range(model.nbody):
        name = body_name(model, i)
        parent = body_name(model, model.body_parentid[i]) if i != 0 else "-"
        px, py, pz = model.body_pos[i]
        print(
            f"{i:>4} {name:<24} {parent:<18} "
            f"{model.body_mass[i]:>10.3f}   ({px:6.3f}, {py:6.3f}, {pz:6.3f})"
        )


if __name__ == "__main__":
    main()
