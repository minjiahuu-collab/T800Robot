"""T800 连杆 body_pos 打印：每个 body 相对父连杆坐标系的位置。

用法:
    python3 07_t800_body_pos.py
"""

from pathlib import Path

import mujoco

XML_PATH = (
    Path(__file__).resolve().parent
    / "engineai_robotics_native_sdk/assets/resource/t800.xml"
)


def body_name(model, bid):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or "(unnamed)"


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))

    print(f"body 总数: {model.nbody}")
    print(f"{'索引':>4} {'名称':<24} {'body_pos (x, y, z) / m':>30}")

    for i in range(model.nbody):
        x, y, z = model.body_pos[i]
        print(f"{i:>4} {body_name(model, i):<24}   ({x:7.4f}, {y:7.4f}, {z:7.4f})")


if __name__ == "__main__":
    main()
