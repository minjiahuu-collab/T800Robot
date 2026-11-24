"""T800 连杆世界坐标打印：每个 body 的 xpos（世界坐标系位置）。

用法:
    python3 08_t800_xpos.py
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
    data = mujoco.MjData(model)

    # 按当前 qpos（默认初始姿态）做一次正运动学，填充 data.xpos
    mujoco.mj_forward(model, data)

    print(f"body 总数: {model.nbody}")
    print(f"{'索引':>4} {'名称':<24} {'xpos (x, y, z) / m':>30}")

    for i in range(model.nbody):
        x, y, z = data.xpos[i]
        print(f"{i:>4} {body_name(model, i):<24}   ({x:7.4f}, {y:7.4f}, {z:7.4f})")


if __name__ == "__main__":
    main()
