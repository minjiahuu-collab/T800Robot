"""T800 关节列表打印：遍历模型中所有关节及其基本信息。

用法:
    python3 05_t800_joints.py
"""

from pathlib import Path

import mujoco

XML_PATH = (
    Path(__file__).resolve().parent
    / "engineai_robotics_native_sdk/assets/resource/t800.xml"
)

JOINT_TYPES = {
    mujoco.mjtJoint.mjJNT_FREE: "free",
    mujoco.mjtJoint.mjJNT_BALL: "ball",
    mujoco.mjtJoint.mjJNT_SLIDE: "slide",
    mujoco.mjtJoint.mjJNT_HINGE: "hinge",
}


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))

    print(f"关节总数: {model.njnt}")
    print(f"{'索引':>4} {'名称':<28} {'类型':<6} {'qposadr':>8} {'dofadr':>7} {'范围(rad)':>22}")

    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or "(unnamed)"
        jtype = JOINT_TYPES[mujoco.mjtJoint(model.jnt_type[i])]
        lo, hi = model.jnt_range[i]
        # 自由关节没有限位，range 无意义
        range_str = f"[{lo:8.3f}, {hi:8.3f}]" if model.jnt_limited[i] else "-"
        print(
            f"{i:>4} {name:<28} {jtype:<6} "
            f"{model.jnt_qposadr[i]:>8} {model.jnt_dofadr[i]:>7} {range_str:>22}"
        )


if __name__ == "__main__":
    main()
