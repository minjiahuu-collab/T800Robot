"""T800 控制限幅验证：给所有电机输入 10000，对比真实生效的力矩。

ctrllimited 的电机在内部会被裁剪到 ctrlrange，从 data.actuator_force 可看到实际值。

用法:
    python3 14_t800_ctrl_limit.py
"""

from pathlib import Path

import mujoco
import numpy as np

XML_PATH = (
    Path(__file__).resolve().parent
    / "engineai_robotics_native_sdk/assets/resource/t800.xml"
)

CTRL_INPUT = 10000.0


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    # 直接给所有执行器一个远超 ctrlrange 的控制量
    data.ctrl[:] = CTRL_INPUT
    mujoco.mj_forward(model, data)  # 计算 actuator_force（此时已按 ctrlrange 裁剪）

    print(f"输入控制量: {CTRL_INPUT:.0f}")
    print(f"{'索引':>4} {'电机名称':<30} {'输入':>8} {'实际力矩':>10} {'是否限幅':>8}")

    clipped = 0
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        raw = data.ctrl[i]
        force = data.actuator_force[i]
        is_clipped = not np.isclose(raw, force)
        clipped += int(is_clipped)
        print(f"{i:>4} {name:<30} {raw:8.1f} {force:10.1f} {'是' if is_clipped else '否':>8}")

    print(f"\n共 {model.nu} 个电机，其中 {clipped} 个被限幅")
    # 与理论裁剪值核对
    expected = np.clip(data.ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    print(f"实际力矩与 ctrlrange 裁剪结果一致: {np.allclose(data.actuator_force, expected)}")


if __name__ == "__main__":
    main()
