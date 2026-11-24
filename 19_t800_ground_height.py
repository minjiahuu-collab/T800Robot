"""T800 初始摆放高度反推 Demo：让默认站姿的脚底正好贴地（z=0）。

步骤：
  1. 把 25 个关节设为策略默认站姿，其余 qpos 保持 XML 初值
  2. mj_forward 计算所有 geom 的世界位姿
  3. 对每个机器人 geom，用局部 AABB 的 8 个角点变换到世界系，求最低 z
  4. 取全局最低点，基座 z 加上 -最低点，即把脚底平移到地面
  5. 再 mj_forward 验证最低点 ≈ 0

用法:
    python3 19_t800_ground_height.py
"""

from pathlib import Path

import mujoco
import numpy as np
import yaml

SDK_DIR = Path(__file__).resolve().parent / "engineai_robotics_native_sdk"
CONFIG_ROOT = SDK_DIR / "assets/config/t800"
CFG_PATH = CONFIG_ROOT / "rl_walking_example/default.yaml"
XML_PATH = SDK_DIR / "assets/resource/t800.xml"


def geom_lowest_z(model, data, g):
    """单个 geom 的世界系最低 z（局部 AABB 8 角点 -> 世界）。"""
    center = model.geom_aabb[g][:3]
    size = model.geom_aabb[g][3:]
    rot = data.geom_xmat[g].reshape(3, 3)
    zs = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                corner = center + size * np.array([sx, sy, sz])
                zs.append((rot @ corner + data.geom_xpos[g])[2])
    return min(zs)


def main():
    cfg = yaml.safe_load(CFG_PATH.read_text())
    default_q = np.concatenate(
        [np.atleast_1d(np.asarray(g, dtype=float)) for g in cfg["default_joint_q"]]
    )

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    joint_ids = [model.actuator_trnid[u, 0] for u in range(model.nu)]
    qpos_ids = np.array([model.jnt_qposadr[j] for j in joint_ids])

    print("=" * 78)
    print("步骤 1：关节设为默认站姿")
    print("=" * 78)
    print(f"基座 z（XML 初值, qpos[2]）: {data.qpos[2]:.6f} m")
    print(f"默认关节角前 6 个: {np.round(default_q[:6], 4)}")
    data.qpos[qpos_ids] = default_q
    mujoco.mj_forward(model, data)

    print()
    print("=" * 78)
    print("步骤 2：逐 geom 计算世界系最低点（局部 AABB 中心±尺寸，共 8 个角点）")
    print("=" * 78)
    print(f"{'geom 名称':<24} {'所属 body':<22} {'局部中心':<22} {'局部尺寸':<22} {'世界最低 z':>10}")
    records = []
    for g in range(model.ngeom):
        bid = model.geom_bodyid[g]
        if bid == 0:  # 跳过 world（地面）
            continue
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        zmin = geom_lowest_z(model, data, g)
        records.append((zmin, g, gname, bname))
        c = model.geom_aabb[g][:3]
        s = model.geom_aabb[g][3:]
        print(f"{gname:<24} {bname:<22} "
              f"({c[0]:+.4f},{c[1]:+.4f},{c[2]:+.4f})   "
              f"({s[0]:.4f},{s[1]:.4f},{s[2]:.4f})      {zmin:+9.5f}")

    records.sort()
    zmin, g_min, gname_min, bname_min = records[0]
    print()
    print("=" * 78)
    print("步骤 3：取全局最低点并反推基座高度")
    print("=" * 78)
    print(f"最低点的 geom: {gname_min} (body: {bname_min})，世界最低 z = {zmin:+.6f} m")
    print(f"其他较低的几个 geom（按 z 从小到大）:")
    for z, _, gn, bn in records[1:4]:
        print(f"    {gn:<24} {bn:<22} z = {z:+.5f} m")

    dz = -zmin
    z_before = data.qpos[2]
    data.qpos[2] += dz
    print()
    print(f"需要的垂直偏移 dz = -zmin = {dz:+.6f} m")
    print(f"基座高度: {z_before:.6f} -> {data.qpos[2]:.6f} m")

    print()
    print("=" * 78)
    print("步骤 4：验证")
    print("=" * 78)
    mujoco.mj_forward(model, data)
    z_check = geom_lowest_z(model, data, g_min)
    print(f"调整后基座高度 qpos[2] = {data.qpos[2]:.6f} m")
    print(f"调整后最低 geom 的世界 z = {z_check:+.6f} m（应约等于 0）")
    print(f"误差 = {abs(z_check):.2e} m")


if __name__ == "__main__":
    main()
