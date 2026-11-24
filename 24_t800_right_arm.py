"""T800 右臂抬起/放下演示：固定 base，保持直立，只动右臂。

- base（躯干自由关节）每个物理子步都被钉在初始位姿，机器人不会倒
- 除右臂外所有关节都用 PD 保持在默认站姿
- W: 抬起右臂（smootherstep 插补，1.5 s 平滑过渡到举起位）
- S: 放下右臂（同样平滑过渡，回到默认自然下垂位）

按键（鼠标点一下 MuJoCo 窗口获取焦点）生效。
每帧恢复 viewer 的渲染 flags，避免按 W/S 时改变界面显示效果。

用法:
    python3 24_t800_right_arm.py
关闭窗口退出。
"""

import collections
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml

SDK_DIR = Path(__file__).resolve().parent / "engineai_robotics_native_sdk"
CONFIG_ROOT = SDK_DIR / "assets/config/t800"
CFG_PATH = CONFIG_ROOT / "rl_walking_example/default.yaml"
XML_PATH = SDK_DIR / "assets/resource/t800.xml"

LOG_DT = 0.1        # 日志周期 (s)
CTRL_DT = 0.01      # 控制周期 (s)
RAMP_TIME = 1.5     # 抬臂/放臂过渡时间 (s)，越长越平缓
# 腿和躯干的阻尼放大系数：抬臂的反作用力矩会激励悬空的腿，加大阻尼抑制抖动
LEG_TORSO_KD_SCALE = 2.0
# 右臂阻尼放大系数：放臂时手臂靠重力下落，阻尼不足会长时间摆动（≥5 会诱发高频抖动）
ARM_KD_SCALE = 4.0

KEY_W, KEY_S = 87, 83

# 举起右臂的关节目标角 (rad)，以默认站姿为基准
RAISE_POSE = {
    "J18_SHOULDER_PITCH_R": -1.2,  # 肩 pitch：向前上方举臂
    "J19_SHOULDER_ROLL_R": -0.4,   # 肩 roll：略微外展，避免贴头
    "J21_ELBOW_PITCH_R": -0.8,     # 肘 pitch：屈肘
}


def load_cfg(path):
    """读取 yaml 中按肢体分组的列表，拼成一维数组。"""
    cfg = yaml.safe_load(path.read_text())

    def flat(key):
        return np.concatenate(
            [np.atleast_1d(np.asarray(g, dtype=float)) for g in cfg[key]]
        )

    return cfg, flat("default_joint_q"), flat("joint_kp"), flat("joint_kd")


def rpy_from_rot(rot):
    """旋转矩阵(ZYX) -> roll/pitch/yaw (deg)。"""
    pitch = math.asin(max(-1.0, min(1.0, -rot[2, 0])))
    roll = math.atan2(rot[2, 1], rot[2, 2])
    yaw = math.atan2(rot[1, 0], rot[0, 0])
    return np.degrees([roll, pitch, yaw])


def lowest_point(model, data):
    """机器人所有 geom 的世界系最低 z（跳过 world 里的地面）。"""
    zmin = np.inf
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0:
            continue
        center = model.geom_aabb[g][:3]
        size = model.geom_aabb[g][3:]
        rot = data.geom_xmat[g].reshape(3, 3)
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    corner = center + size * np.array([sx, sy, sz])
                    zmin = min(zmin, (rot @ corner + data.geom_xpos[g])[2])
    return zmin


def main():
    cfg, default_q, kp, kd = load_cfg(CFG_PATH)

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    all_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[u, 0])
        for u in range(model.nu)
    ]
    # 腿/躯干阻尼加倍，抑制抬臂反作用力激励出的持续小振荡；右臂阻尼加倍加快余振衰减
    kd = kd.copy()
    for i, name in enumerate(all_names):
        if any(k in name for k in ("HIP", "KNEE", "ANKLE", "TORSO")):
            kd[i] *= LEG_TORSO_KD_SCALE
        elif name.endswith("_R") and ("SHOULDER" in name or "ELBOW" in name):
            kd[i] *= ARM_KD_SCALE
    all_qpos = np.array([model.jnt_qposadr[model.actuator_trnid[u, 0]] for u in range(model.nu)])
    all_dof = np.array([model.jnt_dofadr[model.actuator_trnid[u, 0]] for u in range(model.nu)])
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]

    # 举起目标：默认角 + 右臂覆盖
    raise_q = default_q.copy()
    for name, val in RAISE_POSE.items():
        raise_q[all_names.index(name)] = val
    lower_q = default_q.copy()

    # --- 初始状态：默认站姿，脚底贴地 ---
    data.qpos[all_qpos] = default_q
    mujoco.mj_forward(model, data)
    data.qpos[2] += -lowest_point(model, data)
    mujoco.mj_forward(model, data)

    base_q = data.qpos[0:7].copy()   # 被钉住的 base 位姿
    base_bid = int(model.jnt_bodyid[0])
    wrist_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "LINK_WRIST_END_R")
    sh_idx = all_names.index("J18_SHOULDER_PITCH_R")
    el_idx = all_names.index("J21_ELBOW_PITCH_R")

    n_sub = max(1, round(CTRL_DT / model.opt.timestep))
    blend = 0.0          # 0 = 放下, 1 = 举起
    blend_target = 0.0
    q_des = default_q.copy()

    pending = collections.deque()

    def on_key(keycode):
        pending.append(keycode)

    print("T800 右臂控制：base 固定，其余关节保持站姿")
    print("W: 抬起右臂   S: 放下右臂（点一下窗口获取焦点）")
    print()

    next_log = 0.0

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        viewer.cam.distance = 3.2
        viewer.cam.azimuth = 145.0
        viewer.cam.elevation = -8.0
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]

        # 记录启动时的渲染状态，之后每帧恢复，抵消 viewer 自带快捷键的副作用
        flags0 = viewer.opt.flags.copy()
        groups0 = viewer.opt.geomgroup.copy()

        while viewer.is_running():
            t0 = time.time()

            viewer.opt.flags[:] = flags0
            viewer.opt.geomgroup[:] = groups0

            while pending:
                key = pending.popleft()
                if key == KEY_W:
                    blend_target = 1.0
                elif key == KEY_S:
                    blend_target = 0.0

            # 抬臂/放臂平滑过渡：blend 线性推进，再映射成 smootherstep 曲线，
            # 使起点/终点的速度和加速度都为 0，避免反作用力矩突变引起抖动
            step = CTRL_DT / RAMP_TIME
            if blend < blend_target:
                blend = min(blend_target, blend + step)
            elif blend > blend_target:
                blend = max(blend_target, blend - step)
            s = blend * blend * blend * (blend * (blend * 6 - 15) + 10)

            q_des[:] = lower_q + s * (raise_q - lower_q)
            tau = kp * (q_des - data.qpos[all_qpos]) - kd * data.qvel[all_dof]
            data.ctrl[:] = np.clip(tau, lo, hi)

            # 固定 base：每个子步都把自由关节位姿和速度钉住
            for _ in range(n_sub):
                data.qpos[0:7] = base_q
                data.qvel[0:6] = 0.0
                mujoco.mj_step(model, data)
            viewer.sync()

            # --- 100 ms 日志 + 窗口 HUD ---
            if data.time + 1e-9 >= next_log:
                rpy = rpy_from_rot(data.xmat[base_bid].reshape(3, 3))
                wrist = data.xpos[wrist_bid]
                state = "RAISE" if blend_target > 0.5 else "LOWER"
                print(f"[t={data.time:7.3f}s] {state} s={s:.2f} "
                      f"shoulder_pitch={data.qpos[all_qpos[sh_idx]]:+.2f}rad "
                      f"elbow_pitch={data.qpos[all_qpos[el_idx]]:+.2f}rad "
                      f"wrist=({wrist[0]:+.2f},{wrist[1]:+.2f},{wrist[2]:+.2f})m "
                      f"base=({data.qpos[0]:+.2f},{data.qpos[1]:+.2f},{data.qpos[2]:+.3f})m "
                      f"rpy=({rpy[0]:+5.1f},{rpy[1]:+5.1f},{rpy[2]:+6.1f})deg")

                viewer.set_texts([
                    (mujoco.mjtFontScale.mjFONTSCALE_150,
                     mujoco.mjtGridPos.mjGRID_TOPLEFT,
                     "T800 right arm: W = raise, S = lower",
                     f"state={state}  s={s:.2f}"),
                    (mujoco.mjtFontScale.mjFONTSCALE_100,
                     mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                     "base fixed, all other joints hold default pose",
                     f"wrist_z={wrist[2]:.2f}m  base_h={data.qpos[2]:.2f}m"),
                ])
                next_log += LOG_DT

            remain = CTRL_DT - (time.time() - t0)
            if remain > 0:
                time.sleep(remain)


if __name__ == "__main__":
    main()
