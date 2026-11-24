"""T800 自动追着木块走：机器人朝木块方向行走并踢着 10 cm 木块前进。

与 21 的差别：不再使用按键锁存的速度指令，而是每个控制周期根据
"机器人 -> 木块" 的相对位置自动生成速度指令：
  - 世界系 +x 走廊：前进速度随前后间距收敛，横向速度对齐木块
  - 指令先在世界系规划，再旋到机体系；同时用 yaw 指令保持机头朝向 +x
  - 当任一只脚靠近木块时，给木块一个向前的速度增量（模拟踢击），
    木块被踢飞后机器人会继续跟上，形成"追着踢"的循环

木块边长 10 cm，带 freejoint，通过 MjSpec 在运行时注入，不改 SDK 文件。

用法:
    python3 22_t800_chase_box.py
关闭窗口退出。
"""

import math
import time
from pathlib import Path

import MNN
import mujoco
import mujoco.viewer
import numpy as np
import yaml

SDK_DIR = Path(__file__).resolve().parent / "engineai_robotics_native_sdk"
CONFIG_ROOT = SDK_DIR / "assets/config/t800"
CFG_PATH = CONFIG_ROOT / "rl_walking_example/default.yaml"
XML_PATH = SDK_DIR / "assets/resource/t800.xml"

LOG_DT = 0.1         # 日志周期 (s)
CMD_LPF_TAU = 0.1    # 指令平滑时间常数 (s)
SPEED_MIN = 0.15     # 贴近木块时的最低前进速度 (m/s)
SPEED_MAX = 0.6      # 距离较远时的追赶速度 (m/s)
FOOT_KICK_DIST = 0.22  # 脚离木块多近时判定为"踢到" (m)
KICK_DV = 1.5        # 踢击给木块的水平速度增量 (m/s)
KICK_COOLDOWN = 0.8  # 两次踢击的最小间隔 (s)

BOX_SIZE = 0.10                         # 木块边长 (m)
BOX_POS = (1.0, 0.12, BOX_SIZE / 2)     # 放在左腿行进线上，脚能真正踢到


class Policy:
    def __init__(self, path):
        self.interp = MNN.Interpreter(str(path))
        self.sess = self.interp.createSession({"backend": "CPU", "numThread": 4})
        self.input = self.interp.getSessionInput(self.sess)
        self.output = self.interp.getSessionOutput(self.sess)
        self.last_ms = 0.0

    def __call__(self, obs):
        inp = MNN.Tensor(
            (1, obs.size),
            MNN.Halide_Type_Float,
            obs.astype(np.float32),
            MNN.Tensor_DimensionType_Caffe,
        )
        self.input.copyFrom(inp)
        t0 = time.perf_counter()
        self.interp.runSession(self.sess)
        self.last_ms = (time.perf_counter() - t0) * 1e3
        # 注意: 该 MNN 版本 copyToHostTensor 取不到结果，直接读主机内存
        return np.array(self.output.getData(), dtype=np.float32).ravel()


def build_model():
    """加载官方 XML 并在运行时插入 10 cm 木块，返回编译后的 model。"""
    spec = mujoco.MjSpec.from_file(str(XML_PATH))

    body = spec.worldbody.add_body()
    body.name = "wood_box"
    body.pos = list(BOX_POS)
    body.add_freejoint()

    geom = body.add_geom()
    geom.name = "wood_box_geom"
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size = [BOX_SIZE / 2] * 3
    geom.rgba = [0.72, 0.45, 0.20, 1.0]  # 木色
    geom.density = 300.0                 # 木块质量约 0.3 kg
    geom.friction = [0.4, 0.01, 0.01]    # 中等偏低摩擦，被踢后会滑行一段

    return spec.compile()


def rpy_from_rot(rot):
    """旋转矩阵(ZYX) -> roll/pitch/yaw (deg)。"""
    pitch = math.asin(max(-1.0, min(1.0, -rot[2, 0])))
    roll = math.atan2(rot[2, 1], rot[2, 2])
    yaw = math.atan2(rot[1, 0], rot[0, 0])
    return np.degrees([roll, pitch, yaw])


def lowest_point(model, data):
    """机器人本体所有 geom 的世界系最低 z（跳过地面和木块）。"""
    zmin = np.inf
    box_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wood_box")
    for g in range(model.ngeom):
        bid = model.geom_bodyid[g]
        if bid == 0 or bid == box_bid:  # 跳过 world 里的地面和木块
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


def chase_command(base_xy, yaw_rad, box_xy):
    """只沿 +x 走廊追赶木块：前进速度随间距收敛，横向速度对齐木块。

    指令以世界系规划、再旋到机体系，这样机头有偏航时仍能沿世界 +x 走，
    避免绕圈；木块被踢飞后机器人会稳定地跟上去。
    """
    delta = box_xy - base_xy
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        return np.zeros(3), 0.0

    x_gap, y_err = delta[0], delta[1]

    vx_world = float(np.clip(0.9 * x_gap, SPEED_MIN, SPEED_MAX))
    vy_world = float(np.clip(1.2 * y_err, -0.4, 0.4))

    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    vx_body = c * vx_world + s * vy_world
    vy_body = -s * vx_world + c * vy_world

    # 航向保持：让机头尽量对着世界 +x，避免出现"倒着走"的怪姿态
    yaw_err = math.atan2(math.sin(yaw_rad), math.cos(yaw_rad))
    yaw_cmd = float(np.clip(-1.2 * yaw_err, -0.6, 0.6))

    return np.array([vx_body, vy_body, yaw_cmd]), dist


def main():
    cfg = yaml.safe_load(CFG_PATH.read_text())

    model = build_model()
    data = mujoco.MjData(model)
    policy = Policy(CONFIG_ROOT / cfg["policy_file"])

    box_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wood_box")
    box_jid = model.body_jntadr[box_bid]
    box_qadr = model.jnt_qposadr[box_jid]
    box_mass = model.body_mass[box_bid]

    active_names = cfg["active_joint_names"]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in active_names]
    qpos_ids = np.array([model.jnt_qposadr[j] for j in joint_ids])
    dof_ids = np.array([model.jnt_dofadr[j] for j in joint_ids])

    all_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[u, 0])
        for u in range(model.nu)
    ]
    all_joint_ids = [model.actuator_trnid[u, 0] for u in range(model.nu)]
    all_qpos = np.array([model.jnt_qposadr[j] for j in all_joint_ids])
    all_dof = np.array([model.jnt_dofadr[j] for j in all_joint_ids])

    def flat(key):
        """yaml 里按肢体分组且长度不一的列表 -> 一维数组。"""
        return np.concatenate([np.atleast_1d(np.asarray(g, dtype=float)) for g in cfg[key]])

    default_q = flat("default_joint_q")
    kp = flat("joint_kp")
    kd = flat("joint_kd")
    action_scale = flat("action_scale")

    joint_pos_in_all = [all_names.index(n) for n in active_names]
    default_q_act = default_q[joint_pos_in_all]

    n_act = len(active_names)
    num_obs = cfg["num_observations"]
    n_hist = cfg["num_include_obs_steps"]
    obs_scale = np.concatenate(
        [
            np.full(n_act, cfg["observation_scale_dof_pos"]),
            np.full(n_act, cfg["observation_scale_dof_vel"]),
            np.ones(n_act),
            np.full(3, cfg["observation_scale_angular_vel"]),
            np.full(3, cfg["observation_scale_quat"]),
        ]
    )
    cmd_scale = np.array(
        [
            cfg["observation_scale_linear_vel"],
            cfg["observation_scale_linear_vel"],
            cfg["observation_scale_angular_vel"],
        ]
    )
    obs_clip = cfg["observation_clip"]
    dt_ctrl = cfg["control_dt"]
    n_sub = max(1, round(dt_ctrl / model.opt.timestep))

    # --- 初始状态：默认站姿，脚底贴地 ---
    data.qpos[all_qpos] = default_q
    mujoco.mj_forward(model, data)
    data.qpos[2] += -lowest_point(model, data)
    mujoco.mj_forward(model, data)

    base_bid = int(model.jnt_bodyid[0])
    history = np.zeros((n_hist, num_obs))
    action = np.zeros(n_act)

    cmd = np.zeros(3)
    cmd_target = np.zeros(3)

    # 踢击检测：脚靠近木块时给木块一个速度增量，模拟"踢"
    foot_bids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        for n in ("LINK_FOOT_L", "LINK_FOOT_R")
    ]
    box_dof = model.jnt_dofadr[box_jid]
    last_kick = -1.0
    kick_count = 0

    print(f"策略: {cfg['policy_file']}")
    print(f"输入: {num_obs * n_hist + 3}, 动作: {n_act}, 控制: {1 / dt_ctrl:.0f} Hz")
    print(f"木块: 边长 {BOX_SIZE * 100:.0f} cm, 质量 {box_mass:.3f} kg, "
          f"初始位置 {np.round(data.qpos[box_qadr:box_qadr + 3], 3)}")
    print("自动跟随：机器人朝木块方向行走并持续推踢；关闭窗口退出")
    print()

    next_log = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 4.5
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -10.0
        viewer.cam.lookat[:] = [0.5, 0.0, 0.7]

        while viewer.is_running():
            t0 = time.time()

            # --- 用木块位置生成速度指令（chase -> kick -> back 状态机）---
            base_xy = data.qpos[0:2]
            box_xy = data.qpos[box_qadr:box_qadr + 2]
            base_rot = data.xmat[base_bid].reshape(3, 3)
            yaw_rad = math.atan2(base_rot[1, 0], base_rot[0, 0])
            cmd_target, dist = chase_command(base_xy, yaw_rad, box_xy)

            # --- 踢击：脚靠近木块且木块在前方时，给木块一个向前的速度增量 ---
            if data.time - last_kick > KICK_COOLDOWN:
                x_gap = box_xy[0] - base_xy[0]
                foot_dist = min(
                    float(np.linalg.norm(data.xpos[b][:2] - box_xy)) for b in foot_bids
                )
                if foot_dist < FOOT_KICK_DIST and x_gap > -0.05:
                    data.qvel[box_dof] += KICK_DV
                    data.qvel[box_dof + 2] += 0.3  # 稍微挑起，避免贴地卡住
                    last_kick = data.time
                    kick_count += 1

            alpha = min(1.0, dt_ctrl / CMD_LPF_TAU)
            cmd += (cmd_target - cmd) * alpha

            # --- 单步观测 ---
            q = data.qpos[qpos_ids]
            qd = data.qvel[dof_ids]
            ang_vel = data.qvel[3:6]  # 自由关节角速度在机体坐标系
            rot = data.xmat[base_bid].reshape(3, 3)
            proj_grav = rot.T @ np.array([0.0, 0.0, -1.0])

            obs = np.concatenate([q - default_q_act, qd, action, ang_vel, proj_grav])
            obs = np.clip(obs * obs_scale, -obs_clip, obs_clip)

            history[:-1] = history[1:]
            history[-1] = obs

            net_in = np.concatenate([history.ravel(), cmd * cmd_scale])
            action = np.clip(policy(net_in), -cfg["action_clip"], cfg["action_clip"])

            # --- 动作 -> 目标关节角 -> PD 力矩 ---
            q_des = default_q.copy()
            q_des[joint_pos_in_all] += action * action_scale
            q_now = data.qpos[all_qpos]
            tau = kp * (q_des - q_now) - kd * data.qvel[all_dof]
            lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
            data.ctrl[:] = np.clip(tau, lo, hi)

            for _ in range(n_sub):
                mujoco.mj_step(model, data)
            viewer.sync()

            # --- 100 ms 日志 + 窗口 HUD ---
            if data.time + 1e-9 >= next_log:
                pos = data.qpos[0:3]
                rpy = rpy_from_rot(data.xmat[base_bid].reshape(3, 3))
                print(f"[t={data.time:7.3f}s] "
                      f"cmd=({cmd[0]:+.2f},{cmd[1]:+.2f}) dist={dist:.2f}m "
                      f"pos=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.3f})m "
                      f"rpy=({rpy[0]:+5.1f},{rpy[1]:+5.1f},{rpy[2]:+6.1f})deg "
                      f"v=({data.qvel[0]:+.2f},{data.qvel[1]:+.2f})m/s "
                      f"box_x={box_xy[0]:+.2f}m kicks={kick_count} "
                      f"推理={policy.last_ms:.2f}ms")

                viewer.set_texts([
                    (mujoco.mjtFontScale.mjFONTSCALE_150,
                     mujoco.mjtGridPos.mjGRID_TOPLEFT,
                     "T800 chase & kick the box",
                     f"cmd vx={cmd[0]:+.2f} vy={cmd[1]:+.2f}"),
                    (mujoco.mjtFontScale.mjFONTSCALE_100,
                     mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                     f"dist={dist:.2f}m  box_x={box_xy[0]:+.2f}m",
                     f"h={data.qpos[2]:.2f}m  vx={data.qvel[0]:+.2f}m/s"),
                ])
                next_log += LOG_DT

            remain = dt_ctrl - (time.time() - t0)
            if remain > 0:
                time.sleep(remain)


if __name__ == "__main__":
    main()
