"""T800 行走策略测试 + 详细日志：每 100 ms 打印位姿、推理、PD 和步态相关关节信息。

在 17_t800_walk_policy.py 基础上增加日志，方便分析步态：
  - 机器人 pos(x,y,z)、rpy(deg)、基座速度
  - 模型推理：单次耗时、动作范数/最大值
  - PD 控制：最大跟随误差、最大力矩、饱和电机数
  - 腿部 6 个关键关节（髋 pitch / 膝 pitch / 踝 pitch，左右）的目标角与实际角

用法:
    python3 18_t800_walk_policy_log.py
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

CMD_VX = 0.5   # 前进速度指令 (m/s)
LOG_DT = 0.1   # 日志周期 (s)


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
        # 注意: 该 MNN 版本 copyToHostTensor 取不到结果，直接读 host 内存
        return np.array(self.output.getData(), dtype=np.float32).ravel()


def rpy_from_rot(rot):
    """旋转矩阵(ZYX) -> roll/pitch/yaw (deg)。"""
    pitch = math.asin(max(-1.0, min(1.0, -rot[2, 0])))
    roll = math.atan2(rot[2, 1], rot[2, 2])
    yaw = math.atan2(rot[1, 0], rot[0, 0])
    return np.degrees([roll, pitch, yaw])


def lowest_point(model, data):
    """机器人本体所有 geom 的世界系最低 z（用于把初始姿态放到地面）。"""
    zmin = np.inf
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0:  # 跳过 world 里的地面
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
    cfg = yaml.safe_load(CFG_PATH.read_text())

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    policy = Policy(CONFIG_ROOT / cfg["policy_file"])

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
    # data.qpos[2] += -lowest_point(model, data)
    data.qpos[2] += -1
    mujoco.mj_forward(model, data)

    base_bid = int(model.jnt_bodyid[0])
    history = np.zeros((n_hist, num_obs))
    action = np.zeros(n_act)
    cmd = np.array([CMD_VX, 0.0, 0.0])

    # 步态分析关注的关键关节：髋 pitch / 膝 pitch / 踝 pitch（左右）
    key_joints = ["J00_HIP_PITCH_L", "J03_KNEE_PITCH_L", "J04_ANKLE_PITCH_L",
                  "J06_HIP_PITCH_R", "J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R"]
    key_idx = [all_names.index(n) for n in key_joints]
    key_label = "  ".join(n.split("_", 1)[1] for n in key_joints)

    print(f"策略: {cfg['policy_file']}")
    print(f"输入: {num_obs} x {n_hist} + 3 = {num_obs * n_hist + 3}, "
          f"动作: {n_act}, 控制: {1 / dt_ctrl:.0f} Hz, 物理步长: {model.opt.timestep}s")
    print(f"日志周期: {LOG_DT * 1e3:.0f} ms")
    print("关节列顺序（目标/实际, rad）: " + key_label)
    print()

    x0 = data.qpos[0]
    next_log = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -10.0
        viewer.cam.lookat[:] = [0.0, 0.0, 0.8]

        while viewer.is_running():
            t0 = time.time()

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
            qd_now = data.qvel[all_dof]
            tau = kp * (q_des - q_now) - kd * qd_now
            lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
            data.ctrl[:] = np.clip(tau, lo, hi)

            for _ in range(n_sub):
                mujoco.mj_step(model, data)
            viewer.sync()

            # --- 100 ms 日志 ---
            if data.time + 1e-9 >= next_log:
                pos = data.qpos[0:3]
                rpy = rpy_from_rot(data.xmat[base_bid].reshape(3, 3))
                vel = data.qvel[0:3]
                q_err = np.abs(q_des - q_now)
                sat = int(np.sum((tau < lo - 1e-6) | (tau > hi + 1e-6)))

                print(f"[t={data.time:7.3f}s] "
                      f"pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})m "
                      f"rpy=({rpy[0]:+6.2f},{rpy[1]:+6.2f},{rpy[2]:+6.2f})deg "
                      f"v=({vel[0]:+.2f},{vel[1]:+.2f},{vel[2]:+.2f})m/s")
                print(f"          推理 {policy.last_ms:5.2f}ms | |a|={np.linalg.norm(action):5.2f} "
                      f"max|a|={np.max(np.abs(action)):4.2f} | "
                      f"PD err_max={q_err.max():.3f}rad tau_max={np.abs(data.ctrl).max():6.1f}Nm "
                      f"饱和={sat}/{model.nu} | x={data.qpos[0] - x0:+.3f}m")
                pairs = "  ".join(
                    f"{q_des[i]:+.2f}/{q_now[i]:+.2f}" for i in key_idx
                )
                print(f"          关节(des/act): {pairs}")
                next_log += LOG_DT

            remain = dt_ctrl - (time.time() - t0)
            if remain > 0:
                time.sleep(remain)


if __name__ == "__main__":
    main()
