"""T800 行走策略测试 Demo：用 MNN 加载 RL 行走策略，在 MuJoCo 里闭环行走。

流程与 SDK 的 rl_walking_example_runner.cc 一致：
  1. 读取关节位置/速度、基座角速度、重力投影，拼 72 维单步观测
  2. 维护 15 步历史滑动窗口，尾部拼接速度指令(3) → 1083 维网络输入
  3. MNN 推理得到 22 维动作，q_des = default_q + action * action_scale
  4. 位置 PD（kp/kd 来自 yaml）转力矩下发，100 Hz 控制、5 次物理子步

用法:
    python3 17_t800_walk_policy.py
关闭窗口退出。
"""

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

CMD_VX = 0.5  # 前进速度指令 (m/s)


class Policy:
    def __init__(self, path):
        self.interp = MNN.Interpreter(str(path))
        self.sess = self.interp.createSession({"backend": "CPU", "numThread": 4})
        self.input = self.interp.getSessionInput(self.sess)
        self.output = self.interp.getSessionOutput(self.sess)

    def __call__(self, obs):
        inp = MNN.Tensor(
            (1, obs.size),
            MNN.Halide_Type_Float,
            obs.astype(np.float32),
            MNN.Tensor_DimensionType_Caffe,
        )
        self.input.copyFrom(inp)
        self.interp.runSession(self.sess)
        # 注意: 该 MNN 版本 copyToHostTensor 取不到结果，直接读 host 内存
        return np.array(self.output.getData(), dtype=np.float32).ravel()


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

    # 关节名 -> 完整身体关节顺序（与 XML 中 J00..J24 一致）
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

    # 按 active_joint_names 顺序重排 PD 参数（policy 动作顺序 = 该列表顺序）
    joint_pos_in_all = [all_names.index(n) for n in active_names]
    kp_act = kp[joint_pos_in_all]
    kd_act = kd[joint_pos_in_all]
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
    cmd = np.array([CMD_VX, 0.0, 0.0])

    print(f"策略: {cfg['policy_file']}")
    print(f"输入维度: {num_obs} x {n_hist} + 3 = {num_obs * n_hist + 3}, "
          f"动作维度: {n_act}, 控制频率: {1 / dt_ctrl:.0f} Hz")
    print(f"{'时间(s)':>8} {'基座高度(m)':>12} {'前进距离(m)':>12} {'前进速度(m/s)':>14}")
    x0 = data.qpos[0]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -10.0
        viewer.cam.lookat[:] = [0.0, 0.0, 0.8]

        next_stat = 0.0
        while viewer.is_running():
            t0 = time.time()

            # --- 组装单步观测（顺序必须与训练一致）---
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
            tau = kp * (q_des - data.qpos[all_qpos]) - kd * data.qvel[all_dof]
            data.ctrl[:] = np.clip(
                tau, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
            )

            for _ in range(n_sub):
                mujoco.mj_step(model, data)
            viewer.sync()

            if data.time + 1e-9 >= next_stat:
                print(f"{data.time:8.2f} {data.qpos[2]:12.3f} "
                      f"{data.qpos[0] - x0:12.3f} {data.qvel[0]:14.3f}")
                next_stat += 0.5

            remain = dt_ctrl - (time.time() - t0)
            if remain > 0:
                time.sleep(remain)


if __name__ == "__main__":
    main()
