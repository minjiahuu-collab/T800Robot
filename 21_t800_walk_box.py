"""T800 行走策略 + WASD 遥控 + 动态加入 8 cm 木块。

木块通过 MuJoCo 的 MjSpec 在运行时添加到模型里，不修改官方 SDK 下的任何文件：
  spec = MjSpec.from_file(t800.xml) -> worldbody.add_body() -> compile()

木块边长 8 cm，带 freejoint，可被机器人踢动/推动；默认放在机器人
前进路线右前方 1 m 处，避免一开局就绊倒。

用法:
    python3 21_t800_walk_box.py
"""

import collections
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

LOG_DT = 0.1        # 日志周期 (s)
CMD_LPF_TAU = 0.1   # 指令平滑时间常数 (s)
VX_FWD = 0.6        # 前进速度 (m/s)
VX_BACK = 0.4       # 后退速度 (m/s)
VY_SIDE = 0.4       # 侧移速度 (m/s)

KEY_W, KEY_S, KEY_A, KEY_D = 87, 83, 65, 68
KEY_R = 82

BOX_SIZE = 0.08                        # 木块边长 (m)
BOX_POS = (1.0, 0.35, BOX_SIZE / 2)    # 木块初始位置（半边长高度落地）


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
    """加载官方 XML 并在运行时插入 8 cm 木块，返回编译后的 model。"""
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
        if bid == 0 or bid == box_bid:  # 跳过 world 地面的地面和木块
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

    model = build_model()
    data = mujoco.MjData(model)
    policy = Policy(CONFIG_ROOT / cfg["policy_file"])

    box_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wood_box")
    box_jid = model.body_jntadr[box_bid]
    box_qadr = model.jnt_qposadr[box_jid]
    box_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wood_box_geom")
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

    # 速度指令：cmd_target 由按键锁存，cmd 低通后送策略
    cmd_target = np.array([0.0, 0.0, 0.0])
    cmd = cmd_target.copy()

    # viewer 回调在渲染线程，按键先入队，主循环消费
    pending = collections.deque()

    def on_key(keycode):
        pending.append(keycode)

    print(f"策略: {cfg['policy_file']}")
    print(f"输入: {num_obs * n_hist + 3}, 动作: {n_act}, 控制: {1 / dt_ctrl:.0f} Hz")
    print(f"木块: 边长 {BOX_SIZE * 100:.0f} cm, 质量 {box_mass:.3f} kg, "
          f"初始位置 {np.round(data.qpos[box_qadr:box_qadr + 3], 3)}, "
          f"geom id={box_gid}")
    print("W: 前进  S: 后退  A: 左移  D: 右移  R: 停下（点一下窗口获取焦点）")
    print()

    x0 = data.qpos[0]
    next_log = 0.0

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -10.0
        viewer.cam.lookat[:] = [0.0, 0.0, 0.8]

        # 记录启动时的渲染状态，之后每帧恢复，抵消 viewer 自带快捷键的副作用
        flags0 = viewer.opt.flags.copy()
        groups0 = viewer.opt.geomgroup.copy()

        while viewer.is_running():
            t0 = time.time()

            viewer.opt.flags[:] = flags0
            viewer.opt.geomgroup[:] = groups0

            # --- 处理按键：锁存方向指令 ---
            while pending:
                key = pending.popleft()
                if key == KEY_W:
                    cmd_target[:] = [VX_FWD, 0.0, 0.0]
                elif key == KEY_S:
                    cmd_target[:] = [-VX_BACK, 0.0, 0.0]
                elif key == KEY_A:
                    cmd_target[:] = [0.0, VY_SIDE, 0.0]
                elif key == KEY_D:
                    cmd_target[:] = [0.0, -VY_SIDE, 0.0]
                elif key == KEY_R:
                    cmd_target[:] = 0.0

            # 一阶低通，避免指令突变
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
                box_pos = data.qpos[box_qadr:box_qadr + 3]
                print(f"[t={data.time:7.3f}s] "
                      f"cmd=({cmd[0]:+.2f},{cmd[1]:+.2f}) "
                      f"pos=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.3f})m "
                      f"rpy=({rpy[0]:+5.1f},{rpy[1]:+5.1f},{rpy[2]:+6.1f})deg "
                      f"v=({data.qvel[0]:+.2f},{data.qvel[1]:+.2f})m/s "
                      f"box=({box_pos[0]:+.2f},{box_pos[1]:+.2f},{box_pos[2]:.3f})m "
                      f"推理={policy.last_ms:.2f}ms")

                viewer.set_texts([
                    (mujoco.mjtFontScale.mjFONTSCALE_150,
                     mujoco.mjtGridPos.mjGRID_TOPLEFT,
                     "T800 WASD teleop + box",
                     f"cmd vx={cmd[0]:+.2f} vy={cmd[1]:+.2f}"),
                    (mujoco.mjtFontScale.mjFONTSCALE_100,
                     mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                     "W fwd  S back  A left  D right  R stop",
                     f"h={data.qpos[2]:.2f}m  x={data.qpos[0] - x0:+.2f}m"),
                ])
                next_log += LOG_DT

            remain = dt_ctrl - (time.time() - t0)
            if remain > 0:
                time.sleep(remain)


if __name__ == "__main__":
    main()
