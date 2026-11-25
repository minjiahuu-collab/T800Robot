"""MNN 行走策略：推理封装 + 观测历史 + 动作到 PD 力矩。

对应官方策略的输入输出约定：
  单步观测 72 维 = [q - default_q, qd, last_action, ang_vel, proj_grav]
  15 步历史 + 3 维速度指令 -> 动作 22 维 -> q_des = default_q + action * action_scale
"""

import time
from pathlib import Path

import MNN
import numpy as np

from .pd import JointPD


class MNNPolicy:
    def __init__(self, path: Path):
        self.interp = MNN.Interpreter(str(path))
        self.sess = self.interp.createSession({"backend": "CPU", "numThread": 4})
        self.input = self.interp.getSessionInput(self.sess)
        self.output = self.interp.getSessionOutput(self.sess)
        self.last_ms = 0.0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
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
        # 该 MNN 版本 copyToHostTensor 取不到结果，直接读主机内存
        return np.array(self.output.getData(), dtype=np.float32).ravel()


class RlController:
    """观测 -> 策略 -> 目标关节角 -> PD 力矩，一个控制周期调一次 act()。"""

    def __init__(self, cfg, robot, policy_path: Path):
        self.cfg = cfg
        self.robot = robot
        self.policy = MNNPolicy(policy_path)
        self.slots = [robot.actuator_names.index(n) for n in cfg.active_joint_names]
        self.default_q_act = cfg.default_q[self.slots]
        self.history = np.zeros((cfg.n_hist, cfg.num_obs))
        self.action = np.zeros(len(self.slots))
        self.pd = JointPD(cfg.kp, cfg.kd, robot.ctrl_lo, robot.ctrl_hi)

    @property
    def last_ms(self) -> float:
        return self.policy.last_ms

    def act(self, cmd: np.ndarray) -> np.ndarray:
        cfg, robot = self.cfg, self.robot

        q = robot.qpos()[self.slots]
        qd = robot.qvel()[self.slots]
        ang_vel = robot.data.qvel[3:6]  # 自由关节角速度在机体坐标系
        proj_grav = robot.rotation().T @ np.array([0.0, 0.0, -1.0])

        obs = np.concatenate([q - self.default_q_act, qd, self.action, ang_vel, proj_grav])
        obs = np.clip(obs * cfg.obs_scale, -cfg.obs_clip, cfg.obs_clip)

        self.history[:-1] = self.history[1:]
        self.history[-1] = obs

        net_in = np.concatenate([self.history.ravel(), cmd * cfg.cmd_scale])
        self.action = np.clip(self.policy(net_in), -cfg.action_clip, cfg.action_clip)

        q_des = cfg.default_q.copy()
        q_des[self.slots] += self.action * cfg.action_scale
        return self.pd.torque(q_des, robot.qpos(), robot.qvel())
