"""关节位置 PD -> 力矩，并按 ctrlrange 裁剪。"""

import numpy as np


class JointPD:
    def __init__(self, kp: np.ndarray, kd: np.ndarray, ctrl_lo: np.ndarray, ctrl_hi: np.ndarray):
        self.kp = kp
        self.kd = kd
        self.ctrl_lo = ctrl_lo
        self.ctrl_hi = ctrl_hi

    def torque(self, q_des: np.ndarray, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        tau = self.kp * (q_des - q) - self.kd * qd
        return np.clip(tau, self.ctrl_lo, self.ctrl_hi)
