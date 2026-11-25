"""机器人索引表与常用状态查询。

所有"名字 -> 下标"的映射集中在这里，避免每个脚本各自维护一份。
"""

import math
from typing import Iterable, Optional

import mujoco
import numpy as np


class Robot:
    """对 (model, data) 的一层轻量封装，不做全局单例，便于多实例/多进程。"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

        self.actuator_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[u, 0])
            for u in range(model.nu)
        ]
        joint_ids = [int(model.actuator_trnid[u, 0]) for u in range(model.nu)]
        self.qpos_idx = np.array([model.jnt_qposadr[j] for j in joint_ids])
        self.dof_idx = np.array([model.jnt_dofadr[j] for j in joint_ids])
        self.ctrl_lo = model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = model.actuator_ctrlrange[:, 1].copy()

        self.base_bid = int(model.jnt_bodyid[0])  # freejoint 所在 body
        self._base_q: Optional[np.ndarray] = None

    # --- 索引 ---
    def joint_slot(self, name: str) -> int:
        """关节/电机在 actuator 列表中的位置。"""
        return self.actuator_names.index(name)

    def body_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    # --- 状态 ---
    def qpos(self) -> np.ndarray:
        return self.data.qpos[self.qpos_idx]

    def qvel(self) -> np.ndarray:
        return self.data.qvel[self.dof_idx]

    def joint_angle(self, slot: int) -> float:
        return float(self.data.qpos[self.qpos_idx[slot]])

    def xpos(self, name: str) -> np.ndarray:
        return self.data.xpos[self.body_id(name)]

    def rotation(self) -> np.ndarray:
        return self.data.xmat[self.base_bid].reshape(3, 3)

    def rpy(self) -> np.ndarray:
        """基座 roll/pitch/yaw（deg，ZYX 顺序）。"""
        rot = self.rotation()
        pitch = math.asin(max(-1.0, min(1.0, -rot[2, 0])))
        roll = math.atan2(rot[2, 1], rot[2, 2])
        yaw = math.atan2(rot[1, 0], rot[0, 0])
        return np.degrees([roll, pitch, yaw])

    # --- 初始化 ---
    def set_joint_pose(self, q: np.ndarray) -> None:
        self.data.qpos[self.qpos_idx] = q

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    def lowest_point(self, skip_bodies: Iterable[int] = ()) -> float:
        """本体所有 geom 的世界系最低 z，跳过 world 与指定 body。"""
        skip = set(skip_bodies)
        zmin = np.inf
        for g in range(self.model.ngeom):
            bid = self.model.geom_bodyid[g]
            if bid == 0 or bid in skip:
                continue
            center = self.model.geom_aabb[g][:3]
            size = self.model.geom_aabb[g][3:]
            rot = self.data.geom_xmat[g].reshape(3, 3)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        corner = center + size * np.array([sx, sy, sz])
                        zmin = min(zmin, (rot @ corner + self.data.geom_xpos[g])[2])
        return float(zmin)

    def place_on_ground(self, offset: float = 0.0, skip_bodies: Iterable[int] = ()) -> None:
        """按最低点反推基座高度，使脚底贴地。"""
        self.data.qpos[2] += -self.lowest_point(skip_bodies) + offset
        self.forward()

    # --- 固定基座 ---
    def capture_base(self) -> None:
        """记录当前基座位姿；仅适用于带 freejoint 的浮动基座。"""
        self._base_q = self.data.qpos[:7].copy()

    def pin_base(self) -> None:
        """把基座位姿和速度复位到 capture_base 时的状态。"""
        if self._base_q is None:
            raise RuntimeError("调用 pin_base 前需要先 capture_base()")
        self.data.qpos[:7] = self._base_q
        self.data.qvel[:6] = 0.0
