"""官方 default.yaml 的解析结果。

注意：yaml 里按肢体分组的数组拼平后，顺序与 model.nu 的 actuator 顺序一致。
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .paths import DEFAULT_CFG


def _flat(cfg: dict, key: str) -> np.ndarray:
    """把长度不一的嵌套列表拼成一维数组。"""
    return np.concatenate([np.atleast_1d(np.asarray(g, dtype=float)) for g in cfg[key]])


@dataclass(frozen=True)
class RobotConfig:
    policy_file: str
    active_joint_names: list
    num_obs: int
    n_hist: int
    dt_ctrl: float
    default_q: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    action_scale: np.ndarray
    obs_scale: np.ndarray
    cmd_scale: np.ndarray
    obs_clip: float
    action_clip: float

    @classmethod
    def load(cls, path: Path = DEFAULT_CFG) -> "RobotConfig":
        cfg = yaml.safe_load(Path(path).read_text())
        n_act = len(cfg["active_joint_names"])

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

        return cls(
            policy_file=cfg["policy_file"],
            active_joint_names=list(cfg["active_joint_names"]),
            num_obs=cfg["num_observations"],
            n_hist=cfg["num_include_obs_steps"],
            dt_ctrl=cfg["control_dt"],
            default_q=_flat(cfg, "default_joint_q"),
            kp=_flat(cfg, "joint_kp"),
            kd=_flat(cfg, "joint_kd"),
            action_scale=_flat(cfg, "action_scale"),
            obs_scale=obs_scale,
            cmd_scale=cmd_scale,
            obs_clip=cfg["observation_clip"],
            action_clip=cfg["action_clip"],
        )
