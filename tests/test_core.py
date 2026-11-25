"""回归测试：模型维度、贴地初始化与基座固定、PD 保持与策略接口。"""

import mujoco
import numpy as np
import pytest

from t800_sim.config import RobotConfig
from t800_sim.control.pd import JointPD
from t800_sim.control.policy import RlController
from t800_sim.model import load_model
from t800_sim.paths import CONFIG_ROOT
from t800_sim.robot import Robot

EXPECTED_HEIGHT = 1.0383


@pytest.fixture(scope="module")
def model():
    return load_model()


def _make_sim(model):
    cfg = RobotConfig.load()
    data = mujoco.MjData(model)
    robot = Robot(model, data)
    robot.set_joint_pose(cfg.default_q)
    robot.forward()
    robot.place_on_ground()
    robot.capture_base()
    return cfg, data, robot


def test_model_and_config(model):
    cfg = RobotConfig.load()
    assert (model.nq, model.nv, model.nu) == (32, 31, 25)
    assert cfg.default_q.size == model.nu
    assert len(cfg.active_joint_names) == 22
    assert cfg.kp.size == model.nu
    assert cfg.kd.size == model.nu
    # action_scale 只覆盖策略控制的 22 个关节，不含头部与躯干 yaw
    assert cfg.action_scale.size == len(cfg.active_joint_names)


def test_ground_placement_and_base_pin(model):
    _, data, robot = _make_sim(model)
    assert data.qpos[2] == pytest.approx(EXPECTED_HEIGHT, abs=1e-3)
    assert np.allclose(robot.rpy(), 0.0, atol=0.2)

    data.qpos[0] = 0.5
    data.qvel[0] = 1.0
    robot.pin_base()
    assert data.qpos[0] == pytest.approx(0.0, abs=1e-9)
    assert data.qvel[0] == pytest.approx(0.0, abs=1e-9)


def test_pd_hold_then_policy_action(model):
    cfg, data, robot = _make_sim(model)
    pd = JointPD(cfg.kp, cfg.kd, robot.ctrl_lo, robot.ctrl_hi)
    q_des = cfg.default_q.copy()

    for _ in range(150):  # 1.5 s
        data.ctrl[:] = pd.torque(q_des, robot.qpos(), robot.qvel())
        for _ in range(5):
            robot.pin_base()
            mujoco.mj_step(model, data)

    assert data.qpos[2] == pytest.approx(EXPECTED_HEIGHT, abs=0.01)
    assert np.allclose(robot.rpy(), 0.0, atol=1.0)
    assert np.max(np.abs(robot.qpos() - q_des)) < 0.1

    ctrl = RlController(cfg, robot, CONFIG_ROOT / cfg.policy_file)
    tau = ctrl.act(np.zeros(3))
    assert tau.shape == (model.nu,)
    assert np.all(np.isfinite(tau))
    assert np.all(tau >= robot.ctrl_lo - 1e-6)
    assert np.all(tau <= robot.ctrl_hi + 1e-6)