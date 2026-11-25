"""24 的包化样板：固定 base、PD 保持站姿、W/S 抬放右臂。

与 24_t800_right_arm.py 的行为一致，但模型加载、关节索引、最低点反推、PD、
viewer 循环与日志节流全部来自 t800_sim 包，本文件只保留任务本身。

用法:
    python3 24b_t800_right_arm_pkg.py
"""

import mujoco

from t800_sim import loop
from t800_sim.config import RobotConfig
from t800_sim.control.pd import JointPD
from t800_sim.model import load_model
from t800_sim.robot import Robot
from t800_sim.viewer import ViewerBackend

LOG_DT = 0.1
RAMP_TIME = 1.5
LEG_TORSO_KD_SCALE = 2.0
ARM_KD_SCALE = 4.0

KEY_W, KEY_S = 87, 83

RAISE_POSE = {
    "J18_SHOULDER_PITCH_R": -1.2,
    "J19_SHOULDER_ROLL_R": -0.4,
    "J21_ELBOW_PITCH_R": -0.8,
}


def smootherstep(b: float) -> float:
    return b * b * b * (b * (b * 6 - 15) + 10)


def main() -> None:
    cfg = RobotConfig.load()
    model = load_model()
    data = mujoco.MjData(model)
    robot = Robot(model, data)
    assert len(cfg.default_q) == model.nu, "yaml 关节数与模型 actuator 数不一致"

    # 腿/躯干阻尼加倍抑制抬臂反作用引起的摆动；右臂阻尼加倍加快放臂余振衰减
    kd = cfg.kd.copy()
    for i, name in enumerate(robot.actuator_names):
        if any(k in name for k in ("HIP", "KNEE", "ANKLE", "TORSO")):
            kd[i] *= LEG_TORSO_KD_SCALE
        elif name.endswith("_R") and ("SHOULDER" in name or "ELBOW" in name):
            kd[i] *= ARM_KD_SCALE
    pd = JointPD(cfg.kp, kd, robot.ctrl_lo, robot.ctrl_hi)

    robot.set_joint_pose(cfg.default_q)
    robot.forward()
    robot.place_on_ground()
    robot.capture_base()

    lower_q = cfg.default_q.copy()
    raise_q = lower_q.copy()
    for name, val in RAISE_POSE.items():
        raise_q[robot.joint_slot(name)] = val

    n_sub = max(1, round(cfg.dt_ctrl / model.opt.timestep))
    wrist_bid = robot.body_id("LINK_WRIST_END_R")
    sh_slot = robot.joint_slot("J18_SHOULDER_PITCH_R")
    el_slot = robot.joint_slot("J21_ELBOW_PITCH_R")

    blend = 0.0          # 0 = 放下, 1 = 举起
    blend_target = 0.0
    ramp_step = cfg.dt_ctrl / RAMP_TIME
    throttle = loop.LogThrottle(LOG_DT)

    def step(view: ViewerBackend) -> None:
        nonlocal blend, blend_target

        while (key := view.poll_key()) is not None:
            if key == KEY_W:
                blend_target = 1.0
            elif key == KEY_S:
                blend_target = 0.0

        if blend < blend_target:
            blend = min(blend_target, blend + ramp_step)
        elif blend > blend_target:
            blend = max(blend_target, blend - ramp_step)
        s = smootherstep(blend)

        q_des = lower_q + s * (raise_q - lower_q)
        data.ctrl[:] = pd.torque(q_des, robot.qpos(), robot.qvel())

        for _ in range(n_sub):
            robot.pin_base()
            mujoco.mj_step(model, data)

        if throttle.due(data.time):
            rpy = robot.rpy()
            wrist = robot.xpos("LINK_WRIST_END_R")
            state = "RAISE" if blend_target > 0.5 else "LOWER"
            print(f"[t={data.time:7.3f}s] {state} s={s:.2f} "
                  f"shoulder_pitch={robot.joint_angle(sh_slot):+.2f}rad "
                  f"elbow_pitch={robot.joint_angle(el_slot):+.2f}rad "
                  f"wrist=({wrist[0]:+.2f},{wrist[1]:+.2f},{wrist[2]:+.2f})m "
                  f"base=({data.qpos[0]:+.2f},{data.qpos[1]:+.2f},{data.qpos[2]:+.3f})m "
                  f"rpy=({rpy[0]:+5.1f},{rpy[1]:+5.1f},{rpy[2]:+6.1f})deg")

            view.set_texts([
                (mujoco.mjtFontScale.mjFONTSCALE_150,
                 mujoco.mjtGridPos.mjGRID_TOPLEFT,
                 "T800 right arm: W = raise, S = lower",
                 f"state={state}  s={s:.2f}"),
                (mujoco.mjtFontScale.mjFONTSCALE_100,
                 mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                 "base fixed, all other joints hold default pose",
                 f"wrist_z={wrist[2]:.2f}m  base_h={data.qpos[2]:.2f}m"),
            ])

    print("T800 右臂控制（t800_sim 包版本）：base 固定，其余关节保持站姿")
    print("W: 抬起右臂   S: 放下右臂（点一下窗口获取焦点）")
    print()

    with ViewerBackend(model, data) as view:
        view.set_camera(distance=3.2, azimuth=145.0, elevation=-8.0, lookat=[0.0, 0.0, 1.0])
        loop.run(step, view, cfg.dt_ctrl)


if __name__ == "__main__":
    main()
