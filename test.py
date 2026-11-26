# import csv, json, numpy as np
# s = "recordings/walk_vx0.50_vy0.00_yaw0.00/box/seed_0000/ep_0000"
# rows = list(csv.DictReader(open(f"{s}/states.csv")))
# x = np.array([float(r["base_x"]) for r in rows])
# t = np.array([float(r["time"]) for r in rows])
# meta = json.load(open(f"{s}/meta.json"))
# print("我的复算:", (x[-1]-x[0])/(t[-1]-t[0]))
# print("meta 记录:", meta["metrics"]["mean_vx"], "dx:", meta["metrics"]["final_dx"])

import csv, json, sys, mujoco, numpy as np
from pathlib import Path
xml = Path(__file__).resolve().parent / "engineai_robotics_native_sdk/assets/resource/t800.xml"
model = mujoco.MjModel.from_xml_path(str(xml))
data = mujoco.MjData(model)
s = sys.argv[1] if len(sys.argv) > 1 else "recordings/walk_vx0.50_vy0.00_yaw0.00/box/seed_0000/ep_0000"
meta = json.load(open(f"{s}/meta.json"))
states = list(csv.DictReader(open(f"{s}/states.csv")))
gt = {int(r["frame"]): r for r in csv.DictReader(open(f"{s}/ground_truth.csv"))}
bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "LINK_WRIST_END_L")

for fr in (0, 75, 150):
    r = states[fr]
    data.qpos[:7] = [float(r[k]) for k in ("base_x","base_y","base_z","qw","qx","qy","qz")]
    for name in meta["joints"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = float(r[name])
    mujoco.mj_kinematics(model, data)
    fk = data.xpos[bid]
    rec = [float(gt[fr][k]) for k in ("ee_l_x","ee_l_y","ee_l_z")]
    print(fr, "FK:", np.round(fk, 4), "记录:", np.round(rec, 4),
          "误差:", np.abs(fk - rec).max())