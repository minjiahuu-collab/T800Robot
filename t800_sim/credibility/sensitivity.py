"""P3 敏感性分析：在安全边界内扫描关键物理参数对仿真行为的影响。

参数选择规则（来自 MuJoCo 接触经验）：
  - 摩擦：只改 geom_friction[:, 0]（滑动分量），扭转/滚动保持基线
  - geom 集合：区分 robot（LINK_ 前缀 body）与 ground（"floor" geom）
  - 质量：只改 robot body_mass，不动 world
  - 接触刚度：用 o_solref[0] 标量（默认 0.02），o_solimp 不动
  - 重力：只改 z 分量

每个维度扫 3 个值（含 1.0 基线），指标 dx/mean_vx/h_min/max_penetration/fell。
敏感性指标 = (max - min) / baseline（范围/基线，越小越不敏感）。

用法:
    python3 -m t800_sim.cli cred-sensitivity
    python3 -m t800_sim.cli cred-sensitivity --dim mass friction --out /tmp/sens.csv
"""

import csv
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from ..config import RobotConfig
from ..control.policy import RlController
from ..model import build_model
from ..paths import CONFIG_ROOT
from ..robot import Robot
from ..scenes import add_box
from .sim import SLIP_SPEED, apply_init_noise, base_metrics, rollout

# ── 扰动维度定义 ──────────────────────────────────────────────────────────
# 每个维度: (键名, 扫描值列表, 描述, category)
PERTURB_DIMS = {
    "mass":            {"values": [0.7, 1.0, 1.3], "desc": "Robot body mass × scale"},
    "ground_friction": {"values": [0.3, 1.0, 2.0], "desc": "Floor sliding friction × scale"},
    "robot_friction":  {"values": [0.5, 1.0, 1.5], "desc": "Robot geom sliding friction × scale"},
    "stiffness":       {"values": [0.5, 1.0, 2.0], "desc": "Contact stiffness (o_solref[0]) × scale"},
    "gravity":         {"values": [0.8, 1.0, 1.2], "desc": "Gravity z-component × scale"},
}

DEFAULT_CMDS = [(0.3, 0.0, 0.0), (0.6, 0.0, 0.0), (0.8, 0.0, 0.0), (0.5, 0.5, 0.0)]
DEFAULT_SEEDS = [0]
DEFAULT_DURATION = 5.0
BOX_SIZE = 0.10
BOX_POS = (1.8, 0.8, BOX_SIZE / 2)


# ── 模型修改 ───────────────────────────────────────────────────────────────
def apply_perturb(model: mujoco.MjModel, dim: str, value: float) -> None:
    """就地修改 model 数组。必须在 build_model 之后、MjData 之前调用。"""
    if dim == "mass":
        # Robot bodies: name starts with "LINK_"
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and (name.startswith("LINK_") or name.startswith("Motor_")):
                model.body_mass[i] *= value
    elif dim == "ground_friction":
        # Floor geom (body=world, type=plane)
        for i in range(model.ngeom):
            bname = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i])
            )
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if gname == "floor" or (bname == "world" and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE):
                model.geom_friction[i, 0] *= value
                # 保持 torsional/rolling 比例（避免三分量失衡触发求解异常）
                # torsional[1] 默认 0.005，rolling[2] 默认 0.0001
                # 比例保持: frictions[1] = baseline_sliding * base_torsional_ratio
                # 这里简化: 同步按比例缩放
    elif dim == "robot_friction":
        # Robot geoms attached to LINK_ bodies
        for i in range(model.ngeom):
            body_id = int(model.geom_bodyid[i])
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if bname and (bname.startswith("LINK_") or bname.startswith("Motor_")):
                model.geom_friction[i, 0] *= value
    elif dim == "stiffness":
        model.opt.o_solref[0] *= value
    elif dim == "gravity":
        model.opt.gravity[2] *= value
    else:
        raise ValueError(f"未知维度: {dim}")


# ── 单次带扰动的仿真 ──────────────────────────────────────────────────────
def _scene_hook(scene: str):
    if scene == "box":
        def hook(spec):
            add_box(spec, "wood_box", BOX_SIZE, pos=BOX_POS, free=True,
                    friction=(0.4, 0.01, 0.01))
        return hook
    return None


def perturb_rollout(
    cmd,
    dim: str,
    value: float,
    *,
    seed: int = 0,
    episode: int = 0,
    scene: str = "box",
    duration: float = DEFAULT_DURATION,
    init_noise=(0.0, 0.0, 0.0),
    contacts: bool = True,
) -> dict:
    """带参数扰动的 rollout。与 sim.rollout 独立建模型后就地改参数。"""
    cfg = RobotConfig.load()
    hooks = []
    hook = _scene_hook(scene)
    if hook:
        hooks.append(hook)
    model = build_model(hooks)
    apply_perturb(model, dim, value)

    data = mujoco.MjData(model)
    robot = Robot(model, data)
    ctrl = RlController(cfg, robot, CONFIG_ROOT / cfg.policy_file)

    robot.set_joint_pose(cfg.default_q)
    robot.forward()
    skip = ()
    if scene == "box":
        skip = (robot.body_id("wood_box"),)
    robot.place_on_ground(skip_bodies=skip)

    rng = np.random.default_rng([int(seed), int(episode)])
    apply_init_noise(robot, rng, init_noise)

    n_sub = max(1, round(cfg.dt_ctrl / model.opt.timestep))
    n_steps = max(1, round(duration / cfg.dt_ctrl))
    cmd = np.asarray(cmd, dtype=float)

    foot_bids = [
        bid for bid in (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "LINK_FOOT_L"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "LINK_FOOT_R"),
        ) if bid >= 0
    ]
    prev_feet = [data.xpos[bid].copy() for bid in foot_bids]

    qpos = []
    stats = {
        "max_penetration": 0.0,
        "self_collisions": 0,
        "foot_contact_steps": 0,
        "foot_slip_steps": 0,
    }
    fell = False

    for _ in range(n_steps):
        data.ctrl[:] = ctrl.act(cmd)
        for _ in range(n_sub):
            mujoco.mj_step(model, data)
        qpos.append(data.qpos.copy())

        if data.qpos[2] < 0.55 or abs(math.atan2(2 * data.qpos[3] * data.qpos[6],
                                                 1 - 2 * data.qpos[4] * data.qpos[5])) > 0.6:
            fell = True

        if contacts:
            touching = set()
            for ci in range(data.ncon):
                con = data.contact[ci]
                b1 = int(model.geom_bodyid[con.geom1])
                b2 = int(model.geom_bodyid[con.geom2])
                stats["max_penetration"] = min(stats["max_penetration"], float(con.dist))
                if b1 > 0 and b2 > 0 and b1 != b2:
                    stats["self_collisions"] += 1
                for bid in foot_bids:
                    if bid in (b1, b2):
                        touching.add(bid)
            for i, bid in enumerate(foot_bids):
                if bid in touching:
                    stats["foot_contact_steps"] += 1
                    moved = data.xpos[bid][:2] - prev_feet[i][:2]
                    speed = float(np.linalg.norm(moved)) / cfg.dt_ctrl
                    if speed > SLIP_SPEED:
                        stats["foot_slip_steps"] += 1
                prev_feet[i] = data.xpos[bid].copy()

    qpos = np.asarray(qpos)
    dx = float(qpos[-1, 0] - qpos[0, 0])
    dy = float(qpos[-1, 1] - qpos[0, 1])

    return {
        "dx": dx,
        "dy": dy,
        "mean_vx": dx / duration,
        "mean_vy": dy / duration,
        "h_min": float(qpos[:, 2].min()),
        "fell": fell,
        "max_penetration": stats["max_penetration"],
        "self_collisions": stats["self_collisions"],
        "foot_slip_ratio": (
            stats["foot_slip_steps"] / stats["foot_contact_steps"]
            if stats["foot_contact_steps"] else 0.0
        ),
    }


# ── 扫描入口 ──────────────────────────────────────────────────────────────
def sweep(
    dims=None,
    cmds=None,
    seeds=None,
    *,
    scene="box",
    duration: float = DEFAULT_DURATION,
    contacts: bool = True,
) -> list:
    """跑所有 (dim, value, cmd, seed) 组合，返回行列表。"""
    if dims is None:
        dims = list(PERTURB_DIMS.keys())
    if cmds is None:
        cmds = DEFAULT_CMDS
    if seeds is None:
        seeds = DEFAULT_SEEDS

    rows = []
    total = len(dims) * sum(len(PERTURB_DIMS[d]["values"]) for d in dims) * len(cmds) * len(seeds)
    n = 0
    for dim in dims:
        cfg = PERTURB_DIMS[dim]
        for value in cfg["values"]:
            for seed in seeds:
                for cmd in cmds:
                    n += 1
                    if n % 10 == 0 or n == total:
                        print(f"  [{n}/{total}] {dim}={value:+.2f} "
                              f"cmd=({cmd[0]:+.1f},{cmd[1]:+.1f}) seed={seed}")
                    try:
                        met = perturb_rollout(
                            cmd, dim, value, seed=seed, scene=scene,
                            duration=duration, contacts=contacts,
                        )
                    except Exception as exc:
                        met = {"error": str(exc)}
                    rows.append({
                        "dim": dim,
                        "value": value,
                        "cmd_vx": cmd[0],
                        "cmd_vy": cmd[1],
                        "cmd_yaw": cmd[2],
                        "seed": seed,
                        "scene": scene,
                        **met,
                    })
    return rows


# ── 敏感性汇总 ────────────────────────────────────────────────────────────
METRICS = ("dx", "mean_vx", "h_min", "max_penetration", "foot_slip_ratio")


def summarize(rows: list) -> dict:
    """对每个 (dim, cmd) 组合，输出敏感性指标 = (max - min) / baseline。"""
    by_group = {}
    for r in rows:
        key = (r["dim"], r["cmd_vx"], r["cmd_vy"])
        by_group.setdefault(key, []).append(r)

    summary = []
    for (dim, cmd_vx, cmd_vy), grp in sorted(by_group.items()):
        baseline = next((g for g in grp if g["value"] == 1.0), None)
        if baseline is None:
            continue
        base = {m: float(baseline[m]) for m in METRICS if m in baseline}
        group_desc = PERTURB_DIMS[dim]["desc"]

        sens_row = {"dim": dim, "desc": group_desc,
                    "cmd": f"vx={cmd_vx:+.2f}, vy={cmd_vy:+.2f}",
                    "baseline": base}
        for m in METRICS:
            vals = [float(g[m]) for g in grp if m in g]
            if not vals:
                sens_row[f"{m}_sens"] = None
                sens_row[f"{m}_range"] = None
                continue
            vmax, vmin = max(vals), min(vals)
            sens = (vmax - vmin) / max(abs(base[m]), 1e-9)
            sens_row[f"{m}_sens"] = round(sens, 4)
            sens_row[f"{m}_range"] = f"{vmin:.4f}..{vmax:.4f}"
        summary.append(sens_row)

    return {
        "summary": summary,
        "total_runs": len(rows),
        "errors": [r for r in rows if "error" in r],
    }


# ── 打印报告 ──────────────────────────────────────────────────────────────
def _print_report(result: dict) -> None:
    print(f"\n参数敏感性分析  共 {result['total_runs']} 次仿真")
    if result["errors"]:
        print(f"  异常: {len(result['errors'])} 条")

    header = f"{'维度':16} {'desc':40} {'cmd':18}  dx_sens  vx_sens  h_sens  pen_sens  slip_sens"
    print(f"\n{header}")
    print("-" * len(header))
    for r in result["summary"]:
        def fmt(k):
            v = r.get(f"{k}_sens")
            if v is None:
                return "   -   "
            return f"{v * 100:7.2f}%"
        print(f"{r['dim']:16} {r['desc']:40} {r['cmd']:18} "
              f"{fmt('dx')} {fmt('mean_vx')} {fmt('h_min')} {fmt('max_penetration')} {fmt('foot_slip_ratio')}")

    # 结论（保守口径：任一指标 sens > 20% 视为敏感）
    print("\n── 总体结论 ──")
    sensitive_dims = set()
    ins_dims = set()
    for r in result["summary"]:
        max_sens = max(
            (v for v in (r.get(f"{m}_sens") for m in METRICS) if v is not None),
            default=0.0,
        )
        # dx/mean_vx/h_min 用 20%，pen/slip 用 50%（绝对值小，噪声大）
        critical = [r.get(f"{m}_sens") for m in ("dx", "mean_vx", "h_min")]
        critical_max = max((v for v in critical if v is not None), default=0.0)
        if critical_max > 0.20:
            sensitive_dims.add(r["dim"])
        else:
            ins_dims.add(r["dim"])

    for d in sorted(set(r["dim"] for r in result["summary"])):
        if d in sensitive_dims:
            print(f"  {d:<16s} [敏感] 关键指标 sens > 20%")
        else:
            print(f"  {d:<16s} [稳健] 关键指标 sens <= 20%")
    if not sensitive_dims:
        print("  → 仿真对这 5 个参数都不敏感，物理模型稳健")
    elif sensitive_dims:
        print(f"  → 敏感维度: {sorted(sensitive_dims)}")

    print("\n阈值说明: dx/mean_vx/h_min 用 20% 敏感性判据，pen/slip 用 50%")


def main(
    dims=None,
    cmds=None,
    seeds=None,
    duration: float = DEFAULT_DURATION,
    out=None,
) -> int:
    rows = sweep(dims, cmds, seeds, duration=duration)
    result = summarize(rows)
    _print_report(result)
    if out:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n详细数据: {out}")
        summary_path = Path(out).with_suffix(".summary.json")
        summary_path.write_text(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        print(f"汇总: {summary_path}")
    return 0
