"""P2 物理合理性检查。

两种模式:
  lint_session  只读 CSV 的静态检查（时间戳、数值有限性、四元数、基座高度、
                关节限位、控制量范围/饱和、关节角速度），可直接跑已有数据
  lint_rollout  重仿真一遍，额外检查自碰撞、接触穿透、站立相足端滑移
"""

import csv
import json
from pathlib import Path

import mujoco
import numpy as np

from ..model import load_model
from ..validate import find_sessions
from .sim import base_metrics, rollout

PENETRATION_ERROR = -0.020   # 接触穿透超过 20 mm 判硬错误
PENETRATION_WARN = -0.010    # 默认接触刚度下平地行走稳态穿透约 5~9 mm
SLIP_RATIO_WARN = 0.2        # 站立相滑移帧占比
SATURATION_RATIO_WARN = 0.2  # 控制量贴限幅帧占比
JOINT_SPEED_WARN = 60.0      # rad/s
BASE_Z_MIN = 0.3             # m，低于此高度视为摔倒/异常
QUAT_TOL = 1e-3
LIMIT_TOL = 1e-3
JOINT_LIMIT_ERROR = 0.02      # rad，超过视为硬违规；1e-3~0.02 属于软限位合规量


def _read_states(path: Path):
    with open(path / "states.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def _col(rows, key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows], dtype=float)


def _issue(bucket: list, check: str, detail: str, frame=None) -> None:
    item = {"check": check, "detail": detail}
    if frame is not None:
        item["frame"] = int(frame)
    bucket.append(item)


def lint_session(path, model=None) -> dict:
    """对一条已落盘 session 做静态物理合理性检查。"""
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    rows = _read_states(path)
    joints = list(meta.get("joints") or [])
    errors: list = []
    warnings: list = []
    stats = {"frames": len(rows)}
    if not rows:
        _issue(errors, "states", "states.csv 为空")
        return {"session": str(path), "ok": False, "errors": errors,
                "warnings": warnings, "stats": stats}

    model = model or load_model()
    t = _col(rows, "time")
    render_t = _col(rows, "render_time")

    if len(t) > 1:
        dt = np.diff(t)
        if np.any(dt <= 0):
            _issue(errors, "time_monotonic", "仿真时间出现非递增",
                   int(np.argmax(dt <= 0)))
        else:
            expected = 1.0 / float(meta.get("fps") or 30.0)
            if np.max(np.abs(dt - expected)) > 0.5 * expected:
                _issue(warnings, "time_step", f"采样间隔偏离 1/fps 超过 50%")
        if np.any(np.diff(render_t) < -1e-6):
            _issue(warnings, "render_time", "墙钟时间出现回退")

    keys = (["base_x", "base_y", "base_z", "qw", "qx", "qy", "qz"]
            + [f"ctrl_{j}" for j in joints] + joints)
    mat = np.column_stack([_col(rows, k) for k in keys])
    bad = ~np.isfinite(mat)
    if bad.any():
        _issue(errors, "finite", f"存在 {int(bad.sum())} 个非有限数值",
               int(np.argmax(bad.any(axis=1))))

    quat = mat[:, 3:7]
    bad_q = np.abs(np.linalg.norm(quat, axis=1) - 1.0) > QUAT_TOL
    if bad_q.any():
        _issue(errors, "quat_norm", f"{int(bad_q.sum())} 帧基座四元数模长偏离 1",
               int(np.argmax(bad_q)))

    z_min = float(mat[:, 2].min())
    stats["base_z_min"] = z_min
    if z_min < BASE_Z_MIN:
        _issue(errors, "base_height", f"基座最低 {z_min:.3f} m < {BASE_Z_MIN} m",
               int(np.argmin(mat[:, 2])))

    for idx, name in enumerate(joints):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0 or not model.jnt_limited[jid]:
            continue
        lo, hi = model.jnt_range[jid]
        col = mat[:, 7 + len(joints) + idx]
        exceed = np.maximum(lo - col, col - hi)
        worst = float(exceed.max())
        if worst > LIMIT_TOL:
            frames = int((exceed > LIMIT_TOL).sum())
            level = errors if worst > JOINT_LIMIT_ERROR else warnings
            _issue(level, "joint_limit",
                   f"{name} 越界 {frames} 帧，最大 {worst:.3f} rad "
                   f"(范围 [{lo:.3f}, {hi:.3f}])",
                   int(np.argmax(exceed)))

    n_ctrl = min(len(joints), model.nu)
    if n_ctrl:
        lo = model.actuator_ctrlrange[:n_ctrl, 0]
        hi = model.actuator_ctrlrange[:n_ctrl, 1]
        ctrl = mat[:, 7:7 + n_ctrl]
        out = (ctrl < lo - 1e-6) | (ctrl > hi + 1e-6)
        if out.any():
            _issue(errors, "ctrl_range", f"{int(out.sum())} 个控制量超出执行器范围",
                   int(np.argmax(out.any(axis=1))))
        sat = np.isclose(ctrl, hi, atol=1e-6) | np.isclose(ctrl, lo, atol=1e-6)
        sat_ratio = float(sat.mean())
        stats["ctrl_saturation"] = round(sat_ratio, 4)
        if sat_ratio > SATURATION_RATIO_WARN:
            _issue(warnings, "ctrl_saturation",
                   f"控制量贴限幅占比 {sat_ratio * 100:.1f}% > "
                   f"{SATURATION_RATIO_WARN * 100:.0f}%")

    joint_q = mat[:, 7 + len(joints):]
    if len(t) > 2 and joint_q.size and np.all(np.diff(t) > 0):
        speed = np.abs(np.gradient(joint_q, t, axis=0))
        max_speed = float(speed.max())
        stats["joint_speed_max"] = round(max_speed, 2)
        if max_speed > JOINT_SPEED_WARN:
            idx = np.unravel_index(int(np.argmax(speed)), speed.shape)
            _issue(warnings, "joint_speed",
                   f"最大关节角速度 {max_speed:.1f} rad/s > {JOINT_SPEED_WARN}",
                   int(idx[0]))

    if meta.get("gt_object") and (path / "ground_truth.csv").is_file():
        with open(path / "ground_truth.csv", newline="") as f:
            gt = list(csv.DictReader(f))
        if gt:
            obj_z = np.array([float(r["obj_z"]) for r in gt])
            stats["obj_z_min"] = round(float(np.nanmin(obj_z)), 4)
            if np.nanmin(obj_z) < -0.01:
                _issue(errors, "gt_object",
                       f"真值物体最低 z={np.nanmin(obj_z):.3f} m 低于地面")
            quat_n = np.linalg.norm(
                np.column_stack([np.array([float(r[c]) for r in gt])
                                 for c in ("obj_qw", "obj_qx", "obj_qy", "obj_qz")]),
                axis=1,
            )
            if np.any(np.abs(quat_n - 1.0) > QUAT_TOL):
                _issue(warnings, "gt_quat", "真值物体四元数模长偏离 1")

    return {
        "session": str(path),
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }


def lint_rollout(meta: dict) -> dict:
    """用 session 的 cmd/seed 重仿真一遍，检查接触层面的合理性。"""
    run = rollout(
        duration=float(meta.get("duration") or 5.0),
        cmd=meta.get("cmd") or (0.6, 0.0, 0.0),
        scene=meta.get("scene_id") or "box",
        seed=int(meta.get("seed", 0)),
        episode=int(meta.get("episode", 0)),
        init_noise=meta.get("init_noise") or (0.0, 0.0, 0.0),
        contacts=True,
    )
    errors: list = []
    warnings: list = []
    st = run["contacts"]
    if st["self_collisions"] > 0:
        _issue(errors, "self_collision", f"发生 {st['self_collisions']} 次自碰撞")
    if st["max_penetration"] < PENETRATION_ERROR:
        _issue(errors, "penetration",
               f"最大穿透 {st['max_penetration'] * 1e3:.1f} mm "
               f"< {PENETRATION_ERROR * 1e3:.1f} mm")
    elif st["max_penetration"] < PENETRATION_WARN:
        _issue(warnings, "penetration",
               f"最大穿透 {st['max_penetration'] * 1e3:.1f} mm")

    contact_steps = st["foot_contact_steps"]
    slip_ratio = st["foot_slip_steps"] / contact_steps if contact_steps else 0.0
    if slip_ratio > SLIP_RATIO_WARN:
        _issue(warnings, "foot_slip",
               f"站立相足端滑移占比 {slip_ratio * 100:.1f}%")

    ctrl = run["ctrl"]
    model = run["model"]
    lo = model.actuator_ctrlrange[:, 0]
    hi = model.actuator_ctrlrange[:, 1]
    sat_ratio = float((np.isclose(ctrl, hi, atol=1e-6)
                       | np.isclose(ctrl, lo, atol=1e-6)).mean())
    if sat_ratio > SATURATION_RATIO_WARN:
        _issue(warnings, "ctrl_saturation",
               f"重仿真控制量贴限幅占比 {sat_ratio * 100:.1f}%")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "stats": {
            **st,
            "foot_slip_ratio": round(slip_ratio, 4),
            "ctrl_saturation": round(sat_ratio, 4),
            **base_metrics(run),
        },
    }


def main(session="recordings", deep: int = 0, out=None) -> int:
    """检查单个 session 或根目录下所有 session，返回进程退出码。"""
    sessions = find_sessions(session)
    if not sessions:
        print(f"未找到 session: {session}")
        return 1

    model = load_model()
    reports = []
    for i, path in enumerate(sessions):
        rep = lint_session(path, model)
        if i < int(deep):
            meta = json.loads((path / "meta.json").read_text())
            try:
                rep["deep"] = lint_rollout(meta)
            except Exception as exc:  # 重仿真失败本身是可复现性问题
                rep["deep"] = {"ok": False, "errors": [
                    {"check": "deep_rollout", "detail": f"重仿真失败: {exc}"}],
                    "warnings": [], "stats": {}}
        reports.append(rep)

        total_err = len(rep["errors"]) + len(rep.get("deep", {}).get("errors", []))
        total_warn = len(rep["warnings"]) + len(rep.get("deep", {}).get("warnings", []))
        status = "FAIL" if total_err else ("WARN" if total_warn else "PASS")
        print(f"[{status}] {rep['session']}  "
              f"errors={total_err} warnings={total_warn} frames={rep['stats']['frames']}")
        for item in rep["errors"] + rep.get("deep", {}).get("errors", []):
            print(f"    ! {item['check']}: {item['detail']}")
        for item in rep["warnings"] + rep.get("deep", {}).get("warnings", []):
            print(f"    ~ {item['check']}: {item['detail']}")

    total_errors = sum(len(r["errors"]) + len(r.get("deep", {}).get("errors", []))
                       for r in reports)
    total_warnings = sum(len(r["warnings"]) + len(r.get("deep", {}).get("warnings", []))
                         for r in reports)
    ok_sessions = sum(
        1 for r in reports
        if not r["errors"] and not r.get("deep", {}).get("errors")
    )
    print(f"\n{ok_sessions}/{len(reports)} 个 session 无硬违规，"
          f"errors={total_errors}, warnings={total_warnings}")

    if out:
        Path(out).write_text(json.dumps(reports, ensure_ascii=False, indent=2))
        print(f"报告: {out}")
    return 0 if total_errors == 0 else 1
