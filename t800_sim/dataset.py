"""数据集级 manifest 与 train/val 划分。

扫描 record-grid 产出的 session（含 meta.json 的目录），汇总每条 episode 的
场景、指令、指标与校验结果，写成 manifest.csv，并按 seed 分组（默认）或按
session 哈希做确定性的 train/val 划分，额外输出 split.json 汇总。

按 seed 分组可避免同一初始条件同时出现在训练集和验证集，适合小数据量。

用法:
    python3 -m t800_sim.cli manifest
    python3 -m t800_sim.cli manifest --root recordings --val-ratio 0.3 --seed 0
    python3 -m t800_sim.cli manifest --split-by session --val-ratio 0.2
"""

import csv
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

from .validate import find_sessions, validate_session

FIELDS = (
    "session", "task", "scene", "seed", "episode",
    "randomized", "plan_hash", "rand_session",
    "mass_scale", "ground_friction", "robot_friction_scale",
    "solref_timeconst_scale", "joint_damping_scale", "torque_limit_scale",
    "kp_scale", "kd_scale",
    "fovy_scale", "light_intensity_scale", "texture_swap_prob",
    "cmd_vx", "cmd_vy", "cmd_yaw",
    "frames", "duration", "mean_vx", "final_dx", "fell", "success",
    "cameras", "schema_version", "policy_sha256", "git_commit",
    "valid", "size_mb", "split",
)


def _dir_size_mb(path: Path) -> float:
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return round(total / (1024 * 1024), 2)


def _row(root: Path, session: Path) -> dict:
    meta = json.loads((session / "meta.json").read_text())
    metrics = meta.get("metrics") or {}
    cmd = meta.get("cmd") or [None, None, None]
    report = validate_session(session)
    policy = meta.get("policy") or {}
    rand = meta.get("randomization") or {}
    rand_on = bool(rand.get("enabled"))
    rparams = rand.get("params") or {}
    rdyn = rparams.get("dynamics") or {}
    rcon = rparams.get("contact") or {}
    ract = rparams.get("actuator") or {}
    rcam = rparams.get("camera") or {}
    rren = rparams.get("rendering") or {}
    rplan = rand.get("plan") or {}
    # 逐 body 质量缩放是向量；manifest 单列用均值代表。
    mass_scale = rdyn.get("mass_scale", "")
    if isinstance(mass_scale, (list, tuple)):
        mass_scale = (
            round(sum(float(v) for v in mass_scale) / len(mass_scale), 6)
            if mass_scale else ""
        )
    return {
        "session": str(session.relative_to(root)),
        "task": meta.get("task", ""),
        "scene": meta.get("scene_id", ""),
        "seed": meta.get("seed", -1),
        "episode": meta.get("episode", 0),
        "randomized": rand_on,
        "plan_hash": rplan.get("plan_hash", "") if rand_on else "",
        "rand_session": rplan.get("session", "") if rand_on else "",
        "mass_scale": mass_scale,
        "ground_friction": rcon.get("ground_friction", ""),
        "robot_friction_scale": rcon.get("robot_friction_scale", ""),
        "solref_timeconst_scale": rcon.get("solref_timeconst_scale", ""),
        "joint_damping_scale": ract.get("joint_damping_scale", ""),
        "torque_limit_scale": ract.get("torque_limit_scale", ""),
        "kp_scale": ract.get("kp_scale", ""),
        "kd_scale": ract.get("kd_scale", ""),
        "fovy_scale": rcam.get("fovy_scale", ""),
        "light_intensity_scale": rren.get("light_intensity_scale", ""),
        "texture_swap_prob": rren.get("texture_swap_prob", ""),
        "cmd_vx": cmd[0],
        "cmd_vy": cmd[1],
        "cmd_yaw": cmd[2],
        "frames": meta.get("frames", 0),
        "duration": round(float(metrics.get("duration", meta.get("duration", 0.0))), 3),
        "mean_vx": round(float(metrics.get("mean_vx", 0.0)), 4),
        "final_dx": round(float(metrics.get("final_dx", 0.0)), 4),
        "fell": metrics.get("fell", ""),
        "success": metrics.get("success", ""),
        "cameras": len(meta.get("cameras") or []),
        "schema_version": meta.get("schema_version", ""),
        "policy_sha256": policy.get("sha256", ""),
        "git_commit": meta.get("git_commit", ""),
        "valid": report["ok"],
        "size_mb": _dir_size_mb(session),
        "split": "",
    }


def _seed_split(rows, val_ratio, seed):
    """按 seed 分组划分：同一初始条件只会落在 train 或 val 一侧。"""
    seeds = sorted({int(r["seed"]) for r in rows})
    if len(seeds) < 2:
        return set(), seeds
    n_val = max(1, math.ceil(len(seeds) * val_ratio))
    n_val = min(n_val, len(seeds) - 1)
    rng = random.Random(seed)
    val_seeds = set(rng.sample(seeds, n_val))
    return val_seeds, seeds


def _session_split(rows, val_ratio, seed):
    """按 session 路径哈希划分：同条 session 结果稳定，但对 seed 不做隔离。"""
    out = set()
    for r in rows:
        digest = hashlib.sha256(f"{seed}:{r['session']}".encode()).hexdigest()
        if int(digest[:8], 16) / 0xFFFFFFFF < val_ratio:
            out.add(r["session"])
    return out


def build_manifest(root="recordings", *, val_ratio: float = 0.3,
                   split_by: str = "seed", seed: int = 0,
                   out=None, split_out=None) -> dict:
    """扫描 root 下所有 session，写 manifest.csv 与 split.json，返回汇总。"""
    root = Path(root)
    sessions = find_sessions(root)
    if not sessions:
        raise FileNotFoundError(f"{root} 下没有找到 session（缺少 meta.json）")

    rows = [_row(root, s) for s in sessions]
    rows.sort(key=lambda r: (r["task"], int(r["seed"]), int(r["episode"])))

    valid_rows = [r for r in rows if r["valid"]]
    if split_by == "seed":
        val_seeds, all_seeds = _seed_split(valid_rows, val_ratio, seed)
        for r in valid_rows:
            r["split"] = "val" if int(r["seed"]) in val_seeds else "train"
        split_info = {"by": "seed", "val_seeds": sorted(val_seeds),
                      "all_seeds": all_seeds}
    elif split_by == "session":
        val_sessions = _session_split(valid_rows, val_ratio, seed)
        for r in valid_rows:
            r["split"] = "val" if r["session"] in val_sessions else "train"
        split_info = {"by": "session"}
    else:
        raise ValueError(f"未知划分方式: {split_by}（可选 seed / session）")

    out_path = Path(out) if out else root / "manifest.csv"
    split_path = Path(split_out) if split_out else root / "split.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    by_task = Counter((r["task"], r["split"]) for r in valid_rows)
    policies = {r["policy_sha256"] for r in rows if r["policy_sha256"]}
    randomized = [r for r in rows if r["randomized"]]
    summary = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(root),
        "schema_version": "1.0",
        "sessions": len(rows),
        "randomized_sessions": len(randomized),
        "valid_sessions": len(valid_rows),
        "train_sessions": len(train),
        "val_sessions": len(val),
        "frames": sum(int(r["frames"]) for r in rows),
        "size_mb": round(sum(float(r["size_mb"]) for r in rows), 1),
        "success_rate": round(
            sum(bool(r["success"]) for r in rows) / len(rows), 4
        ) if rows else 0.0,
        "policy_sha256": sorted(policies),
        "split": {
            "val_ratio": val_ratio,
            "seed": seed,
            **split_info,
            "train": [r["session"] for r in train],
            "val": [r["session"] for r in val],
        },
        "by_task": {
            task: {split: by_task.get((task, split), 0) for split in ("train", "val")}
            for task, _ in sorted(by_task)
        },
    }
    split_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"数据集: {root} -> {out_path}")
    print(f"session: {len(rows)} 条（有效 {len(valid_rows)}），"
          f"{summary['frames']} 帧，{summary['size_mb']} MB，"
          f"success={summary['success_rate'] * 100:.1f}%")
    if randomized:
        masses = [float(r["mass_scale"]) for r in randomized if r["mass_scale"] != ""]
        mass_txt = f"{min(masses):.3f}~{max(masses):.3f}" if masses else "-"
        print(f"随机化: {len(randomized)}/{len(rows)} 条，mass_scale {mass_txt}")
    if 0 < len(randomized) < len(rows):
        print("警告: manifest 混有随机化与非随机化 session，训练前请确认是否需要隔离")
    if len(policies) > 1:
        print(f"警告: manifest 内包含 {len(policies)} 个不同策略版本")
    print(f"划分: by={split_by}, val_ratio={val_ratio}, seed={seed} -> "
          f"train {len(train)} / val {len(val)}")
    if split_info.get("val_seeds") is not None:
        print(f"验证集 seeds: {split_info['val_seeds']}")
    for task in summary["by_task"]:
        counts = summary["by_task"][task]
        print(f"  {task:32s} train={counts['train']:3d} val={counts['val']:3d}")
    if len(valid_rows) != len(rows):
        print(f"警告: {len(rows) - len(valid_rows)} 条 session 校验未通过，未参与划分")
    return summary


def main(root="recordings", **kwargs) -> None:
    build_manifest(root, **kwargs)
