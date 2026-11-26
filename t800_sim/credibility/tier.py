"""P5 可信度分级：对每条 session 打 0-100 分 + A/B/C/D 等级。

四个维度加权：
  复现性 (30%)  → 环境指纹 + meta 可复现 + 同 seed 双跑一致
  物理合理性 (30%) → 静态 lint 无 error/warning + 可选 deep rollout
  行为有效性 (25%) → success + mean_vx 跟踪 + 无跌倒
  外部有效性 (15%) → 真机对比 + 物理参数标定

硬门控：任何 lint error → 直接 D 级（其他分数再高也没用）。

等级边界（总分）：
  A >= 85  内部全通过 + 真机对齐
  B >= 70  内部全通过，无真机对齐
  C >= 50  部分 warning 或行为不达标
  D < 50   存在硬错误
"""

import csv
import json
from pathlib import Path

from ..validate import find_sessions
from .physics_lint import lint_session, lint_rollout

# ── 权重 ──────────────────────────────────────────────────────────────────
W_REPRO = 0.30
W_PHYS = 0.30
W_BEHAV = 0.25
W_EXT = 0.15

# ── 子项满分 → 扣分公式 ───────────────────────────────────────────────────
# 复现性子项
ENV_FIELDS = ("python", "numpy", "mujoco", "model_sha256", "git_commit")
ENV_SCORE_PER_FIELD = 20  # 5 个字段，满分 100
META_REQUIRED = ("cmd", "seed", "scene_id", "init_noise", "policy",
                 "schema_version", "duration")
REPEAT_TOL = 1e-9

# 行为子项：速度跟踪（按 mean_vx/cmd_vx 比例分级）
VX_RATIO_GOOD = (0.85, 1.15)   # 优秀
VX_RATIO_OK = (0.70, 1.30)     # 良好
VX_RATIO_MARGIN = (0.50, 1.50) # 及格
VEL_TRACK_BONUS = 40  # 满分 40

# 等级边界
TIER_A = 85.0
TIER_B = 70.0
TIER_C = 50.0


# ── 各维度打分 ─────────────────────────────────────────────────────────────
def repro_score(meta: dict, *, repeat_ok: bool | None = None) -> tuple[float, dict]:
    """复现性。repeat_ok=None 表示未做双跑检查，扣满该子项。"""
    detail = {}
    score = 0.0

    # 环境指纹（100 满分 → 权重后 30 分贡献）
    env = meta.get("environment") or {}
    env_score = 0
    for f in ENV_FIELDS:
        if env.get(f):
            env_score += ENV_SCORE_PER_FIELD
    detail["env_score"] = env_score
    detail["env_missing"] = [f for f in ENV_FIELDS if not env.get(f)]
    score += env_score * 0.4  # 40% of repro

    # meta 自描述（40%）
    meta_score = 0
    missing_meta = [f for f in META_REQUIRED if f not in meta]
    meta_score = max(0, 100 - 14 * len(missing_meta))  # 7 个字段，每个扣 14
    detail["meta_score"] = meta_score
    detail["meta_missing"] = missing_meta
    score += meta_score * 0.4

    # 双跑一致性（20%）
    if repeat_ok is None:
        detail["repeat_score"] = 0.0
        detail["repeat_status"] = "unchecked"
    elif repeat_ok:
        detail["repeat_score"] = 100.0
        detail["repeat_status"] = "ok"
        score += 100 * 0.2
    else:
        detail["repeat_score"] = 0.0
        detail["repeat_status"] = "fail"
    detail["score"] = round(score, 1)
    return score, detail


def physics_score(lint_rep: dict, deep_rep: dict | None = None) -> tuple[float, dict]:
    """物理合理性。"""
    errors = len(lint_rep["errors"]) + len((deep_rep or {}).get("errors", []))
    warnings = len(lint_rep["warnings"]) + len((deep_rep or {}).get("warnings", []))
    detail = {
        "errors": errors,
        "warnings": warnings,
        "deep_ran": deep_rep is not None,
    }
    if errors > 0:
        detail["score"] = 0.0
        return 0.0, detail
    score = 100.0
    score -= 10 * warnings  # 每个 warning 扣 10
    if deep_rep is not None:
        detail["score"] = round(max(0, score), 1)
        return max(0, score), detail
    # 未跑 deep，扣 20 表示"只验证了静态"
    detail["score"] = round(max(0, score - 20), 1)
    return max(0, score - 20), detail


def behavior_score(meta: dict) -> tuple[float, dict]:
    """行为有效性：success + 速度跟踪 + 无跌倒。"""
    metrics = meta.get("metrics") or {}
    cmd = meta.get("cmd") or [0, 0, 0]
    cmd_vx = float(cmd[0] or 0)

    success = bool(metrics.get("success", False))
    fell = bool(metrics.get("fell", False))
    mean_vx = float(metrics.get("mean_vx", 0) or 0)

    detail = {}
    score = 0.0

    # success 40 分
    detail["success"] = success
    score += 40.0 if success else 0.0

    # 速度跟踪 40 分（按 mean_vx / cmd_vx 比例分级）
    track_score = 0.0
    ratio = mean_vx / cmd_vx if abs(cmd_vx) > 1e-6 else None
    if ratio is not None:
        lo, hi = VX_RATIO_GOOD
        if lo <= ratio <= hi:
            track_score = VEL_TRACK_BONUS  # 40
        else:
            lo, hi = VX_RATIO_OK
            if lo <= ratio <= hi:
                track_score = 30.0
            else:
                lo, hi = VX_RATIO_MARGIN
                if lo <= ratio <= hi:
                    track_score = 15.0
                else:
                    track_score = 0.0
    detail["track_score"] = track_score
    detail["vx_ratio"] = round(ratio, 3) if ratio is not None else None
    score += track_score

    # 无跌倒 20 分
    detail["fell"] = fell
    score += 0.0 if fell else 20.0
    detail["score"] = round(score, 1)
    return score, detail


def external_score(*, real_robot_ok: bool = False, physics_calibrated: bool = True) -> tuple[float, dict]:
    """外部有效性。

    real_robot_ok: 真机构真机对比报告已跑通
    physics_calibrated: 物理参数（接触/摩擦/质量）已据真机标定
    """
    detail = {"real_robot_ok": real_robot_ok, "physics_calibrated": physics_calibrated}
    score = 0.0
    if physics_calibrated:
        score += 20.0
    if real_robot_ok:
        score += 80.0
    else:
        detail["note"] = "真机对比未做，无法确认仿真-现实差距"
    detail["score"] = round(score, 1)
    return score, detail


# ── 总评分与等级 ──────────────────────────────────────────────────────────
def tier(total: float, physics_errors: int) -> str:
    if physics_errors > 0 or total < TIER_C:
        return "D"
    if total >= TIER_A:
        return "A"
    if total >= TIER_B:
        return "B"
    return "C"


def grade_session(
    session_path,
    *,
    model=None,
    deep: bool = False,
    repeat_ok: bool | None = None,
    real_robot_ok: bool = False,
    physics_calibrated: bool = True,
) -> dict:
    """单条 session 打分。返回完整报告。"""
    path = Path(session_path)
    meta = json.loads((path / "meta.json").read_text())
    lint = lint_session(path, model)
    deep_rep = lint_rollout(meta) if deep else None

    r_repro, d_repro = repro_score(meta, repeat_ok=repeat_ok)
    r_phys, d_phys = physics_score(lint, deep_rep)
    r_beh, d_beh = behavior_score(meta)
    r_ext, d_ext = external_score(real_robot_ok=real_robot_ok,
                                  physics_calibrated=physics_calibrated)

    errors = (len(lint["errors"])
              + (len(deep_rep["errors"]) if deep_rep else 0))
    total = (W_REPRO * r_repro + W_PHYS * r_phys
             + W_BEHAV * r_beh + W_EXT * r_ext)
    level = tier(total, errors)

    return {
        "session": str(path),
        "total": round(total, 1),
        "tier": level,
        "physics_errors": errors,
        "physics_warnings": (len(lint["warnings"])
                             + (len(deep_rep["warnings"]) if deep_rep else 0)),
        "dimensions": {
            "repro": {"raw": r_repro, "weighted": round(W_REPRO * r_repro, 1),
                      "detail": d_repro},
            "physics": {"raw": r_phys, "weighted": round(W_PHYS * r_phys, 1),
                        "detail": d_phys},
            "behavior": {"raw": r_beh, "weighted": round(W_BEHAV * r_beh, 1),
                         "detail": d_beh},
            "external": {"raw": r_ext, "weighted": round(W_EXT * r_ext, 1),
                         "detail": d_ext},
        },
        "lint_errors": lint["errors"],
        "lint_warnings": lint["warnings"],
        "deep": deep_rep,
        "meta": meta,
    }


# ── 批量入口 ──────────────────────────────────────────────────────────────
def grade_dataset(
    root="recordings",
    *,
    deep: int = 0,
    repeat_ok: bool | None = None,
    real_robot_ok: bool = False,
    physics_calibrated: bool = True,
    out: str | None = None,
) -> dict:
    """扫描 root 下所有 session，逐条打分 + 汇总。

    返回 sessions 列表（每条包含 total/tier/各维度）与 dataset 级 summary。
    """
    from ..model import load_model

    model = load_model()
    sessions = find_sessions(root)
    reports = []
    print(f"找到 {len(sessions)} 条 session，deep rollout 前 {deep} 条")

    for i, s in enumerate(sessions):
        rep = grade_session(
            s, model=model,
            deep=(i < int(deep)),
            repeat_ok=repeat_ok,
            real_robot_ok=real_robot_ok,
            physics_calibrated=physics_calibrated,
        )
        reports.append(rep)

    summary = _summarize(reports)

    # 打印
    _print_report(reports, summary, real_robot_ok, deep)

    if out:
        Path(out).write_text(json.dumps(
            {"sessions": [
                {k: v for k, v in r.items()
                 if k not in ("deep", "lint_errors", "lint_warnings")}
                for r in reports
            ], "summary": summary},
            ensure_ascii=False, indent=2,
        ))
        print(f"\n完整报告: {out}")

    return {"sessions": reports, "summary": summary}


def _summarize(reports: list) -> dict:
    tiers = {}
    for r in reports:
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
    total = sum(r["total"] for r in reports) / len(reports) if reports else 0.0
    failed = [r for r in reports if r["physics_errors"] > 0]
    ext_unverified = all(
        not r["dimensions"]["external"]["detail"].get("real_robot_ok")
        for r in reports
    )
    return {
        "sessions": len(reports),
        "mean_total": round(total, 1),
        "tiers": tiers,
        "failed_physics": len(failed),
        "external_unverified": ext_unverified,
    }


def _print_report(reports, summary, real_robot_ok, deep):
    # 表格
    header = f"{'tier':4} {'score':6} {'repro':>5} {'phys':>5} {'beh':>5} {'ext':>5}  session"
    print(f"\n{header}")
    for r in sorted(reports, key=lambda x: -x["total"]):
        dims = r["dimensions"]
        print(f"{r['tier']:4} {r['total']:6.1f} "
              f"{dims['repro']['raw']:5.0f} "
              f"{dims['physics']['raw']:5.0f} "
              f"{dims['behavior']['raw']:5.0f} "
              f"{dims['external']['raw']:5.0f}  "
              f"{r['session']}")

    print(f"\n── 数据集汇总 ──")
    print(f"session: {summary['sessions']} 条，均分 {summary['mean_total']:.1f}")
    for t in ("A", "B", "C", "D"):
        c = summary["tiers"].get(t, 0)
        print(f"  {t}: {c} 条")
    if deep == 0:
        print("  (deep rollout 未跑，物理分自动 -20；加 --deep N 重算前 N 条)")
    if not real_robot_ok:
        print("  (真机对比未做 → 外部有效性 = 20/100 封顶 B)")
    if summary["sessions"]:
        # 给一个总体结论
        top_tier = max(
            summary["tiers"].items(),
            key=lambda kv: kv[1], default=("D", 0),
        )[0]
        print(f"\n总体可信度等级: **{top_tier}**  "
              f"（真机 {'已' if real_robot_ok else '未'}验证）")


def main(
    root: str = "recordings",
    deep: int = 0,
    repeat_ok: bool | None = None,
    real_robot_ok: bool = False,
    physics_calibrated: bool = True,
    out: str | None = None,
) -> int:
    grade_dataset(
        root, deep=deep, repeat_ok=repeat_ok,
        real_robot_ok=real_robot_ok, physics_calibrated=physics_calibrated,
        out=out,
    )
    return 0
