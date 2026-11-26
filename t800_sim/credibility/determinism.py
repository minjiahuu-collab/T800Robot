"""P1 确定性与数值收敛检查。

- 复跑一致性：同 seed/同配置跑 N 次，逐控制步比较 qpos/qvel/ctrl
- 步长收敛：timestep 减半后，位移与平均速度的相对漂移
"""

import json

import numpy as np

from .sim import base_metrics, rollout

REPEAT_TOL = 1e-9
DIVERGE_TOL = 0.05


def _max_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    return float(np.abs(a - b).max())


def check(duration: float = 2.0, cmd=(0.6, 0.0, 0.0), scene: str = "box",
          seed: int = 0, repeats: int = 2, timestep_ratio: float = 0.5,
          metric_tol: float = 0.02) -> dict:
    """跑复跑一致性 + 步长收敛，返回报告（ok 表示两项都通过）。"""
    kwargs = dict(duration=duration, cmd=cmd, scene=scene, seed=seed)
    base = rollout(**kwargs)

    repeat = {"qpos": 0.0, "qvel": 0.0, "ctrl": 0.0}
    for _ in range(max(0, int(repeats) - 1)):
        other = rollout(**kwargs)
        repeat["qpos"] = max(repeat["qpos"], _max_diff(base["qpos"], other["qpos"]))
        repeat["qvel"] = max(repeat["qvel"], _max_diff(base["qvel"], other["qvel"]))
        repeat["ctrl"] = max(repeat["ctrl"], _max_diff(base["ctrl"], other["ctrl"]))
    repeat["max"] = max(repeat.values())
    repeat_ok = repeat["max"] <= REPEAT_TOL

    fine = rollout(timestep=base["timestep"] * timestep_ratio, **kwargs)
    m_coarse, m_fine = base_metrics(base), base_metrics(fine)
    rel = {
        key: abs(m_fine[key] - m_coarse[key]) / max(abs(m_coarse[key]), 1e-6)
        for key in ("dx", "mean_vx")
    }
    convergence_ok = all(v <= metric_tol for v in rel.values())

    divergence_step = -1
    if fine["qpos"].shape == base["qpos"].shape:
        diff = np.abs(fine["qpos"] - base["qpos"]).max(axis=1)
        idx = np.where(diff > DIVERGE_TOL)[0]
        if idx.size:
            divergence_step = int(idx[0])

    return {
        "ok": bool(repeat_ok and convergence_ok),
        "repeat": {**repeat, "ok": repeat_ok, "repeats": int(repeats)},
        "convergence": {
            "ok": convergence_ok,
            "timestep": [base["timestep"], fine["timestep"]],
            "relative": rel,
            "coarse": m_coarse,
            "fine": m_fine,
            "divergence_step": divergence_step,
        },
    }


def main(duration: float = 2.0, cmd=(0.6, 0.0, 0.0), scene: str = "box",
         seed: int = 0, repeats: int = 2, timestep_ratio: float = 0.5,
         metric_tol: float = 0.02, out=None) -> int:
    report = check(duration, cmd, scene, seed, repeats, timestep_ratio, metric_tol)
    rep = report["repeat"]
    conv = report["convergence"]

    print(f"场景: {scene}, cmd=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}), "
          f"seed={seed}, duration={duration:.1f}s")
    print(f"[P1] 复跑 {rep['repeats']} 次: max|Δqpos|={rep['qpos']:.3e}, "
          f"|Δqvel|={rep['qvel']:.3e}, |Δctrl|={rep['ctrl']:.3e} "
          f"-> {'PASS' if rep['ok'] else 'FAIL'}（阈值 {REPEAT_TOL:g}）")
    print(f"[P1] 步长收敛 {conv['timestep'][0] * 1e3:.2f}ms -> "
          f"{conv['timestep'][1] * 1e3:.2f}ms: "
          f"dx {conv['coarse']['dx']:+.3f} -> {conv['fine']['dx']:+.3f}m "
          f"(Δ{conv['relative']['dx'] * 100:.2f}%), "
          f"mean_vx {conv['coarse']['mean_vx']:+.3f} -> {conv['fine']['mean_vx']:+.3f}m/s "
          f"(Δ{conv['relative']['mean_vx'] * 100:.2f}%) "
          f"-> {'PASS' if conv['ok'] else 'FAIL'}（阈值 {metric_tol * 100:.0f}%）")
    if conv["divergence_step"] >= 0:
        print(f"     状态在控制步 {conv['divergence_step']} 首次分叉（> {DIVERGE_TOL}），"
              f"混沌系统正常，以标量指标为准")
    print(f"结论: {'PASS' if report['ok'] else 'FAIL'}")

    if out:
        with open(out, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"报告: {out}")
    return 0 if report["ok"] else 1
