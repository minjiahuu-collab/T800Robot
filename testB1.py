"""B1：指令-响应单调性（读 manifest.csv，不重跑仿真）。

通过标准：vy=yaw=0 时，mean_vx 随 cmd_vx 严格上升。
运行: python3 testB1.py
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "recordings" / "manifest.csv"
MIN_STEP = 0.01  # m/s，避免浮点噪声误判


def main() -> int:
    if not MANIFEST.is_file():
        print(f"找不到 {MANIFEST}")
        print("先运行: python3 -m t800_sim.cli manifest")
        return 1

    rows = list(csv.DictReader(MANIFEST.open()))
    groups = {}
    for r in rows:
        if abs(float(r["cmd_vy"])) > 1e-9 or abs(float(r["cmd_yaw"])) > 1e-9:
            continue
        groups.setdefault(float(r["cmd_vx"]), []).append(float(r["mean_vx"]))

    if len(groups) < 2:
        print(f"只有 {len(groups)} 组 cmd_vx，至少需要 2 组才能判断单调性")
        return 1

    xs = sorted(groups)
    means = [sum(groups[x]) / len(groups[x]) for x in xs]
    print("cmd_vx  样本  mean_vx")
    for x, m in zip(xs, means):
        print(f"{x:6.2f}  {len(groups[x]):4d}  {m:7.3f}")

    bad = [
        (a, b, ma, mb)
        for a, b, ma, mb in zip(xs, xs[1:], means, means[1:])
        if not mb > ma + MIN_STEP
    ]
    for a, b, ma, mb in bad:
        print(f"  违反: cmd {a:.2f}->{b:.2f} 时 mean_vx {ma:.3f}->{mb:.3f}"
              f"（未提升超过 {MIN_STEP}）")
    ok = not bad
    print("结论:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
