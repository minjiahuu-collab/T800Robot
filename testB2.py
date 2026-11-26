"""B2：零指令应基本静止。

跑 3 s 零速度指令，检查横向/纵向漂移与是否跌倒。
通过标准：|dx| <= 0.2 m 且 |dy| <= 0.2 m 且未跌倒。
运行: python3 testB2.py
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = Path("/tmp/zero_cmd")
SESSION = OUT / "zero" / "box" / "seed_0000" / "ep_0000"
DX_MAX = 0.2
DY_MAX = 0.2


def main() -> int:
    cmd = [
        sys.executable, "-m", "t800_sim.cli", "record",
        "--duration", "3", "--fps", "10", "--width", "160", "--height", "120",
        "--cmd", "0", "0", "0", "--task", "zero", "--seed", "0", "--out", str(OUT),
    ]
    print("运行:", " ".join(cmd[2:]))
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print("采集失败:")
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        return 1

    meta = json.loads((SESSION / "meta.json").read_text())
    rows = list(csv.DictReader((SESSION / "states.csv").open()))
    x0, y0 = float(rows[0]["base_x"]), float(rows[0]["base_y"])
    x1, y1 = float(rows[-1]["base_x"]), float(rows[-1]["base_y"])
    dx, dy = x1 - x0, y1 - y0
    fell = bool(meta["metrics"]["fell"])

    print(f"帧数={len(rows)}  dx={dx:+.3f}m  dy={dy:+.3f}m  跌倒={fell}")
    ok = abs(dx) <= DX_MAX and abs(dy) <= DY_MAX and not fell
    if abs(dx) > DX_MAX:
        print(f"  违反: |dx|={abs(dx):.3f} > {DX_MAX}")
    if abs(dy) > DY_MAX:
        print(f"  违反: |dy|={abs(dy):.3f} > {DY_MAX}")
    if fell:
        print("  违反: 零指令下跌倒")
    print("结论:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
