"""数据资产校验：检查一次采集 session 的完整性与自洽性。

校验项:
  - meta.json 存在、可解析、必需字段齐全
  - states.csv 表头齐全、行数等于 meta.frames、frame 连续 0..N-1
  - cameras.csv 行数 = 帧数 × 相机数
  - ground_truth.csv 行数 = 帧数
  - 每台相机帧目录文件名连续 000000..N-1，首帧尺寸与 meta 一致

用法:
    python3 -m t800_sim.cli validate --session recordings
    python3 -m t800_sim.cli validate --session recordings/walk/box/seed_0000/ep_0000
"""

import csv
import json
import sys
from pathlib import Path

from PIL import Image

REQUIRED_META = (
    "schema_version", "task", "scene_id", "seed", "policy", "git_commit",
    "width", "height", "fps", "timestep", "cameras", "joints", "frames",
    "duration", "success", "metrics",
)
REQUIRED_STATES = (
    "frame", "time", "render_time", "vx_cmd", "vy_cmd", "yaw_cmd",
    "base_x", "base_y", "base_z", "qw", "qx", "qy", "qz",
)


def _read_csv(path: Path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def find_sessions(root) -> list:
    """root 本身是 session 则返回自身，否则递归找所有含 meta.json 的目录。"""
    root = Path(root)
    if (root / "meta.json").is_file():
        return [root]
    if not root.exists():
        return []
    return sorted(p.parent for p in root.rglob("meta.json"))


def validate_session(path) -> dict:
    """校验一个 session 目录，返回 {'session', 'ok', 'errors', ...}。"""
    path = Path(path)
    report = {
        "session": str(path),
        "ok": False,
        "frames": 0,
        "cameras": 0,
        "errors": [],
        "warnings": [],
    }
    errors = report["errors"]

    meta_path = path / "meta.json"
    if not meta_path.is_file():
        errors.append("缺少 meta.json")
        return report
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError as exc:
        errors.append(f"meta.json 解析失败: {exc}")
        return report

    for key in REQUIRED_META:
        if key not in meta:
            errors.append(f"meta 缺少字段: {key}")
    if meta.get("schema_version") != "1.0":
        errors.append(f"schema_version 期望 1.0，实际 {meta.get('schema_version')!r}")
    if "environment" not in meta:
        report["warnings"].append("meta 缺少 environment 环境指纹（P0 之前的旧数据）")

    n = meta.get("frames")
    if not isinstance(n, int) or n < 0:
        errors.append(f"meta.frames 非法: {n!r}")
        n = None

    cam_list = meta.get("cameras") or []
    ncam = len(cam_list)
    report["frames"] = n or 0
    report["cameras"] = ncam
    if ncam == 0:
        errors.append("meta.cameras 为空")

    states_path = path / "states.csv"
    if not states_path.is_file():
        errors.append("缺少 states.csv")
    else:
        rows, fields = _read_csv(states_path)
        missing = [c for c in REQUIRED_STATES if c not in fields]
        if missing:
            errors.append(f"states.csv 缺少列: {', '.join(missing)}")
        if n is not None:
            if len(rows) != n:
                errors.append(f"states.csv 行数 {len(rows)} != frames {n}")
            frames = []
            for row in rows:
                try:
                    frames.append(int(row["frame"]))
                except (KeyError, TypeError, ValueError):
                    frames.append(-1)
            if frames != list(range(len(rows))):
                errors.append("states.csv frame 列不是连续 0..N-1")

    if n is not None and n > 0:
        gt_path = path / "ground_truth.csv"
        if not gt_path.is_file():
            errors.append("缺少 ground_truth.csv")
        else:
            gt_rows, gt_fields = _read_csv(gt_path)
            if len(gt_rows) != n:
                errors.append(f"ground_truth.csv 行数 {len(gt_rows)} != frames {n}")
            for col in ("obj_x", "obj_y", "obj_z", "ee_l_x", "ee_r_x"):
                if col not in gt_fields:
                    errors.append(f"ground_truth.csv 缺少列: {col}")
                    break

        cam_path = path / "cameras.csv"
        if not cam_path.is_file():
            errors.append("缺少 cameras.csv")
        else:
            cam_rows, _ = _read_csv(cam_path)
            expect = n * ncam
            if len(cam_rows) != expect:
                errors.append(
                    f"cameras.csv 行数 {len(cam_rows)} != frames×cameras {expect}"
                )

    width = meta.get("width")
    height = meta.get("height")
    for cam in cam_list:
        name = cam.get("name")
        if not name:
            errors.append("meta.cameras 存在无名相机")
            continue
        cam_dir = path / "frames" / name
        if not cam_dir.is_dir():
            errors.append(f"缺少相机帧目录: frames/{name}")
            continue
        if n is None:
            continue
        expected = [f"{i:06d}.png" for i in range(n)]
        actual = sorted(p.name for p in cam_dir.glob("*.png"))
        if actual != expected:
            errors.append(
                f"frames/{name} 帧文件不连续: 期望 {n} 张，实际 {len(actual)} 张"
            )
        if n > 0 and (width, height) and isinstance(width, int) and isinstance(height, int):
            try:
                with Image.open(cam_dir / expected[0]) as img:
                    if img.size != (width, height):
                        errors.append(
                            f"frames/{name}/000000.png 尺寸 {img.size} "
                            f"!= ({width}, {height})"
                        )
            except OSError as exc:
                errors.append(f"frames/{name}/000000.png 无法读取: {exc}")

    metrics = meta.get("metrics") or {}
    for key in ("mean_vx", "final_dx", "fell", "success"):
        if key not in metrics:
            errors.append(f"meta.metrics 缺少字段: {key}")

    report["ok"] = not errors
    return report


def main(session) -> int:
    """校验单个 session 或目录下所有 session，返回进程退出码。"""
    sessions = find_sessions(session)
    if not sessions:
        print(f"未找到 session（目录下没有 meta.json）: {session}")
        return 1

    all_ok = True
    reports = []
    for s in sessions:
        rep = validate_session(s)
        reports.append(rep)
        all_ok = all_ok and rep["ok"]
        status = "PASS" if rep["ok"] else "FAIL"
        print(f"[{status}] {s}  frames={rep['frames']} cameras={rep['cameras']}")
        for exc in rep["errors"]:
            print(f"    - {exc}")
        for warn in rep["warnings"][:2]:
            print(f"    ~ {warn}")
    ok = sum(rep["ok"] for rep in reports)
    print(f"\n{ok}/{len(reports)} 个 session 通过校验")
    return 0 if all_ok else 1


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "recordings"
    raise SystemExit(main(target))
