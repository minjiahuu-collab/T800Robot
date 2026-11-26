"""B3：图像与真值的空间一致性（多相机）。

用 cameras.csv 的相机位姿把真值木块中心投影到每一路图像；对投影落在画面内的
(帧, 相机) 组合，再按木色找像素质心，两者应重合。
通过标准：至少 2 个组合落在画面内，且每个组合色块 >= 30 像素、偏差 <= 12 像素。
说明：这套采集里木块只在左腕视角可见（头部相机俯视地面、右腕背对木块），
      因此脚本会自动跳过看不见的组合，不算失败。
运行: python3 testB3.py [session目录]
"""

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / "recordings/walk_vx0.50_vy0.00_yaw0.00/box/seed_0000/ep_0000"
CAMERAS = ("head", "wrist_left", "wrist_right")
BLOB_MIN = 30
DIST_MAX = 12.0
MIN_CASES = 2


def focal_pixels(meta: dict) -> dict:
    h = int(meta["height"])
    return {c["name"]: h / (2 * math.tan(math.radians(float(c["fovy"])) / 2))
            for c in meta["cameras"]}


def project(meta, cam_row, point, focal):
    w, h = int(meta["width"]), int(meta["height"])
    rot = np.array([[float(cam_row[f"r{i}{j}"]) for j in range(3)] for i in range(3)])
    cam_pos = np.array([float(cam_row[k]) for k in ("x", "y", "z")])
    pc = rot.T @ (np.asarray(point, dtype=float) - cam_pos)
    if pc[2] >= -1e-6:
        return None
    u = w / 2 + focal * pc[0] / -pc[2]
    v = h / 2 - focal * pc[1] / -pc[2]
    return float(u), float(v)


def wood_blob(image: Image.Image):
    a = np.asarray(image.convert("RGB"), dtype=np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mask = (r > 90) & (r > g + 20) & (g > b + 15) & (r - b > 40)
    count = int(mask.sum())
    if count == 0:
        return count, None
    ys, xs = np.nonzero(mask)
    return count, (float(xs.mean()), float(ys.mean()))


def main() -> int:
    session = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not (session / "meta.json").is_file():
        print(f"找不到 session: {session}")
        return 1

    meta = json.loads((session / "meta.json").read_text())
    focal = focal_pixels(meta)
    total = int(meta["frames"])
    frames = sorted(set(range(0, total, 10)) | {total - 1})
    gt = {int(r["frame"]): r for r in csv.DictReader((session / "ground_truth.csv").open())}
    cams = {(int(r["frame"]), r["cam"]): r
            for r in csv.DictReader((session / "cameras.csv").open())}
    w, h = int(meta["width"]), int(meta["height"])

    print(f"session: {session}")
    print("frame  cam           投影像素        色块质心        像素  偏差")
    cases, failures = 0, []
    for fr in frames:
        for cam in CAMERAS:
            if fr not in gt or (fr, cam) not in cams:
                continue
            point = [float(gt[fr][k]) for k in ("obj_x", "obj_y", "obj_z")]
            uv = project(meta, cams[(fr, cam)], point, focal[cam])
            if uv is None or not (0 <= uv[0] < w and 0 <= uv[1] < h):
                continue  # 木块不在该画面内，跳过

            cases += 1
            image_path = session / "frames" / cam / f"{fr:06d}.png"
            count, centroid = wood_blob(Image.open(image_path))
            dist = float("inf") if centroid is None else math.dist(uv, centroid)
            ok = count >= BLOB_MIN and dist <= DIST_MAX
            if not ok:
                failures.append(
                    f"帧 {fr}/{cam}: 色块={count}px 偏差={dist:.1f}px"
                )
            tip = "" if centroid else "        -        "
            if centroid:
                tip = f"({centroid[0]:6.1f},{centroid[1]:6.1f})"
            print(f"{fr:5d}  {cam:13s} ({uv[0]:6.1f},{uv[1]:6.1f})  {tip:>16}"
                  f"  {count:5d}  {dist:5.1f}")

    if cases < MIN_CASES:
        print(f"  违反: 只有 {cases} 个组合落在画面内，少于 {MIN_CASES}，无法验证")
    for item in failures:
        print("  违反:", item)
    ok = failures == [] and cases >= MIN_CASES
    print(f"有效组合: {cases}")
    print("结论:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
