"""B4：三路同帧拼图，供人工目检同步性。

把 head / wrist_left / wrist_right 在同一帧的图像拼成一行，取首/中/尾三帧，
输出 /tmp/frames_sync.png。每行是同一时刻，三列是三个视角。
通过标准：三路文件齐全且尺寸与 meta 一致（自动检查）；画面内容需人眼确认。
运行: python3 testB4.py [session目录]
"""

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / "recordings/walk_vx0.50_vy0.00_yaw0.00/box/seed_0000/ep_0000"
OUT = Path("/tmp/frames_sync.png")
CAMERAS = ("head", "wrist_left", "wrist_right")


def main() -> int:
    session = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    meta_path = session / "meta.json"
    if not meta_path.is_file():
        print(f"找不到 session: {session}")
        return 1

    meta = json.loads(meta_path.read_text())
    width, height = int(meta["width"]), int(meta["height"])
    total = int(meta["frames"])
    frames = sorted({0, total // 2, total - 1})

    canvas = Image.new("RGB", (width * len(CAMERAS), height * len(frames)), (0, 0, 0))
    failures = []
    for row, fr in enumerate(frames):
        for col, cam in enumerate(CAMERAS):
            path = session / "frames" / cam / f"{fr:06d}.png"
            if not path.is_file():
                failures.append(f"缺图: {path}")
                continue
            image = Image.open(path).convert("RGB")
            if image.size != (width, height):
                failures.append(
                    f"{cam}/{fr:06d}.png 尺寸 {image.size} != ({width}, {height})"
                )
                continue
            canvas.paste(image, (col * width, row * height))

    canvas.save(OUT)
    print(f"帧: {frames}（行=时刻，列={', '.join(CAMERAS)}）")
    print(f"拼图: {OUT}  ({canvas.width}x{canvas.height})")
    for item in failures:
        print("  违反:", item)
    print("自动检查: 文件齐全且尺寸一致 ->", "PASS" if not failures else "FAIL")
    print("请打开拼图目检：同一行三张应是同一瞬间；木块随机器人接近逐渐变大")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
