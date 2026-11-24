"""T800 骨架可视化：用 matplotlib 按世界坐标绘制连杆骨架（3D 视角）。

用法:
    python3 09_t800_skeleton.py                 # 弹窗显示
    MPLBACKEND=Agg python3 09_t800_skeleton.py  # 无显示环境（不弹窗）
"""

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.ticker import MaxNLocator

XML_PATH = (
    Path(__file__).resolve().parent
    / "engineai_robotics_native_sdk/assets/resource/t800.xml"
)

# 中文字体候选：优先用 Windows 自带字体，其次用 Linux 已装的 Noto/文泉驿
FONT_CANDIDATES = [
    "/mnt/c/Windows/Fonts/simhei.ttf",
    "/mnt/c/Windows/Fonts/msyh.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]

# 只标注主要连杆，避免文字过密
LABELS = {
    "LINK_BASE", "LINK_KNEE_PITCH_L", "LINK_KNEE_PITCH_R",
    "LINK_FOOT_L", "LINK_FOOT_R", "LINK_WRIST_END_L", "LINK_WRIST_END_R",
    "LINK_HEAD_YAW", "LINK_SHOULDER_PITCH_L", "LINK_SHOULDER_PITCH_R",
}


def setup_chinese_font():
    """注册可用的中文字体，避免中文显示成方框。"""
    for path in FONT_CANDIDATES:
        if not Path(path).exists():
            continue
        try:
            fm.fontManager.addfont(path)
            name = fm.FontProperties(fname=path).get_name()
            plt.rcParams["font.sans-serif"] = [name] + plt.rcParams["font.sans-serif"]
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号
            print(f"使用中文字体: {name} ({path})")
            return
        except Exception:
            continue
    print("警告: 未找到可用中文字体，图中中文可能显示为方框")


def draw_skeleton(ax, xpos, parent, names):
    # 骨架连线：每个 body 与其父连杆之间画一段（世界坐标 x/y/z）
    for i in range(1, len(xpos)):
        p = parent[i]
        ax.plot(
            [xpos[p][0], xpos[i][0]],
            [xpos[p][1], xpos[i][1]],
            [xpos[p][2], xpos[i][2]],
            "-o", color="tab:blue", markersize=3, linewidth=1.5,
        )

    # 标注主要连杆名称（z 方向抬高一点，减少与骨架线的重叠）
    for i, txt in enumerate(names):
        if txt not in LABELS:
            continue
        ax.text(
            xpos[i][0], xpos[i][1], xpos[i][2] + 0.02, txt.replace("LINK_", ""),
            fontsize=7, color="dimgray",
        )


def set_world_frame(ax, xpos):
    """按世界坐标系等比例显示：水平面保留足够宽度，高度按实际比例拉伸。"""
    lo, hi = xpos.min(axis=0), xpos.max(axis=0)
    sizes = hi - lo
    center = (lo + hi) / 2

    xy_span = max(float(sizes[0]), float(sizes[1]), 0.6)  # 水平面最小显示宽度
    z_pad = 0.06

    ax.set_xlim(center[0] - xy_span / 2, center[0] + xy_span / 2)
    ax.set_ylim(center[1] - xy_span / 2, center[1] + xy_span / 2)
    ax.set_zlim(lo[2] - z_pad, hi[2] + z_pad)
    # 盒体比例 = 各轴实际范围 => 三轴单位长度一致，骨架不拉伸
    ax.set_box_aspect((xy_span, xy_span, sizes[2] + 2 * z_pad))

    # 轴范围较窄时减少刻度数量，避免底部刻度文字挤在一起
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=6))

    ax.set_xlabel("世界 X / m", labelpad=8)
    ax.set_ylabel("世界 Y / m", labelpad=12)
    ax.set_zlabel("世界 Z / m", labelpad=10)
    # 调整视角：斜前上方，可同时看清前后、左右和高度
    ax.view_init(elev=12, azim=-70)


def main():
    setup_chinese_font()

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # 计算世界坐标 data.xpos

    xpos = np.array(data.xpos)
    parent = model.body_parentid.copy()
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "(unnamed)"
        for i in range(model.nbody)
    ]

    fig = plt.figure(figsize=(8, 9))
    ax = fig.add_subplot(projection="3d")
    draw_skeleton(ax, xpos, parent, names)
    set_world_frame(ax, xpos)

    ax.set_title("T800 机器人骨架结构（世界坐标系）", fontsize=13)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
