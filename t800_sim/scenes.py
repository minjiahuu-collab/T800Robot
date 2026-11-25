"""运行时往 MjSpec 里加物体（不改 SDK 文件）。"""

import mujoco

WOOD_RGBA = [0.72, 0.45, 0.20, 1.0]


def add_box(
    spec: mujoco.MjSpec,
    name: str,
    size: float,
    *,
    pos=(0.0, 0.0, 0.0),
    parent=None,
    free: bool = False,
    density: float = 300.0,
    friction=None,
    rgba=None,
    collide: bool = True,
):
    """加一个边长为 size 的正方体。

    parent 为空时挂在 world 下；free=True 时带 freejoint（会独立运动）。
    collide=False 时关闭碰撞，只做视觉跟随。
    """
    parent_body = spec.body(parent) if parent is not None else spec.worldbody
    body = parent_body.add_body()
    body.name = name
    body.pos = list(pos)
    if free:
        body.add_freejoint()

    geom = body.add_geom()
    geom.name = f"{name}_geom"
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size = [size / 2] * 3
    geom.rgba = list(rgba if rgba is not None else WOOD_RGBA)
    geom.density = density
    if friction is not None:
        geom.friction = list(friction)
    if not collide:
        geom.contype = 0
        geom.conaffinity = 0

    return body
