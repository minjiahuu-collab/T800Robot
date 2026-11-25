"""模型加载与构建。

load_model  : 直接读 XML，等价于 MjModel.from_xml_path
build_model : 通过 MjSpec，允许在编译前插入相机/物体/参数（供 25、28 等模块使用）
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import mujoco

from .paths import XML_PATH, ensure_sdk

SpecHook = Callable[[mujoco.MjSpec], None]


def load_model(xml_path: Path = XML_PATH) -> mujoco.MjModel:
    ensure_sdk()
    return mujoco.MjModel.from_xml_path(str(xml_path))


def build_model(
    spec_hooks: Sequence[SpecHook] = (),
    xml_path: Path = XML_PATH,
) -> mujoco.MjModel:
    ensure_sdk()
    spec = mujoco.MjSpec.from_file(str(xml_path))
    for hook in spec_hooks:
        hook(spec)
    return spec.compile()
