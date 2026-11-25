"""仓库内资源路径。全部相对仓库根解析，不依赖当前工作目录。"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SDK_DIR = REPO_ROOT / "engineai_robotics_native_sdk"
ASSETS_DIR = SDK_DIR / "assets"
CONFIG_ROOT = ASSETS_DIR / "config/t800"
DEFAULT_CFG = CONFIG_ROOT / "rl_walking_example/default.yaml"
XML_PATH = ASSETS_DIR / "resource/t800.xml"


def ensure_sdk() -> None:
    """SDK 以 git submodule 引入，缺失时给出可执行的修复命令。"""
    if not XML_PATH.is_file():
        raise FileNotFoundError(
            f"找不到 T800 模型: {XML_PATH}\n"
            "SDK 是 git submodule，请先执行: git submodule update --init --recursive"
        )
