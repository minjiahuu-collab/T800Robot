"""P0 环境指纹：记录复现仿真所需的版本与模型哈希。

写入每次采集的 meta.json，便于判断两条数据是否出自同一环境/模型。
"""

import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

from ..paths import REPO_ROOT, XML_PATH

_CACHE = None


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    """当前仓库 HEAD 短哈希；不在 git 仓库时返回空串。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return out.stdout.strip()


def environment_fingerprint() -> dict:
    """Python/NumPy/MuJoCo 版本、CPU 数、线程环境、模型 XML 哈希。"""
    global _CACHE
    if _CACHE is not None:
        return dict(_CACHE)
    try:
        mujoco_version = mujoco.__version__
    except AttributeError:
        mujoco_version = ""
    _CACHE = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count() or 0,
        "numpy": np.__version__,
        "mujoco": mujoco_version,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
        "model_sha256": sha256_file(XML_PATH) if Path(XML_PATH).is_file() else "",
        "git_commit": git_commit(),
    }
    return dict(_CACHE)
