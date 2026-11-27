"""域随机化配置的 schema：解析、校验、冻结。

四条设计约束：
  1. 类型显式 —— 每个参数必须写 type，不允许"给个范围就默认均匀"。
  2. 未知字段报错 —— 拼写错误、写错单位必须立即失败，而不是被静默忽略。
  3. 作用域显式 —— 摩擦/质量这类参数必须指明作用集合，禁止全局一刀切。
  4. 可审计 —— 解析时记录配置文件的 sha256，随样本一起写入 session meta。

解析结果是不可变对象，采样器只读不写，避免"采样过程中被改配置"的隐性 bug。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import numpy as np
import yaml

from ..paths import RANDOMIZATION_CFG


class SchemaError(ValueError):
    """随机化配置不合法。"""


_TYPES = ("uniform", "loguniform", "normal", "choice", "fixed")

# 每种分布允许出现的字段，其余一律视为拼写错误
_ALLOWED_KEYS = {
    "uniform": {"type", "low", "high", "enabled", "scope", "desc", "per_body"},
    "loguniform": {"type", "low", "high", "enabled", "scope", "desc", "per_body"},
    "normal": {"type", "mean", "std", "clip", "enabled", "scope", "desc", "per_body"},
    "choice": {"type", "values", "enabled", "scope", "desc"},
    "fixed": {"type", "value", "enabled", "scope", "desc"},
}

_TOP_KEYS = {"version", "seed", "mode", "groups", "desc"}
_GROUP_KEYS = {"desc", "enabled", "params"}
_MODES = ("per_episode", "per_session", "fixed")

# 允许出现的作用域。写错必须立即失败，不能让 apply 层按固定语义继续跑。
_SCOPES = ("global", "robot_bodies", "robot_geoms", "ground", "solver", "joints")

# 这些参数的作用域属于物理语义，必须显式写对。
_REQUIRED_SCOPES = {
    "mass_scale": "robot_bodies",
    "com_offset_m": "robot_bodies",
    "ground_friction": "ground",
    "robot_friction_scale": "robot_geoms",
    "solref_timeconst_scale": "solver",
    "joint_damping_scale": "joints",
}

# per_body 会把参数按对象数量展开：质心偏移是向量，质量是逐 body 标量。
_PER_BODY_PARAMS = {"com_offset_m", "mass_scale"}


@dataclass(frozen=True)
class ParamSpec:
    """单个可随机化参数。spec 保存原始字段（已校验），供采样器解释。"""

    name: str
    type: str
    spec: Dict[str, Any]
    scope: str
    enabled: bool
    desc: str


@dataclass(frozen=True)
class GroupSpec:
    name: str
    params: Dict[str, ParamSpec]
    enabled: bool
    desc: str


@dataclass(frozen=True)
class Plan:
    version: int
    seed: int
    mode: str
    groups: Dict[str, GroupSpec]
    source_hash: str
    path: Path

    def enabled_params(self) -> Iterator[Tuple[str, ParamSpec]]:
        """按 yaml 中的书写顺序产出启用项，保证采样顺序确定。"""
        for group in self.groups.values():
            if not group.enabled:
                continue
            for param in group.params.values():
                if param.enabled:
                    yield group.name, param

    def count_enabled(self) -> int:
        return sum(1 for _ in self.enabled_params())


# ── 数值校验 ────────────────────────────────────────────────────────────────
def _as_array(value: Any, where: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise SchemaError(f"{where}: 需要数值或数值列表，实际是 {value!r}") from None
    if arr.size == 0:
        raise SchemaError(f"{where}: 不能为空")
    if not np.all(np.isfinite(arr)):
        raise SchemaError(f"{where}: 必须都是有限值，实际是 {value!r}")
    return arr


def _check_broadcastable(a: np.ndarray, b: np.ndarray, where: str) -> None:
    try:
        np.broadcast_shapes(a.shape, b.shape)
    except ValueError:
        raise SchemaError(
            f"{where}: 两个数组形状不兼容 {a.shape} 与 {b.shape}"
        ) from None


def _parse_param(group: str, name: str, raw: Any) -> ParamSpec:
    where = f"{group}.{name}"
    if not isinstance(raw, dict):
        raise SchemaError(f"{where}: 参数定义必须是映射，实际是 {type(raw).__name__}")

    ptype = raw.get("type")
    if ptype not in _TYPES:
        raise SchemaError(
            f"{where}: type 必须是 {_TYPES} 之一，实际是 {ptype!r}"
        )

    unknown = set(raw) - _ALLOWED_KEYS[ptype]
    if unknown:
        raise SchemaError(
            f"{where}: {ptype} 不认识的字段 {sorted(unknown)}；"
            f"允许的字段是 {sorted(_ALLOWED_KEYS[ptype])}"
        )

    scope = str(raw.get("scope", "global"))
    if scope not in _SCOPES:
        raise SchemaError(f"{where}: scope 必须是 {_SCOPES} 之一，实际是 {scope!r}")

    required_scope = _REQUIRED_SCOPES.get(name)
    if required_scope is not None:
        if "scope" not in raw:
            raise SchemaError(f"{where}: 该参数必须显式声明 scope: {required_scope}")
        if scope != required_scope:
            raise SchemaError(
                f"{where}: scope 必须是 {required_scope!r}，实际是 {scope!r}"
            )

    per_body = raw.get("per_body", False)
    if not isinstance(per_body, bool):
        raise SchemaError(f"{where}.per_body: 必须是布尔值，实际是 {per_body!r}")
    if per_body and name not in _PER_BODY_PARAMS:
        raise SchemaError(f"{where}: 该参数不支持 per_body")
    if per_body and scope != "robot_bodies":
        raise SchemaError(f"{where}: per_body 只能用于 scope: robot_bodies")
    if per_body and ptype not in ("uniform", "loguniform", "normal"):
        raise SchemaError(f"{where}: per_body 只支持连续分布，实际是 {ptype}")

    if ptype in ("uniform", "loguniform"):
        if "low" not in raw or "high" not in raw:
            raise SchemaError(f"{where}: {ptype} 必须同时提供 low 与 high")
        low = _as_array(raw["low"], f"{where}.low")
        high = _as_array(raw["high"], f"{where}.high")
        _check_broadcastable(low, high, where)
        if np.any(high <= low):
            raise SchemaError(f"{where}: 要求 high > low，实际 low={raw['low']} high={raw['high']}")
        if ptype == "loguniform" and np.any(low <= 0):
            raise SchemaError(f"{where}: loguniform 要求 low > 0，实际 low={raw['low']}")

    elif ptype == "normal":
        if "mean" not in raw or "std" not in raw:
            raise SchemaError(f"{where}: normal 必须同时提供 mean 与 std")
        mean = _as_array(raw["mean"], f"{where}.mean")
        std = _as_array(raw["std"], f"{where}.std")
        _check_broadcastable(mean, std, where)
        if np.any(std < 0):
            raise SchemaError(f"{where}: std 必须非负，实际 std={raw['std']}")
        if "clip" in raw:
            clip = raw["clip"]
            if isinstance(clip, (list, tuple)) and len(clip) != 2:
                raise SchemaError(f"{where}.clip: 必须是 [下界, 上界] 或单个标量")

    elif ptype == "choice":
        values = raw.get("values")
        if not isinstance(values, (list, tuple)) or not values:
            raise SchemaError(f"{where}: choice 的 values 必须是非空列表")

    elif ptype == "fixed":
        if "value" not in raw:
            raise SchemaError(f"{where}: fixed 必须提供 value")

    return ParamSpec(
        name=name,
        type=ptype,
        spec=dict(raw),
        scope=scope,
        enabled=bool(raw.get("enabled", True)),
        desc=str(raw.get("desc", "")),
    )


def _parse_group(name: str, raw: Any) -> GroupSpec:
    if not isinstance(raw, dict):
        raise SchemaError(f"group {name}: 必须是映射")
    unknown = set(raw) - _GROUP_KEYS
    if unknown:
        raise SchemaError(
            f"group {name}: 不认识的字段 {sorted(unknown)}；"
            f"允许的字段是 {sorted(_GROUP_KEYS)}（参数要写在 params 下）"
        )
    params_raw = raw.get("params", {})
    if not isinstance(params_raw, dict):
        raise SchemaError(f"group {name}.params: 必须是映射")
    bad = [key for key in params_raw if not isinstance(key, str)]
    if bad:
        raise SchemaError(
            f"group {name}: 参数名必须是字符串，实际有 {bad!r}；"
            "YAML 会把 on/off/yes/no 解析成布尔值，请改名"
        )
    params = {
        key: _parse_param(name, key, value)
        for key, value in params_raw.items()
    }
    return GroupSpec(
        name=name,
        params=params,
        enabled=bool(raw.get("enabled", True)),
        desc=str(raw.get("desc", "")),
    )


def load_plan(path: Path | str | None = None) -> Plan:
    """读取并校验配置，返回不可变 Plan。"""
    target = Path(path) if path is not None else RANDOMIZATION_CFG
    if not target.is_file():
        raise FileNotFoundError(f"找不到随机化配置: {target}")

    raw_bytes = target.read_bytes()
    try:
        cfg = yaml.safe_load(raw_bytes.decode("utf-8"))
    except yaml.YAMLError as exc:
        raise SchemaError(f"{target}: YAML 解析失败: {exc}") from None
    if not isinstance(cfg, dict):
        raise SchemaError(f"{target}: 顶层必须是映射")

    unknown = set(cfg) - _TOP_KEYS
    if unknown:
        raise SchemaError(
            f"顶层不认识的字段 {sorted(unknown)}；允许的字段是 {sorted(_TOP_KEYS)}"
        )

    version = int(cfg.get("version", 1))
    if version != 1:
        raise SchemaError(f"不支持的配置版本: {version}")

    seed = int(cfg.get("seed", 0))
    mode = str(cfg.get("mode", "per_episode"))
    if mode not in _MODES:
        raise SchemaError(f"mode 必须是 {_MODES} 之一，实际是 {mode!r}")

    groups_raw = cfg.get("groups")
    if not isinstance(groups_raw, dict) or not groups_raw:
        raise SchemaError("groups 必须是非空映射")
    bad = [name for name in groups_raw if not isinstance(name, str)]
    if bad:
        raise SchemaError(
            f"维度名必须是字符串，实际有 {bad!r}；"
            "YAML 会把 on/off/yes/no 解析成布尔值，请改名"
        )

    groups = {name: _parse_group(name, value) for name, value in groups_raw.items()}

    return Plan(
        version=version,
        seed=seed,
        mode=mode,
        groups=groups,
        source_hash=hashlib.sha256(raw_bytes).hexdigest()[:16],
        path=target,
    )
