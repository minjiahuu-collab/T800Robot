"""随机化采样器：从配置派生可复现的参数样本。

核心设计：
  1. 种子不是"递增计数器"，而是由 (base_seed, session, episode) 三元组派生。
     好处：并行跑 8 个 worker、或只重跑第 7 条 episode，得到的样本和串行完全一致。
  2. 字符串哈希用 zlib.crc32，不用内置 hash()。内置 hash 每个进程加盐，
     换个进程就变了，会让"同配置复现"这个前提直接失效。
  3. 采样顺序固定按 yaml 书写顺序，不使用集合/字典无序遍历。
  4. 每个样本附带 meta（配置哈希、seed、session、episode），写进 session 后可审计：
     拿到任何一条数据，都能反推它当时用了什么物理参数。
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from .schema import ParamSpec, Plan, SchemaError, load_plan


def _stable_hash(text: str) -> int:
    """跨进程稳定的字符串哈希。"""
    return zlib.crc32(text.encode("utf-8"))


def derive_rng(plan: Plan, session: str, episode: int) -> np.random.Generator:
    """按 (seed, session, episode) 派生独立随机流。

    per_session: 同一 session 的所有 episode 共享样本（轨迹回放需要固定物理参数）
    fixed      : 全局只取一次
    """
    if plan.mode == "fixed":
        episode_key = 0
    elif plan.mode == "per_session":
        episode_key = 0
    else:
        episode_key = int(episode)
    entropy = [int(plan.seed), _stable_hash(str(session)), episode_key]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _to_python(value: Any) -> Any:
    """把 numpy 标量/数组转成可直接 json 序列化的 Python 对象。"""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    return arr.tolist()


def _base_shape(param: ParamSpec) -> tuple:
    """参数在配置里声明的基础形状；per_body 在它前面加一维。"""
    spec = param.spec
    if param.type in ("uniform", "loguniform"):
        return np.broadcast_shapes(
            np.asarray(spec["low"], dtype=float).shape,
            np.asarray(spec["high"], dtype=float).shape,
        )
    if param.type == "normal":
        return np.broadcast_shapes(
            np.asarray(spec["mean"], dtype=float).shape,
            np.asarray(spec["std"], dtype=float).shape,
        )
    return ()


def _fit(value: np.ndarray, shape: tuple, where: str) -> np.ndarray:
    try:
        return np.broadcast_to(value, shape)
    except ValueError:
        raise SchemaError(
            f"{where}: 不能广播到形状 {shape}，实际是 {value.shape}"
        ) from None


def _draw(rng: np.random.Generator, param: ParamSpec, shape: tuple | None = None) -> Any:
    """按分布类型抽一个样本。shape 用于逐 body 等按对象展开的采样。"""
    spec = param.spec
    target = _base_shape(param) if shape is None else tuple(shape)

    if param.type == "fixed":
        if shape is not None:
            raise SchemaError(f"{param.name}: fixed 不支持形状展开")
        return spec["value"]

    if param.type == "choice":
        if shape is not None:
            raise SchemaError(f"{param.name}: choice 不支持形状展开")
        values = list(spec["values"])
        return values[int(rng.integers(len(values)))]

    if param.type == "uniform":
        lo = _fit(np.asarray(spec["low"], dtype=float), target, f"{param.name}.low")
        hi = _fit(np.asarray(spec["high"], dtype=float), target, f"{param.name}.high")
        return _to_python(rng.uniform(lo, hi))

    if param.type == "loguniform":
        # 在 log 空间均匀采样：对跨数量级的正值更合理，
        # 直接在原空间均匀采样会让样本大量堆在高值一侧。
        lo = _fit(np.log(np.asarray(spec["low"], dtype=float)), target, f"{param.name}.low")
        hi = _fit(np.log(np.asarray(spec["high"], dtype=float)), target, f"{param.name}.high")
        return _to_python(np.exp(rng.uniform(lo, hi)))

    if param.type == "normal":
        mean = _fit(np.asarray(spec["mean"], dtype=float), target, f"{param.name}.mean")
        std = _fit(np.asarray(spec["std"], dtype=float), target, f"{param.name}.std")
        value = rng.normal(mean, std)
        if "clip" in spec:
            clip = spec["clip"]
            if isinstance(clip, (int, float)):
                lo_v, hi_v = -abs(float(clip)), abs(float(clip))
            else:
                lo_v, hi_v = clip[0], clip[1]
            value = np.clip(
                value,
                _fit(np.asarray(lo_v, dtype=float), target, f"{param.name}.clip"),
                _fit(np.asarray(hi_v, dtype=float), target, f"{param.name}.clip"),
            )
        return _to_python(value)

    raise SchemaError(f"{param.name}: 未知分布类型 {param.type!r}")


def per_body_shapes(plan: Plan, count: int) -> Dict[str, tuple]:
    """为 per_body 参数生成 (对象数量, 基础形状) 的采样形状提示。"""
    count = int(count)
    if count <= 0:
        raise ValueError(f"count 必须为正，实际是 {count}")
    shapes: Dict[str, tuple] = {}
    for group, param in plan.enabled_params():
        if param.spec.get("per_body"):
            shapes[f"{group}.{param.name}"] = (count,) + _base_shape(param)
    return shapes


def sample(
    plan: Plan,
    session: str = "default",
    episode: int = 0,
    shapes: Dict[str, tuple] | None = None,
) -> Dict[str, Any]:
    """抽取一个样本，返回 {"meta": ..., "params": {group: {name: value}}}。

    shapes 是 {group.name: shape} 的形状提示，用于 per_body 参数按 body 数量展开。
    """
    rng = derive_rng(plan, session, episode)
    params: Dict[str, Dict[str, Any]] = {}
    for group_name, param in plan.enabled_params():
        key = f"{group_name}.{param.name}"
        hint = shapes.get(key) if shapes else None
        params.setdefault(group_name, {})[param.name] = _draw(rng, param, hint)
    return {
        "meta": {
            "plan_path": str(plan.path),
            "plan_hash": plan.source_hash,
            "plan_version": plan.version,
            "mode": plan.mode,
            "seed": plan.seed,
            "session": str(session),
            "episode": int(episode),
        },
        "params": params,
    }


def sample_many(
    plan: Plan,
    sessions: Sequence[str],
    episodes: int,
    shapes: Dict[str, tuple] | None = None,
) -> List[Dict[str, Any]]:
    """批量采样，供 CLI 预览与回归测试使用。"""
    out = []
    for session in sessions:
        for episode in range(max(0, int(episodes))):
            out.append(sample(plan, session, episode, shapes=shapes))
    return out


# ── CLI ─────────────────────────────────────────────────────────────────────
def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    return str(value)


def _flatten(params: Dict[str, Dict[str, Any]]) -> str:
    parts = [
        f"{group}.{name}={_fmt(value)}"
        for group, items in params.items()
        for name, value in items.items()
    ]
    return "  ".join(parts)


def main(
    config: str | Path | None = None,
    sessions: Sequence[str] = ("default",),
    episodes: int = 3,
    out: str | None = None,
    bodies: int | None = None,
) -> int:
    plan = load_plan(config)
    shapes = per_body_shapes(plan, bodies) if bodies is not None else None
    samples = sample_many(plan, sessions, episodes, shapes=shapes)

    groups = [
        f"{g.name}({sum(1 for p in g.params.values() if p.enabled)})"
        for g in plan.groups.values() if g.enabled
    ]
    print(f"配置: {plan.path}")
    print(f"  hash={plan.source_hash}  mode={plan.mode}  seed={plan.seed}")
    print(f"  启用维度: {' '.join(groups) or '无'}   共 {plan.count_enabled()} 个参数")
    per_body_params = [
        f"{group}.{param.name}"
        for group, param in plan.enabled_params()
        if param.spec.get("per_body")
    ]
    if bodies is not None:
        print(f"  逐 body 参数按 {int(bodies)} 个对象展开")
    elif per_body_params:
        print(f"  提示: {' '.join(per_body_params)} 需加 --bodies N 才会逐 body 展开")

    if not samples:
        print("  (episodes=0，只做校验)")
        return 0

    for item in samples:
        meta = item["meta"]
        print(f"[{meta['session']} ep={meta['episode']}] {_flatten(item['params'])}")

    if out:
        payload = {
            "plan": {
                "path": str(plan.path),
                "hash": plan.source_hash,
                "version": plan.version,
                "mode": plan.mode,
                "seed": plan.seed,
            },
            "samples": samples,
        }
        Path(out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"已写出: {out}")
    return 0
