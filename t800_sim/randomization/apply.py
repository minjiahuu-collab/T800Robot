"""把随机化样本应用到 MuJoCo 模型与运行时。

分工：
  apply_model    : 修改已编译的 MjModel（质量、质心、摩擦、接触参数、阻尼、力矩上限）
  runtime_params : 汇总控制器侧参数（增益缩放、控制周期、指令延迟、传感器噪声）
  SensorModel    : 按 runtime_params 扰动观测信号，供策略控制回路使用

三条约束：
  1. 作用域显式 —— 只改名字前缀为 LINK_/Motor_ 的机器人 body、world 的平面地面，
     不把摩擦/质量一刀切到场景物体上。
  2. 未启用即默认值 —— 样本里没有的参数不碰模型。
  3. 可审计 —— 返回 AppliedSample，记录改了哪些参数、影响了多少 body/geom/joint。

为什么在编译后改 MjModel，而不是全部走 MjSpec：
  MuJoCo 4.0.1 的 MjSpec 把 joint damping 写成了三元向量，编译时被忽略；
  而 body 质量/惯量在编译后修改又必须刷新 invweight 等派生量。
  统一在编译后写数组并调用 mj_setConst，行为最可控。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterator, Mapping, Optional, Tuple

import mujoco
import numpy as np

from .sampler import per_body_shapes

ROBOT_BODY_PREFIXES = ("LINK_", "Motor_")
GRAVITY = 9.81

# serial_t800.xml 里 <default class="joint"> 声明了 damping=0.6，但关节没有挂
# 这个 class，编译后 dof_damping 全是 0，缩放 0 仍然是 0。随机化需要一个非零
# 基线才能让 joint_damping_scale 有意义。
JOINT_DAMPING_BASELINE = 0.6

# 模型层参数
MODEL_PARAMS = frozenset(
    {
        ("dynamics", "mass_scale"),
        ("dynamics", "com_offset_m"),
        ("contact", "ground_friction"),
        ("contact", "robot_friction_scale"),
        ("contact", "solref_timeconst_scale"),
        ("actuator", "joint_damping_scale"),
        ("actuator", "torque_limit_scale"),
        ("camera", "fovy_scale"),
        ("camera", "pos_noise_m"),
        ("rendering", "light_intensity_scale"),
        ("rendering", "texture_swap_prob"),
    }
)

# 运行时参数
RUNTIME_PARAMS = frozenset(
    {
        ("actuator", "kp_scale"),
        ("actuator", "kd_scale"),
        ("actuator", "motor_delay_ms"),
        ("actuator", "control_dt_ms"),
        ("sensors", "imu_bias_gyro_rads"),
        ("sensors", "imu_bias_accel_mps2"),
        ("sensors", "imu_random_walk_gyro"),
        ("sensors", "encoder_bits"),
        ("sensors", "force_noise_std_n"),
        ("sensors", "frame_dropout_prob"),
        ("sensors", "latency_ms"),
    }
)

_MODEL_KEYS = frozenset(f"{group}.{name}" for group, name in MODEL_PARAMS)
_RUNTIME_KEYS = frozenset(f"{group}.{name}" for group, name in RUNTIME_PARAMS)
_KNOWN_PARAMS = _MODEL_KEYS | _RUNTIME_KEYS


class ApplyError(ValueError):
    """随机化样本无法应用到当前模型。"""


# ── 样本解析 ────────────────────────────────────────────────────────────────
def _iter_params(params: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    if not isinstance(params, Mapping):
        raise ApplyError(f"params 必须是映射，实际是 {type(params).__name__}")
    for group, items in params.items():
        if not isinstance(items, Mapping):
            raise ApplyError(f"{group}: 参数集合必须是映射，实际是 {type(items).__name__}")
        for name, value in items.items():
            yield f"{group}.{name}", value


def _check_supported(params: Mapping[str, Any]) -> Dict[str, Any]:
    """展平成 {group.name: value}，同时拒绝应用层还不认识的参数。"""
    flat = dict(_iter_params(params))
    unknown = sorted(key for key in flat if key not in _KNOWN_PARAMS)
    if unknown:
        raise ApplyError(
            f"应用层还不支持这些参数: {unknown}；"
            "请在配置里关闭对应 group，或先实现对应的应用逻辑"
        )
    return flat


def _scalar(value: Any, where: str, *, positive: bool = False,
            nonneg: bool = False) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ApplyError(f"{where}: 需要标量，实际是 {value!r}") from None
    if not math.isfinite(out):
        raise ApplyError(f"{where}: 必须是有限值，实际是 {value!r}")
    if positive and out <= 0:
        raise ApplyError(f"{where}: 必须大于 0，实际是 {out}")
    if nonneg and out < 0:
        raise ApplyError(f"{where}: 必须非负，实际是 {out}")
    return out


def _vec(value: Any, where: str, size: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (size,):
        raise ApplyError(f"{where}: 需要长度 {size} 的向量，实际形状 {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ApplyError(f"{where}: 必须都是有限值，实际是 {value!r}")
    return arr


def _integer(value: Any, where: str, *, minimum: int = 0) -> int:
    number = _scalar(value, where)
    if number != int(number):
        raise ApplyError(f"{where}: 需要整数，实际是 {value!r}")
    out = int(number)
    if out < minimum:
        raise ApplyError(f"{where}: 必须 >= {minimum}，实际是 {out}")
    return out


# ── 模型索引 ────────────────────────────────────────────────────────────────
def _robot_body_ids(model: mujoco.MjModel) -> np.ndarray:
    ids = [
        bid for bid in range(1, model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or "").startswith(
            ROBOT_BODY_PREFIXES
        )
    ]
    if not ids:
        raise ApplyError(
            f"模型里找不到 {ROBOT_BODY_PREFIXES} 前缀的机器人 body，无法限定作用域"
        )
    return np.asarray(ids, dtype=int)


def _robot_geoms(model: mujoco.MjModel, body_ids: np.ndarray) -> np.ndarray:
    mask = np.isin(model.geom_bodyid, body_ids)
    ids = np.flatnonzero(mask)
    if ids.size == 0:
        raise ApplyError("机器人的 body 上找不到 geom，无法应用接触参数")
    return ids


def _ground_geoms(model: mujoco.MjModel) -> np.ndarray:
    ids = np.flatnonzero(
        (model.geom_bodyid == 0)
        & (model.geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE))
    )
    if ids.size == 0:
        raise ApplyError("模型里找不到 world 下的平面地面 geom，无法应用 ground_friction")
    return ids


def _robot_actuators(model: mujoco.MjModel, body_ids: np.ndarray) -> np.ndarray:
    robot_bodies = set(int(bid) for bid in body_ids)
    ids = []
    for uid in range(model.nu):
        jid = int(model.actuator_trnid[uid, 0])
        if jid >= 0 and int(model.jnt_bodyid[jid]) in robot_bodies:
            ids.append(uid)
    if not ids:
        raise ApplyError("模型里找不到驱动机器人关节的 actuator，无法应用 torque_limit_scale")
    return np.asarray(ids, dtype=int)


def _robot_joint_dofs(model: mujoco.MjModel, body_ids: np.ndarray) -> np.ndarray:
    dofs = []
    robot_bodies = set(int(bid) for bid in body_ids)
    for jid in range(model.njnt):
        if int(model.jnt_bodyid[jid]) not in robot_bodies:
            continue
        jtype = int(model.jnt_type[jid])
        count = (
            6 if jtype == int(mujoco.mjtJoint.mjJNT_FREE)
            else 3 if jtype == int(mujoco.mjtJoint.mjJNT_BALL)
            else 1
        )
        start = int(model.jnt_dofadr[jid])
        dofs.extend(range(start, start + count))
    if not dofs:
        raise ApplyError("模型里找不到机器人关节自由度，无法应用 joint_damping_scale")
    return np.asarray(dofs, dtype=int)


def _swap_textures(model: mujoco.MjModel, rng: np.random.Generator) -> bool:
    """在"尺寸相同"的材质之间置换 RGB 贴图，模拟外观域随机化。

    只在同一 (width, height, nchannel) 分组内置换，避免把 2048x2048 的
    机器人皮肤贴到 1000x1000 的地面上导致拉伸/错位。返回是否发生了置换。
    """
    rgb = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
    groups: Dict[tuple, list] = {}
    for mid in range(model.nmat):
        tid = int(model.mat_texid[mid, rgb])
        if tid < 0:
            continue
        key = (int(model.tex_width[tid]), int(model.tex_height[tid]),
               int(model.tex_nchannel[tid]))
        groups.setdefault(key, []).append(mid)
    swapped = False
    for mats in groups.values():
        if len(mats) < 2:
            continue
        tids = [int(model.mat_texid[mid, rgb]) for mid in mats]
        perm = rng.permutation(len(tids))
        for pos, mid in enumerate(mats):
            model.mat_texid[mid, rgb] = tids[int(perm[pos])]
        swapped = True
    return swapped


# ── 结果对象 ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelReport:
    """模型层改动摘要，用于日志与 session meta。"""

    applied: Tuple[str, ...] = ()
    bodies: int = 0
    geoms: int = 0
    dofs: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "applied": list(self.applied),
            "bodies": self.bodies,
            "geoms": self.geoms,
            "dofs": self.dofs,
        }


@dataclass(frozen=True)
class SensorParams:
    """每个 episode 固定的传感器扰动参数。"""

    gyro_bias: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    accel_bias: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    gyro_walk: float = 0.0
    encoder_bits: Optional[int] = None
    force_noise_std: float = 0.0
    dropout_prob: float = 0.0
    latency_steps: int = 0
    latency_ms: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "gyro_bias": list(self.gyro_bias),
            "accel_bias": list(self.accel_bias),
            "gyro_walk": self.gyro_walk,
            "encoder_bits": self.encoder_bits,
            "force_noise_std": self.force_noise_std,
            "dropout_prob": self.dropout_prob,
            "latency_steps": self.latency_steps,
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True)
class RuntimeParams:
    """控制器侧参数；模型层不写这些字段。"""

    kp_scale: float = 1.0
    kd_scale: float = 1.0
    motor_delay_steps: int = 0
    control_dt: Optional[float] = None
    sensor: SensorParams = field(default_factory=SensorParams)

    def scaled_gains(self, kp: np.ndarray, kd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """把缩放系数落到官方配置的 PD 增益上。"""
        return np.asarray(kp, dtype=float) * self.kp_scale, np.asarray(kd, dtype=float) * self.kd_scale

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kp_scale": self.kp_scale,
            "kd_scale": self.kd_scale,
            "motor_delay_steps": self.motor_delay_steps,
            "control_dt": self.control_dt,
            "sensor": self.sensor.as_dict(),
        }


@dataclass(frozen=True)
class AppliedSample:
    model: ModelReport
    runtime: RuntimeParams

    def as_dict(self) -> Dict[str, Any]:
        return {"model": self.model.as_dict(), "runtime": self.runtime.as_dict()}


# ── 模型层应用 ──────────────────────────────────────────────────────────────
def apply_model(
    model: mujoco.MjModel,
    params: Mapping[str, Any],
    *,
    data: Optional[mujoco.MjData] = None,
    rng: Optional[np.random.Generator] = None,
) -> ModelReport:
    """把样本中的模型参数写到 MjModel，并刷新派生常量。"""
    flat = _check_supported(params)
    applied = []
    bodies = geoms = dofs = 0

    robot_ids = _robot_body_ids(model)

    if "dynamics.mass_scale" in flat:
        raw = flat["dynamics.mass_scale"]
        arr = np.asarray(raw, dtype=float)
        # 质量与主惯量按同一比例缩放，保持每个 body 的密度一致性。
        if arr.ndim == 0:
            scale = _scalar(raw, "dynamics.mass_scale", positive=True)
            model.body_mass[robot_ids] *= scale
            model.body_inertia[robot_ids] *= scale
        elif arr.shape == (robot_ids.size,):
            if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
                raise ApplyError("dynamics.mass_scale: 逐 body 缩放必须为正的有限值")
            model.body_mass[robot_ids] *= arr
            model.body_inertia[robot_ids] *= arr[:, None]
        else:
            raise ApplyError(
                "dynamics.mass_scale: 需要标量或 "
                f"({robot_ids.size},) 的数组，实际形状 {arr.shape}"
            )
        bodies = max(bodies, int(robot_ids.size))
        applied.append("dynamics.mass_scale")

    if "dynamics.com_offset_m" in flat:
        offset = np.asarray(flat["dynamics.com_offset_m"], dtype=float)
        # (3,) 表示所有机器人 body 用同一偏移；(N,3) 才是逐 body 独立偏移。
        if offset.shape == (3,):
            model.body_ipos[robot_ids] += offset
        elif offset.shape == (robot_ids.size, 3):
            model.body_ipos[robot_ids] += offset
        else:
            raise ApplyError(
                "dynamics.com_offset_m: 需要 (3,) 或 "
                f"({robot_ids.size}, 3) 的数组，实际形状 {offset.shape}"
            )
        if not np.all(np.isfinite(offset)):
            raise ApplyError("dynamics.com_offset_m: 必须都是有限值")
        # 质心不在 body 原点后，"simple body" 的 sameframe 前提被破坏，
        # 必须先撤销复用标记，mj_setConst 才能重算派生量。
        model.body_simple[robot_ids] = 0
        model.body_sameframe[robot_ids] = 0
        bodies = max(bodies, int(robot_ids.size))
        applied.append("dynamics.com_offset_m")

    if "contact.ground_friction" in flat:
        mu = _scalar(flat["contact.ground_friction"], "contact.ground_friction", nonneg=True)
        ground = _ground_geoms(model)
        # 只改滑动分量；扭转/滚动保持解析器默认值。
        model.geom_friction[ground, 0] = mu
        geoms = max(geoms, int(ground.size))
        applied.append("contact.ground_friction")

    if "contact.robot_friction_scale" in flat:
        scale = _scalar(
            flat["contact.robot_friction_scale"], "contact.robot_friction_scale", positive=True
        )
        robot_geoms = _robot_geoms(model, robot_ids)
        model.geom_friction[robot_geoms] *= scale
        geoms = max(geoms, int(robot_geoms.size))
        applied.append("contact.robot_friction_scale")

    if "contact.solref_timeconst_scale" in flat:
        scale = _scalar(
            flat["contact.solref_timeconst_scale"], "contact.solref_timeconst_scale", positive=True
        )
        target = np.zeros(model.ngeom, dtype=bool)
        target[_robot_geoms(model, robot_ids)] = True
        target[_ground_geoms(model)] = True
        model.geom_solref[target, 0] *= scale
        geoms = max(geoms, int(target.sum()))
        applied.append("contact.solref_timeconst_scale")

    if "actuator.joint_damping_scale" in flat:
        scale = _scalar(
            flat["actuator.joint_damping_scale"], "actuator.joint_damping_scale", nonneg=True
        )
        dof_idx = _robot_joint_dofs(model, robot_ids)
        base = np.where(
            model.dof_damping[dof_idx] > 0.0,
            model.dof_damping[dof_idx],
            JOINT_DAMPING_BASELINE,
        )
        model.dof_damping[dof_idx] = base * scale
        dofs = max(dofs, int(dof_idx.size))
        applied.append("actuator.joint_damping_scale")

    if "actuator.torque_limit_scale" in flat:
        scale = _scalar(
            flat["actuator.torque_limit_scale"], "actuator.torque_limit_scale", positive=True
        )
        # 只改驱动机器人关节的执行器；场景物体即使有执行器也不受影响。
        model.actuator_ctrlrange[_robot_actuators(model, robot_ids)] *= scale
        applied.append("actuator.torque_limit_scale")

    # ── 相机内外参 ──────────────────────────────────────────────────────
    # 相机在 apply 之前已由 add_cameras 注入；这里直接改编译后的模型数组。
    # batch 等没有相机的场景下 ncam=0，跳过而不是报错（相机参数与运动指标无关）。
    if "camera.fovy_scale" in flat and model.ncam > 0:
        scale = _scalar(flat["camera.fovy_scale"], "camera.fovy_scale", positive=True)
        model.cam_fovy[:] = np.clip(model.cam_fovy * scale, 1e-3, 179.0)
        applied.append("camera.fovy_scale")

    if "camera.pos_noise_m" in flat and model.ncam > 0:
        noise = _vec(flat["camera.pos_noise_m"], "camera.pos_noise_m", 3)
        model.cam_pos[:] += noise  # 相机相对挂载 body 的局部位移
        applied.append("camera.pos_noise_m")

    # ── 光照与材质 ──────────────────────────────────────────────────────
    # MuJoCo 的 light_intensity 字段默认 0 且渲染器并不消费，真正生效的是
    # diffuse/specular，因此按同一比例缩放这两个分量。
    if "rendering.light_intensity_scale" in flat and model.nlight > 0:
        scale = _scalar(
            flat["rendering.light_intensity_scale"],
            "rendering.light_intensity_scale", nonneg=True,
        )
        model.light_diffuse[:] *= scale
        model.light_specular[:] *= scale
        applied.append("rendering.light_intensity_scale")

    if "rendering.texture_swap_prob" in flat:
        prob = _scalar(
            flat["rendering.texture_swap_prob"], "rendering.texture_swap_prob"
        )
        if not 0.0 <= prob <= 1.0:
            raise ApplyError(
                f"rendering.texture_swap_prob: 必须在 [0, 1]，实际是 {prob}"
            )
        if rng is None:
            raise ApplyError("rendering.texture_swap_prob: 需要传入 rng 才能抽样")
        if prob > 0.0 and rng.random() < prob and _swap_textures(model, rng):
            applied.append("rendering.texture_swap_prob")

    if applied:
        # 质量/惯量变化后 invweight、subtreemass 等派生常量必须重算。
        mujoco.mj_setConst(model, data if data is not None else mujoco.MjData(model))

    return ModelReport(applied=tuple(applied), bodies=bodies, geoms=geoms, dofs=dofs)


def runtime_params(params: Mapping[str, Any], *, dt_ctrl: float) -> RuntimeParams:
    """从样本里解析控制器侧参数。"""
    flat = _check_supported(params)
    dt = _scalar(dt_ctrl, "dt_ctrl", positive=True)

    control_dt = None
    if "actuator.control_dt_ms" in flat:
        control_dt = _scalar(
            flat["actuator.control_dt_ms"], "actuator.control_dt_ms", positive=True
        ) / 1000.0
    # 延迟按实际控制步长换算：控制周期本身被随机化时，20 ms 不再等于 1 步。
    step_dt = control_dt if control_dt is not None else dt

    def delay_steps(key: str) -> int:
        ms = _scalar(flat[key], key, nonneg=True)
        # 非零延迟至少占一个控制步，否则 5 ms 这类配置会被静默算成 0。
        return int(math.ceil(ms / 1000.0 / step_dt - 1e-9)) if ms > 0 else 0

    bits = None
    if "sensors.encoder_bits" in flat:
        bits = _integer(flat["sensors.encoder_bits"], "sensors.encoder_bits", minimum=1)
        if bits > 52:
            raise ApplyError(f"sensors.encoder_bits: 超过 52 位没有物理意义，实际是 {bits}")

    dropout = 0.0
    if "sensors.frame_dropout_prob" in flat:
        dropout = _scalar(flat["sensors.frame_dropout_prob"], "sensors.frame_dropout_prob")
        if not 0.0 <= dropout <= 1.0:
            raise ApplyError(f"sensors.frame_dropout_prob: 必须在 [0, 1]，实际是 {dropout}")

    gyro_bias = (0.0, 0.0, 0.0)
    if "sensors.imu_bias_gyro_rads" in flat:
        gyro_bias = tuple(
            _vec(flat["sensors.imu_bias_gyro_rads"], "sensors.imu_bias_gyro_rads", 3).tolist()
        )

    accel_bias = (0.0, 0.0, 0.0)
    if "sensors.imu_bias_accel_mps2" in flat:
        accel_bias = tuple(
            _vec(flat["sensors.imu_bias_accel_mps2"], "sensors.imu_bias_accel_mps2", 3).tolist()
        )

    gyro_walk = 0.0
    if "sensors.imu_random_walk_gyro" in flat:
        gyro_walk = _scalar(
            flat["sensors.imu_random_walk_gyro"], "sensors.imu_random_walk_gyro", nonneg=True
        )

    force_std = 0.0
    if "sensors.force_noise_std_n" in flat:
        force_std = _scalar(
            flat["sensors.force_noise_std_n"], "sensors.force_noise_std_n", nonneg=True
        )

    latency_ms_val = 0.0
    latency_steps = 0
    if "sensors.latency_ms" in flat:
        latency_ms_val = _scalar(flat["sensors.latency_ms"], "sensors.latency_ms", nonneg=True)
        latency_steps = delay_steps("sensors.latency_ms")

    return RuntimeParams(
        kp_scale=_scalar(flat.get("actuator.kp_scale", 1.0), "actuator.kp_scale", positive=True),
        kd_scale=_scalar(flat.get("actuator.kd_scale", 1.0), "actuator.kd_scale", positive=True),
        motor_delay_steps=(
            delay_steps("actuator.motor_delay_ms")
            if "actuator.motor_delay_ms" in flat else 0
        ),
        control_dt=control_dt,
        sensor=SensorParams(
            gyro_bias=gyro_bias,
            accel_bias=accel_bias,
            gyro_walk=gyro_walk,
            encoder_bits=bits,
            force_noise_std=force_std,
            dropout_prob=dropout,
            latency_steps=latency_steps,
            latency_ms=latency_ms_val,
        ),
    )


def body_shapes(model: mujoco.MjModel, plan) -> Dict[str, tuple]:
    """按模型里的机器人 body 数量，为 per_body 参数生成采样形状。"""
    return per_body_shapes(plan, int(_robot_body_ids(model).size))


def apply_sample(
    model: mujoco.MjModel,
    sample: Mapping[str, Any],
    *,
    dt_ctrl: float,
    data: Optional[mujoco.MjData] = None,
    rng: Optional[np.random.Generator] = None,
) -> AppliedSample:
    """把 sampler.sample() 的返回值一次性落到模型与运行时参数。"""
    params = sample.get("params")
    if params is None:
        raise ApplyError("sample 缺少 params 字段")
    return AppliedSample(
        model=apply_model(model, params, data=data, rng=rng),
        runtime=runtime_params(params, dt_ctrl=dt_ctrl),
    )


# ── 观测扰动 ────────────────────────────────────────────────────────────────
class SensorModel:
    """把传感器噪声、量化、丢帧与延迟加到原始观测信号上。

    调用方负责提供 RNG（与采样同源派生），从而保证同一 episode 可复现。
    """

    def __init__(self, params: SensorParams, *, dt: float, rng: np.random.Generator):
        self.params = params
        self.dt = _scalar(dt, "dt", positive=True)
        self.rng = rng
        self._walk = np.zeros(3)
        self._last: Optional[Tuple[np.ndarray, ...]] = None
        self._queue: Deque[Tuple[np.ndarray, ...]] = deque()
        self._quantum = (
            2.0 * math.pi / float(2 ** params.encoder_bits)
            if params.encoder_bits is not None else None
        )

    def observe(self, q, qd, ang_vel, proj_grav):
        """返回 (q, qd, ang_vel, proj_grav)，类型与输入一致。"""
        q = np.array(q, dtype=float, copy=True)
        qd = np.array(qd, dtype=float, copy=True)
        ang_vel = np.array(ang_vel, dtype=float, copy=True)
        proj_grav = np.array(proj_grav, dtype=float, copy=True)

        if self._quantum is not None:
            q = np.round(q / self._quantum) * self._quantum

        bias = np.asarray(self.params.gyro_bias, dtype=float)
        if np.any(bias) or self.params.gyro_walk:
            if self.params.gyro_walk:
                step = self.params.gyro_walk * math.sqrt(self.dt)
                self._walk += self.rng.normal(0.0, step, 3)
            ang_vel = ang_vel + bias + self._walk

        accel_bias = np.asarray(self.params.accel_bias, dtype=float)
        if np.any(accel_bias):
            # 加速度零偏等价于重力方向的小角度倾斜：加到投影重力上再归一化。
            proj_grav = proj_grav + accel_bias / GRAVITY
            norm = float(np.linalg.norm(proj_grav))
            if norm > 0:
                proj_grav = proj_grav / norm

        out = (q, qd, ang_vel, proj_grav)

        if (
            self.params.dropout_prob
            and self._last is not None
            and self.rng.random() < self.params.dropout_prob
        ):
            return self._last
        self._last = out

        if self.params.latency_steps > 0:
            self._queue.append(out)
            if len(self._queue) > self.params.latency_steps:
                return self._queue.popleft()
        return out

    def force(self, values, rng=None):
        """给足端力读数加白噪声。

        rng 不传时复用观测噪声流；采集落盘应传入独立 RNG，避免力噪声采样
        消耗策略观测的随机数、改变后续噪声序列。
        """
        values = np.asarray(values, dtype=float)
        if not self.params.force_noise_std:
            return values
        rng = self.rng if rng is None else rng
        return values + rng.normal(0.0, self.params.force_noise_std, values.shape)
