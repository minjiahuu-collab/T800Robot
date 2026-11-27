"""域随机化：配置驱动的参数采样与仿真参数应用。"""

from .apply import (
    ApplyError,
    AppliedSample,
    ModelReport,
    RuntimeParams,
    SensorModel,
    SensorParams,
    apply_model,
    apply_sample,
    body_shapes,
    runtime_params,
)
from .schema import Plan, ParamSpec, GroupSpec, SchemaError, load_plan
from .sampler import derive_rng, per_body_shapes, sample, sample_many

__all__ = [
    "Plan",
    "ParamSpec",
    "GroupSpec",
    "SchemaError",
    "load_plan",
    "sample",
    "sample_many",
    "derive_rng",
    "per_body_shapes",
    "ApplyError",
    "AppliedSample",
    "ModelReport",
    "RuntimeParams",
    "SensorParams",
    "SensorModel",
    "apply_model",
    "apply_sample",
    "body_shapes",
    "runtime_params",
]
