---
name: t800-domain-randomization
description: Add or extend T800 MuJoCo domain randomization dimensions through schema, sampler, apply, config and tests. Use when adding or fixing mass, friction, sensor, camera or rendering randomization. Do not use for one-off sim runs.
---

# T800 域随机化维度扩展

本仓库 (`t800_sim`) 的域随机化按固定分层实现。新增或修改一个维度时，严格按下面的分层与顺序推进，保证可复现、下游一致、不破坏向后兼容。

## 适用 / 不适用

- 适用：新增随机化参数、修改分布/作用域、实现 per-body、修复 schema/apply 不一致、补齐下游列或校验。
- 不适用：只想跑一次仿真（直接用 CLI）；与随机化无关的功能开发。

## 分层地图（改哪里）

- `configs/randomization.yaml`：唯一权威配置。参数需显式 `type`；物理作用域需显式 `scope`；逐对象用 `per_body: true`。
- `t800_sim/randomization/schema.py`：解析与校验。`_SCOPES`、`_REQUIRED_SCOPES`（物理参数强制作用域）、`_PER_BODY_PARAMS`（仅 `com_offset_m`、`mass_scale`）。非法即抛 `SchemaError`。
- `t800_sim/randomization/sampler.py`：`derive_rng(seed,session,episode)`、`sample(plan, session, episode, shapes=...)`、`per_body_shapes(plan, count)`、`_to_python`。采样顺序按 yaml 书写顺序，跨进程稳定（zlib.crc32）。
- `t800_sim/randomization/apply.py`：`MODEL_PARAMS`/`RUNTIME_PARAMS`；`apply_model(model, params, data=, rng=)` 改 MjModel；`runtime_params(...)` 汇总控制器参数；`SensorModel` 扰动观测；`body_shapes(model, plan)` 给出逐 body 形状。
- `t800_sim/tasks/record.py`：`build_model → sample(shapes=body_shapes) → apply_sample(rng=) → MjData/Robot → RlController`。apply 必须早于 MjData/Robot（缩放后的 ctrlrange 才会被读）。
- `t800_sim/batch.py`：`rollout(...)` 无窗口评测；`RAND_FIELDS`/`_rand_values` 把标量参数展开成 CSV 列。
- `t800_sim/dataset.py`：`FIELDS` 与 `_row()` 生成 manifest；向量参数在单列里**取均值**。
- `t800_sim/capture.py`：相机注入、逐帧落盘、2D 投影、相机链路丢帧/延迟。
- `t800_sim/validate.py`：session 完整性校验（新增数据列/文件要同步检查）。

## 操作步骤

1. 判定层级：改模型物理量 → `MODEL_PARAMS`；只影响控制器/观测 → `RUNTIME_PARAMS`。
2. schema：物理作用域语义进 `_REQUIRED_SCOPES`；逐对象参数进 `_PER_BODY_PARAMS`（要求 `scope: robot_bodies` 且为连续分布）。
3. apply：在 `apply_model`（模型层）或 `runtime_params`（运行时）实现；随机项需要 `rng`，缺失要显式报错；向量/标量双兼容；改了质量/惯量/阻尼等派生量后必须 `mj_setConst`。
4. sampler：通常无需改动（`per_body_shapes` 自动按 body 数展开）；确认 `_base_shape` 正确。
5. config：写 `type` + `scope`（+ `per_body`）+ `desc`；按需把 group 设为 `enabled: true`。
6. 下游：`batch.RAND_FIELDS`/`_rand_values`、`dataset.FIELDS`/`_row` 同步加列；**向量在单列取均值**，禁止把 list 直接 `float()`。
7. 测试：加 GL-free 单测（用含 `LINK_` body 的最小 XML）；把断言旧行为的测试一并更新。
8. 验证：pytest → record/batch/validate/manifest 全链路。

## 必须保持的不变量

- 物理参数必须显式 `scope`；未知 scope/字段立即报错，不静默忽略。
- 采样值必须是 Python 原生类型（经 `_to_python`），可直接 JSON 序列化。
- 默认配置行为即"不随机化时与改造前完全一致"（向后兼容）。
- 逐 body 参数：apply 接受标量或 `(N,)`，`N` = `LINK_`/`Motor_` body 数；惯量按同一系数缩放（`arr[:, None]`）。
- 相机/渲染注意点：`light_intensity` 字段无效（默认 0 且渲染器不消费），要缩放 `light_diffuse`/`light_specular`；贴图 RGB 在 `mat_texid[:, mjTEXROLE_RGB=1]`，只在同 `(w,h,nch)` 分组内置换。
- 各 RNG 流独立：采样按 `(seed,session,episode)`；策略观测 `0x5E05`；力噪声 `0xF0`；相机链路 `0xC472`；apply 随机项 `0x0C0A`。

## 验证命令（在 WSL 内运行）

```bash
cd /home/wsl-ubuntu24/T800Robot/code
PYTHONPATH=. python3 -m pytest -q tests/
PYTHONPATH=. python3 -m t800_sim.cli record --randomize --duration 0.5 --seed 0 --out /tmp/r
PYTHONPATH=. python3 -m t800_sim.cli validate --session /tmp/r
PYTHONPATH=. python3 -m t800_sim.cli batch --randomize --vx 0.6 --episodes 2 --workers 2 --out /tmp/b.csv
PYTHONPATH=. python3 -m t800_sim.cli manifest --root /tmp/r
```

## 重复跑同类仿真

- 网格：`batch --randomize --vx ... --vy ... --episodes N --seed S --workers 1/2`。
- 采集：`record-grid --seeds ... --episodes N`（按 `task/scene/seed/ep` 分层）。
- 每次跑完都要 `validate` + `manifest`；随机化列用于按物理参数分组统计。

## 环境坑（本仓库踩过）

- 长时 CLI 可能被 harness 杀掉且无输出 → 用 `subprocess.run([...], capture_output=True)` 包装后运行，`PYTHONPATH=.`。
- 直接编辑 `\\wsl.localhost\...` UNC 大文件可能被截断 → 改用 Windows 临时目录里的 Python patch 脚本，在 WSL 里执行（`replace` 前校验命中数==1）。
- 不要对长命令用管道（`| tail`）——上游被 kill 会被 tail 的退出码掩盖。
- `batch --workers 4` 偶发被杀；用 `--workers 1` 或 `2` 稳定。
