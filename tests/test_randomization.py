"""域随机化 schema、采样器与应用层的回归测试。"""

import json
import textwrap

import mujoco
import numpy as np
import pytest

from t800_sim.randomization.apply import (
    ApplyError,
    SensorModel,
    SensorParams,
    _swap_textures,
    apply_model,
    apply_sample,
    body_shapes,
    runtime_params,
)
from t800_sim.capture import (
    available_force_sensors,
    camera_intrinsics,
    project_point,
)
from t800_sim.randomization.sampler import derive_rng, per_body_shapes, sample
from t800_sim.randomization.schema import SchemaError, load_plan

# 最小模型：一个 LINK_ 机器人 body + 一个独立物体 + 平面地面，
# 用来验证作用域过滤（场景物体不应被机器人参数波及）。
TINY_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="LINK_A" pos="0 0 1">
      <joint name="j1" type="hinge" damping="0.4" axis="0 0 1"/>
      <geom name="g1" type="box" size="0.1 0.1 0.1" density="1000"/>
    </body>
    <body name="box_obj" pos="0 0 0.05">
      <freejoint/>
      <geom name="box_geom" type="box" size="0.05 0.05 0.05" density="300"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j1" ctrlrange="-10 10"/>
  </actuator>
</mujoco>
"""


# 两个 LINK_ body，用来验证 com_offset 的逐 body 偏移。
TWO_BODY_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="LINK_A" pos="0 0 1">
      <joint name="ja" type="hinge" damping="0.4" axis="0 0 1"/>
      <geom name="ga" type="box" size="0.1 0.1 0.1" density="1000"/>
      <site name="foot_site"/>
    </body>
    <body name="LINK_B" pos="0 0 1.5">
      <joint name="jb" type="hinge" damping="0.4" axis="0 0 1"/>
      <geom name="gb" type="box" size="0.1 0.1 0.1" density="1000"/>
    </body>
  </worldbody>
  <sensor>
    <force name="force_left_foot" site="foot_site"/>
  </sensor>
</mujoco>
"""


def _plan_from(tmp_path, body: str):
    """把 YAML 片段写成临时配置并加载。"""
    path = tmp_path / "rand.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return load_plan(path)


def _tiny_models():
    return mujoco.MjModel.from_xml_string(TINY_XML), mujoco.MjModel.from_xml_string(TINY_XML)


def _body_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _geom_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def test_default_plan_loads():
    """仓库自带配置必须能通过校验，且至少有启用参数。"""
    plan = load_plan()
    assert plan.version == 1
    assert plan.count_enabled() > 0
    assert plan.source_hash


def test_samples_are_reproducible():
    """同一 (seed, session, episode) 必须得到完全相同的样本。"""
    plan = load_plan()
    assert sample(plan, "walk_vx0.30", 0) == sample(plan, "walk_vx0.30", 0)


def test_episode_and_session_change_sample():
    """换 episode 或 session 必须得到不同样本。"""
    plan = load_plan()
    base = sample(plan, "s0", 0)
    assert sample(plan, "s0", 1) != base
    assert sample(plan, "s1", 0) != base


def test_per_session_mode_shares_sample(tmp_path):
    """per_session 模式下同一 session 的不同 episode 共享参数。"""
    plan = _plan_from(tmp_path, """
        version: 1
        seed: 7
        mode: per_session
        groups:
          g:
            params:
              x: {type: uniform, low: 0.0, high: 1.0}
    """)
    assert sample(plan, "s", 0)["params"] == sample(plan, "s", 5)["params"]
    assert sample(plan, "s", 0)["params"] != sample(plan, "t", 0)["params"]


def test_uniform_stays_in_range():
    plan = load_plan()
    for episode in range(50):
        params = sample(plan, "range", episode)["params"]
        assert 0.90 <= params["dynamics"]["mass_scale"] <= 1.10
        assert 0.30 <= params["contact"]["ground_friction"] <= 1.20


def test_unknown_field_is_rejected(tmp_path):
    """拼写错误必须报错，而不是被静默忽略。"""
    with pytest.raises(SchemaError, match="不认识的字段"):
        _plan_from(tmp_path, """
            version: 1
            groups:
              g:
                params:
                  x: {type: uniform, low: 0.0, high: 1.0, higth: 2.0}
        """)


def test_bad_range_is_rejected(tmp_path):
    with pytest.raises(SchemaError, match="low > 0"):
        _plan_from(tmp_path, """
            version: 1
            groups:
              g:
                params:
                  x: {type: loguniform, low: 0.0, high: 1.0}
        """)


def test_disabled_group_is_skipped(tmp_path):
    plan = _plan_from(tmp_path, """
        version: 1
        seed: 1
        groups:
          active:
            params:
              a: {type: fixed, value: 1}
          inactive:
            enabled: false
            params:
              b: {type: fixed, value: 2}
    """)
    assert sample(plan, "s", 0)["params"] == {"active": {"a": 1}}


def test_yaml_boolean_key_is_rejected(tmp_path):
    """YAML 会把 on/off 解析成布尔键，必须报错而不是生成奇怪的维度名。"""
    with pytest.raises(SchemaError, match="维度名必须是字符串"):
        _plan_from(tmp_path, """
            version: 1
            groups:
              on:
                params:
                  a: {type: fixed, value: 1}
        """)


def test_rng_is_stable_across_calls():
    """种子派生不能依赖进程内哈希盐。"""
    plan = load_plan()
    a = derive_rng(plan, "session-x", 3).random(5).tolist()
    b = derive_rng(plan, "session-x", 3).random(5).tolist()
    assert a == b


# ── 应用层 ─────────────────────────────────────────────────────────────────
def test_mass_scale_updates_mass_inertia_and_constants():
    """质量缩放必须同步缩放主惯量，并刷新 invweight 等派生量。"""
    model, ref = _tiny_models()
    bid = _body_id(model, "LINK_A")
    report = apply_model(model, {"dynamics": {"mass_scale": 1.5}})

    assert model.body_mass[bid] == pytest.approx(ref.body_mass[bid] * 1.5)
    np.testing.assert_allclose(model.body_inertia[bid], ref.body_inertia[bid] * 1.5)
    assert model.body_invweight0[bid, 1] != pytest.approx(ref.body_invweight0[bid, 1])
    assert report.applied == ("dynamics.mass_scale",)
    assert report.bodies == 1


def test_com_offset_only_touches_robot_bodies():
    """质心偏移只作用于 LINK_/Motor_ body，场景物体保持默认。"""
    model, ref = _tiny_models()
    robot = _body_id(model, "LINK_A")
    scene = _body_id(model, "box_obj")
    offset = np.array([0.01, -0.02, 0.03])
    apply_model(model, {"dynamics": {"com_offset_m": offset.tolist()}})

    np.testing.assert_allclose(model.body_ipos[robot], ref.body_ipos[robot] + offset)
    np.testing.assert_allclose(model.body_ipos[scene], ref.body_ipos[scene])


def test_contact_scopes_are_separated():
    """地面摩擦只改 floor，机器人摩擦只改机器人 geom。"""
    model, ref = _tiny_models()
    floor = _geom_id(model, "floor")
    robot_geom = _geom_id(model, "g1")
    scene_geom = _geom_id(model, "box_geom")
    apply_model(model, {"contact": {
        "ground_friction": 0.7,
        "robot_friction_scale": 2.0,
        "solref_timeconst_scale": 1.5,
    }})

    assert model.geom_friction[floor, 0] == pytest.approx(0.7)
    np.testing.assert_allclose(model.geom_friction[robot_geom], ref.geom_friction[robot_geom] * 2.0)
    np.testing.assert_allclose(model.geom_friction[scene_geom], ref.geom_friction[scene_geom])
    assert model.geom_solref[floor, 0] == pytest.approx(ref.geom_solref[floor, 0] * 1.5)
    assert model.geom_solref[robot_geom, 0] == pytest.approx(ref.geom_solref[robot_geom, 0] * 1.5)
    assert model.geom_solref[scene_geom, 0] == pytest.approx(ref.geom_solref[scene_geom, 0])


def test_actuator_scale_and_damping():
    """力矩上限与关节阻尼按比例缩放，且阻尼确实写进了 dof_damping。"""
    model, ref = _tiny_models()
    dof = int(model.jnt_dofadr[0])
    assert model.dof_damping[dof] == pytest.approx(0.4)

    apply_model(model, {"actuator": {
        "torque_limit_scale": 0.8,
        "joint_damping_scale": 2.0,
    }})

    np.testing.assert_allclose(model.actuator_ctrlrange, ref.actuator_ctrlrange * 0.8)
    assert model.dof_damping[dof] == pytest.approx(0.8)


def test_joint_damping_uses_baseline_when_model_has_none():
    """模型基线阻尼为 0 时，缩放应围绕 0.6 的基線生效而不是继续得到 0。"""
    xml = TINY_XML.replace('damping="0.4" ', "")
    model = mujoco.MjModel.from_xml_string(xml)
    dof = int(model.jnt_dofadr[0])
    assert model.dof_damping[dof] == 0.0

    apply_model(model, {"actuator": {"joint_damping_scale": 2.0}})

    assert model.dof_damping[dof] == pytest.approx(0.6 * 2.0)


def test_manifest_exposes_randomization_columns(tmp_path):
    """manifest 应把关键随机化参数展开成列，便于按物理参数分组。"""
    import csv

    from t800_sim.dataset import build_manifest

    session = tmp_path / "walk/seed_0003/ep_0000"
    session.mkdir(parents=True)
    meta = {
        "schema_version": "1.0",
        "task": "walk",
        "scene_id": "empty",
        "seed": 3,
        "episode": 0,
        "frames": 0,
        "duration": 0.0,
        "cmd": [0.6, 0.0, 0.0],
        "cameras": [],
        "metrics": {},
        "randomization": {
            "enabled": True,
            "plan": {"plan_hash": "abc123", "session": "default"},
            "params": {
                "dynamics": {"mass_scale": 1.05},
                "contact": {
                    "ground_friction": 0.8,
                    "robot_friction_scale": 1.0,
                    "solref_timeconst_scale": 1.0,
                },
                "actuator": {
                    "joint_damping_scale": 1.2,
                    "torque_limit_scale": 0.95,
                    "kp_scale": 1.0,
                    "kd_scale": 1.0,
                },
            },
        },
    }
    (session / "meta.json").write_text(json.dumps(meta))

    build_manifest(tmp_path, val_ratio=0.3)

    rows = list(csv.DictReader((tmp_path / "manifest.csv").open()))
    assert len(rows) == 1
    row = rows[0]
    assert row["randomized"] == "True"
    assert row["plan_hash"] == "abc123"
    assert float(row["mass_scale"]) == pytest.approx(1.05)
    assert float(row["ground_friction"]) == pytest.approx(0.8)


def test_unknown_param_fails_loudly():
    """应用层没实现的参数必须报错，不能静默忽略。"""
    model, _ = _tiny_models()
    with pytest.raises(ApplyError, match="还不支持"):
        apply_model(model, {"camera": {"not_a_real_param": 1.02}})


def test_runtime_params_derive_steps_and_values():
    rt = runtime_params(
        {
            "actuator": {
                "kp_scale": 1.1,
                "kd_scale": 0.9,
                "motor_delay_ms": 20,
                "control_dt_ms": 50,
            },
            "sensors": {
                "imu_bias_gyro_rads": [0.001, 0.002, 0.003],
                "encoder_bits": 17,
                "frame_dropout_prob": 0.05,
                "latency_ms": 10,
            },
        },
        dt_ctrl=0.02,
    )
    assert rt.motor_delay_steps == 1          # 20 ms / 20 ms
    assert rt.control_dt == pytest.approx(0.05)
    assert rt.sensor.encoder_bits == 17
    assert rt.sensor.latency_steps == 1
    assert rt.sensor.gyro_bias == (0.001, 0.002, 0.003)

    kp, kd = rt.scaled_gains(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    np.testing.assert_allclose(kp, [1.1, 2.2])
    np.testing.assert_allclose(kd, [2.7, 3.6])


def test_apply_sample_is_serializable():
    model, _ = _tiny_models()
    applied = apply_sample(
        model,
        {"params": {"dynamics": {"mass_scale": 1.02}, "sensors": {"encoder_bits": 16}}},
        dt_ctrl=0.02,
    )
    payload = json.dumps(applied.as_dict(), ensure_ascii=False)
    assert "mass_scale" in payload and "encoder_bits" in payload


def test_sensor_model_is_reproducible():
    params = SensorParams(
        gyro_bias=(0.001, -0.002, 0.003),
        gyro_walk=0.002,
        encoder_bits=10,
        latency_steps=2,
    )

    def run():
        sim = SensorModel(params, dt=0.02, rng=np.random.default_rng(7))
        out = []
        for i in range(5):
            obs = sim.observe([0.001 * i], [0.1], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0])
            out.append(np.concatenate(obs))
        return np.stack(out)

    np.testing.assert_allclose(run(), run())


def test_sensor_quantization_and_dropout():
    sim = SensorModel(
        SensorParams(encoder_bits=8, dropout_prob=1.0),
        dt=0.02,
        rng=np.random.default_rng(0),
    )
    q, _, _, _ = sim.observe(np.array([0.001]), np.zeros(1), np.zeros(3), [0, 0, -1])
    step = 2.0 * np.pi / 256
    assert abs(float(q[0])) <= step / 2 + 1e-12

    # dropout=1：从第二帧开始保持上一帧输出。
    q2, _, _, _ = sim.observe(np.array([1.0]), np.ones(1), np.zeros(3), [0, 0, -1])
    np.testing.assert_allclose(q2, q)


# ── scope 与 per_body ──────────────────────────────────────────────────────
def test_unknown_scope_is_rejected(tmp_path):
    with pytest.raises(SchemaError, match="scope 必须是"):
        _plan_from(tmp_path, """
            version: 1
            groups:
              g:
                params:
                  x: {type: uniform, low: 0.0, high: 1.0, scope: robot_bodys}
        """)


def test_required_scope_is_enforced(tmp_path):
    with pytest.raises(SchemaError, match="必须显式声明 scope"):
        _plan_from(tmp_path, """
            version: 1
            groups:
              dynamics:
                params:
                  mass_scale: {type: uniform, low: 0.9, high: 1.1}
        """)
    with pytest.raises(SchemaError, match="scope 必须是 'robot_bodies'"):
        _plan_from(tmp_path, """
            version: 1
            groups:
              dynamics:
                params:
                  mass_scale: {type: uniform, low: 0.9, high: 1.1, scope: ground}
        """)


def test_per_body_is_restricted_to_vectors_and_mass(tmp_path):
    # 只有 com_offset_m / mass_scale 支持 per_body。
    with pytest.raises(SchemaError, match="不支持 per_body"):
        _plan_from(tmp_path, """
            version: 1
            groups:
              dynamics:
                params:
                  torque_limit_scale:
                    type: uniform
                    low: 0.85
                    high: 1.0
                    per_body: true
        """)


def _per_body_plan(tmp_path):
    return _plan_from(tmp_path, """
        version: 1
        seed: 11
        groups:
          dynamics:
            params:
              com_offset_m:
                type: normal
                mean: [0.0, 0.0, 0.0]
                std: [0.003, 0.003, 0.006]
                scope: robot_bodies
                per_body: true
    """)


def test_per_body_sampling_expands_shape(tmp_path):
    plan = _per_body_plan(tmp_path)
    shapes = per_body_shapes(plan, 4)
    assert shapes == {"dynamics.com_offset_m": (4, 3)}

    a = sample(plan, "s", 0, shapes=shapes)["params"]["dynamics"]["com_offset_m"]
    b = sample(plan, "s", 0, shapes=shapes)["params"]["dynamics"]["com_offset_m"]
    assert np.asarray(a).shape == (4, 3)
    assert a == b
    assert a[0] != a[1]


def test_body_shapes_uses_model_body_count(tmp_path):
    plan = _per_body_plan(tmp_path)
    model = mujoco.MjModel.from_xml_string(TWO_BODY_XML)
    assert body_shapes(model, plan) == {"dynamics.com_offset_m": (2, 3)}


def test_com_offset_applies_per_body():
    model = mujoco.MjModel.from_xml_string(TWO_BODY_XML)
    ref = mujoco.MjModel.from_xml_string(TWO_BODY_XML)
    a = _body_id(model, "LINK_A")
    b = _body_id(model, "LINK_B")
    offsets = [[0.01, 0.0, 0.0], [0.0, 0.02, 0.03]]

    apply_model(model, {"dynamics": {"com_offset_m": offsets}})

    np.testing.assert_allclose(model.body_ipos[a], ref.body_ipos[a] + offsets[0])
    np.testing.assert_allclose(model.body_ipos[b], ref.body_ipos[b] + offsets[1])

    with pytest.raises(ApplyError, match="需要 \\(3,\\) 或"):
        apply_model(ref, {"dynamics": {"com_offset_m": np.zeros((3, 3)).tolist()}})


def test_available_force_sensors():
    assert available_force_sensors(
        mujoco.MjModel.from_xml_string(TWO_BODY_XML)
    ) == ("force_left_foot",)
    assert available_force_sensors(mujoco.MjModel.from_xml_string(TINY_XML)) == ()


def test_force_noise_uses_explicit_rng():
    sim = SensorModel(
        SensorParams(force_noise_std=0.5),
        dt=0.02,
        rng=np.random.default_rng(1),
    )
    zeros = np.zeros(3)
    a = sim.force(zeros, rng=np.random.default_rng(9))
    b = sim.force(zeros, rng=np.random.default_rng(9))
    np.testing.assert_allclose(a, b)
    assert not np.allclose(a, zeros)

    clean = SensorModel(SensorParams(), dt=0.02, rng=np.random.default_rng(1))
    np.testing.assert_allclose(clean.force(zeros, rng=np.random.default_rng(9)), zeros)


# ── camera / rendering / 2D 投影 ─────────────────────────────────────────────
CAM_LIGHT_XML = """
<mujoco>
  <asset>
    <texture name="t1" type="2d" builtin="checker" rgb1="1 0 0" rgb2="0 1 0" width="32" height="32"/>
    <texture name="t2" type="2d" builtin="checker" rgb1="0 0 1" rgb2="1 1 0" width="32" height="32"/>
    <material name="m1" texture="t1"/>
    <material name="m2" texture="t2"/>
  </asset>
  <worldbody>
    <light name="sun" pos="0 0 3" dir="0 0 -1" diffuse="0.7 0.7 0.7" specular="0.3 0.3 0.3"/>
    <geom name="floor" type="plane" size="1 1 0.1" material="m1"/>
    <body name="LINK_A" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 0 1"/>
      <geom name="g1" type="box" size="0.1 0.1 0.1" material="m2"/>
      <camera name="cam0" pos="0 0 0.2" fovy="60"/>
    </body>
  </worldbody>
</mujoco>
"""


def _cam_light_model():
    return mujoco.MjModel.from_xml_string(CAM_LIGHT_XML)


def _cam_id(model, name="cam0"):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)


def test_camera_fovy_scale_applied():
    model, ref = _cam_light_model(), _cam_light_model()
    cid = _cam_id(model)
    apply_model(model, {"camera": {"fovy_scale": 1.1}})
    assert model.cam_fovy[cid] == pytest.approx(ref.cam_fovy[cid] * 1.1)


def test_camera_pos_noise_applied():
    model, ref = _cam_light_model(), _cam_light_model()
    cid = _cam_id(model)
    noise = [0.01, -0.02, 0.03]
    apply_model(model, {"camera": {"pos_noise_m": noise}})
    np.testing.assert_allclose(model.cam_pos[cid], ref.cam_pos[cid] + noise)


def test_light_intensity_scale_applied():
    model, ref = _cam_light_model(), _cam_light_model()
    apply_model(model, {"rendering": {"light_intensity_scale": 0.5}})
    np.testing.assert_allclose(model.light_diffuse, ref.light_diffuse * 0.5)
    np.testing.assert_allclose(model.light_specular, ref.light_specular * 0.5)


def test_texture_swap_prob_gating_and_invariant():
    rgb = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
    # prob=0 不置换
    model, ref = _cam_light_model(), _cam_light_model()
    apply_model(model, {"rendering": {"texture_swap_prob": 0.0}},
                rng=np.random.default_rng(0))
    np.testing.assert_array_equal(model.mat_texid, ref.mat_texid)
    # prob=1 只在同尺寸组内置换：贴图 id 多重集合保持不变
    model2 = _cam_light_model()
    before = sorted(int(x) for x in model2.mat_texid[:, rgb] if x >= 0)
    apply_model(model2, {"rendering": {"texture_swap_prob": 1.0}},
                rng=np.random.default_rng(0))
    after = sorted(int(x) for x in model2.mat_texid[:, rgb] if x >= 0)
    assert after == before


def test_texture_swap_requires_rng():
    with pytest.raises(ApplyError, match="rng"):
        apply_model(_cam_light_model(), {"rendering": {"texture_swap_prob": 0.5}})


def test_swap_textures_skips_singleton_groups():
    # 只有一个材质引用贴图时没有可置换对象。
    model = mujoco.MjModel.from_xml_string(CAM_LIGHT_XML)
    # 把两个材质都指向同一贴图 -> 两个组各 1 个成员 -> 不置换
    rgb = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
    t1 = int(model.mat_texid[0, rgb])
    model.mat_texid[1, rgb] = -1
    assert _swap_textures(model, np.random.default_rng(0)) is False


def test_camera_skipped_when_model_has_no_camera():
    # batch 用的模型没有相机：应用相机参数应静默跳过而不是报错。
    model = mujoco.MjModel.from_xml_string(TINY_XML)
    assert model.ncam == 0
    report = apply_model(model, {"camera": {"fovy_scale": 1.05, "pos_noise_m": [0.0, 0.0, 0.0]}})
    assert "camera.fovy_scale" not in report.applied


def test_project_point_center_and_axes():
    R = np.eye(3)
    fx, fy, cx, cy = camera_intrinsics(70.0, 320, 240)
    u, v, depth, inv = project_point([0, 0, -2], [0, 0, 0], R, 70.0, 320, 240)
    assert u == pytest.approx(cx) and v == pytest.approx(cy)
    assert depth == pytest.approx(2.0) and inv == 1
    # 相机 +x 方向 -> u 增大；+y 方向 -> v 减小（图像 y 向下）
    ur, _, _, _ = project_point([1, 0, -2], [0, 0, 0], R, 70.0, 320, 240)
    assert ur > cx
    _, vu, _, _ = project_point([0, 1, -2], [0, 0, 0], R, 70.0, 320, 240)
    assert vu < cy
    # 相机后方 -> 不可见
    _, _, _, invb = project_point([0, 0, 2], [0, 0, 0], R, 70.0, 320, 240)
    assert invb == 0


def test_project_point_fovy_scales_toward_center():
    R = np.eye(3)
    _, _, cx, _ = camera_intrinsics(70.0, 320, 240)
    u_narrow, _, _, _ = project_point([1, 0, -2], [0, 0, 0], R, 70.0, 320, 240)
    u_wide, _, _, _ = project_point([1, 0, -2], [0, 0, 0], R, 120.0, 320, 240)
    assert (u_wide - cx) < (u_narrow - cx)


def test_apply_sample_with_camera_and_rendering_serializable():
    applied = apply_sample(
        _cam_light_model(),
        {"params": {
            "camera": {"fovy_scale": 1.05, "pos_noise_m": [0.001, 0.0, 0.0]},
            "rendering": {"light_intensity_scale": 1.1, "texture_swap_prob": 0.0},
        }},
        dt_ctrl=0.02, rng=np.random.default_rng(3),
    )
    d = applied.as_dict()
    assert "camera.fovy_scale" in d["model"]["applied"]
    assert "camera.pos_noise_m" in d["model"]["applied"]
    assert "rendering.light_intensity_scale" in d["model"]["applied"]
    json.dumps(d)  # 必须可 JSON 序列化


# ── 逐 body 质量随机化 ──────────────────────────────────────────────────────
def _per_body_mass_plan(tmp_path):
    return _plan_from(tmp_path, """
        version: 1
        seed: 5
        groups:
          dynamics:
            enabled: true
            params:
              mass_scale:
                type: uniform
                low: 0.8
                high: 1.2
                scope: robot_bodies
                per_body: true
    """)


def test_per_body_mass_sampling_and_shape(tmp_path):
    plan = _per_body_mass_plan(tmp_path)
    shapes = per_body_shapes(plan, 3)
    assert shapes == {"dynamics.mass_scale": (3,)}
    a = sample(plan, "s", 0, shapes=shapes)["params"]["dynamics"]["mass_scale"]
    b = sample(plan, "s", 0, shapes=shapes)["params"]["dynamics"]["mass_scale"]
    assert np.asarray(a).shape == (3,)
    assert a == b
    assert all(0.8 <= float(v) <= 1.2 for v in a)


def test_per_body_mass_applied_per_body():
    model = mujoco.MjModel.from_xml_string(TWO_BODY_XML)
    ref = mujoco.MjModel.from_xml_string(TWO_BODY_XML)
    a = _body_id(model, "LINK_A")
    b = _body_id(model, "LINK_B")
    apply_model(model, {"dynamics": {"mass_scale": [1.2, 0.8]}})
    assert model.body_mass[a] == pytest.approx(ref.body_mass[a] * 1.2)
    assert model.body_mass[b] == pytest.approx(ref.body_mass[b] * 0.8)
    np.testing.assert_allclose(model.body_inertia[a], ref.body_inertia[a] * 1.2)
    np.testing.assert_allclose(model.body_inertia[b], ref.body_inertia[b] * 0.8)


def test_scalar_mass_scale_still_global():
    model = mujoco.MjModel.from_xml_string(TWO_BODY_XML)
    ref = mujoco.MjModel.from_xml_string(TWO_BODY_XML)
    a = _body_id(model, "LINK_A")
    b = _body_id(model, "LINK_B")
    apply_model(model, {"dynamics": {"mass_scale": 1.5}})
    assert model.body_mass[a] == pytest.approx(ref.body_mass[a] * 1.5)
    assert model.body_mass[b] == pytest.approx(ref.body_mass[b] * 1.5)


def test_per_body_mass_shape_mismatch_rejected():
    model = mujoco.MjModel.from_xml_string(TWO_BODY_XML)  # 2 个 robot body
    with pytest.raises(ApplyError, match="需要标量或"):
        apply_model(model, {"dynamics": {"mass_scale": [1.0, 1.0, 1.0]}})


def test_per_body_mass_nonpositive_rejected():
    model = mujoco.MjModel.from_xml_string(TWO_BODY_XML)
    with pytest.raises(ApplyError, match="必须为正"):
        apply_model(model, {"dynamics": {"mass_scale": [1.0, 0.0]}})


def test_manifest_summarizes_per_body_mass(tmp_path):
    """逐 body 质量应汇总成单列（均值），summary 不能因列表报错。"""
    import csv
    from t800_sim.dataset import build_manifest

    session = tmp_path / "walk/seed_0001/ep_0000"
    session.mkdir(parents=True)
    meta = {
        "schema_version": "1.0", "task": "walk", "scene_id": "empty",
        "seed": 1, "episode": 0, "frames": 0, "duration": 0.0,
        "cmd": [0.6, 0.0, 0.0], "cameras": [], "metrics": {},
        "randomization": {
            "enabled": True,
            "plan": {"plan_hash": "h", "session": "s"},
            "params": {"dynamics": {"mass_scale": [1.1, 0.9, 1.0]}},
        },
    }
    (session / "meta.json").write_text(json.dumps(meta))
    summary = build_manifest(tmp_path, val_ratio=0.0)
    assert summary["randomized_sessions"] == 1
    rows = list(csv.DictReader((tmp_path / "manifest.csv").open()))
    assert float(rows[0]["mass_scale"]) == pytest.approx(1.0)
