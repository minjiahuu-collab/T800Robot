"""多路相机注入与逐帧落盘（数据资产 schema v1）。

头部、左右手腕各挂一台相机：通过 MjSpec 在运行时注入（不改 SDK 文件）。
MultiCamRecorder 按固定帧率渲染每台相机并写成 PNG，同时落盘四份按 frame
对齐的记录：

    states.csv        本体状态（仿真时间、墙钟时间、指令、控制量、基座位姿、关节角、足端力）
    cameras.csv       每台相机每帧的世界位姿
    ground_truth.csv  物体位姿 + 左右末端位姿（自动真值）
    projections.csv   每个相机对每个真值目标的 2D 像素投影（u,v,depth,in_view）
    meta.json         场景 / seed / 策略哈希 / 指标等元信息

目录结构（一次采集 = 一个 session）:
    <out>/
        meta.json
        states.csv
        cameras.csv
        ground_truth.csv
        frames/head/000000.png
        frames/wrist_left/000000.png
        frames/wrist_right/000000.png
"""

import csv
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import mujoco
import numpy as np
from PIL import Image

from .credibility.environment import environment_fingerprint, git_commit, sha256_file

HEAD_BODY = "LINK_HEAD_YAW"
WRIST_L_BODY = "LINK_WRIST_END_L"
WRIST_R_BODY = "LINK_WRIST_END_R"
EE_BODIES = (WRIST_L_BODY, WRIST_R_BODY)

# (相机名, 挂载 body, 相对位置, 相机 x/y 轴)
# MuJoCo 相机沿自身 -z 看，+x 向右、+y 向上。
CAMERAS = (
    ("head", HEAD_BODY, (0.09, 0.0, 0.10), (0, -1, 0, 0, 0, 1)),           # 朝正前方
    ("wrist_left", WRIST_L_BODY, (0.0, 0.0, -0.02), (1, 0, 0, 0, 1, 0)),  # 沿前臂向前
    ("wrist_right", WRIST_R_BODY, (0.0, 0.0, -0.02), (1, 0, 0, 0, 1, 0)),
)

CAMERA_NAMES = tuple(name for name, *_ in CAMERAS)
# 模型自带的足端力传感器（t800.xml -> serial_sensors.xml）
FORCE_SENSORS = ("force_left_foot", "force_right_foot")
FOVY = 70.0


def available_force_sensors(model) -> tuple:
    """返回模型中真实存在的足端力传感器名。"""
    return tuple(
        name for name in FORCE_SENSORS
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name) >= 0
    )
SCHEMA_VERSION = "1.0"

# 跌倒判据（与 batch.py 保持一致）
FELL_HEIGHT = 0.6
FELL_ANGLE_DEG = 45.0



def _latency_frames(latency_ms: float, fps: float) -> int:
    """把链路延迟（ms）换算成相机帧数；非零延迟至少占 1 帧。"""
    ms = float(latency_ms)
    if ms <= 0.0:
        return 0
    frames = ms / 1000.0 * float(fps)
    return max(1, int(math.ceil(frames - 1e-9)))


def camera_intrinsics(fovy_deg: float, width: int, height: int):
    """由垂直视场角与图像尺寸得到针孔内参 (fx, fy, cx, cy)。

    MuJoCo 相机像素为正方形，fx=fy；主点位于图像中心。
    """
    fovy = math.radians(float(fovy_deg))
    fy = (float(height) / 2.0) / math.tan(fovy / 2.0)
    fx = fy
    return fx, fy, float(width) / 2.0, float(height) / 2.0


def project_point(point_world, cam_pos, cam_mat, fovy_deg, width, height):
    """世界点 -> 像素坐标 (u, v, depth, in_view)。

    cam_mat 为相机 3x3 姿态（列向量是相机 x/y/z 轴在世界系下的方向），
    世界->相机：p_cam = R^T (p - pos)，相机沿 -z 看。图像 y 向下，故 v 翻转。
    点在相机后方或过近时返回 (nan, nan, depth, 0)。
    """
    rel = np.asarray(point_world, dtype=float) - np.asarray(cam_pos, dtype=float)
    rot = np.asarray(cam_mat, dtype=float).reshape(3, 3)
    p_cam = rot.T @ rel
    depth = -float(p_cam[2])
    fx, fy, cx, cy = camera_intrinsics(fovy_deg, width, height)
    if not math.isfinite(depth) or depth <= 1e-6:
        return float("nan"), float("nan"), depth, 0
    u = cx + fx * (p_cam[0] / depth)
    v = cy - fy * (p_cam[1] / depth)
    in_view = int(0.0 <= u < width and 0.0 <= v < height)
    return float(u), float(v), depth, in_view


@dataclass
class _FramePacket:
    """一次渲染产出的整帧数据：图像 + 与之对齐的全部标签。

    延迟缓冲里存的是这个对象的引用（数组已 copy），从而投递时读到的是
    渲染时刻的快照，而不是被后续仿真步进覆盖后的当前值。
    """

    images: Dict[str, Image.Image]
    cam_poses: List[Tuple[str, np.ndarray, np.ndarray]]
    obj: list
    ee_l: list
    ee_r: list
    sim_time: float
    render_time: float
    cmd: tuple
    base: list
    forces: list
    ctrl: list
    joints: list
    projections: list


class CameraLink:
    """对相机帧流施加丢帧（重复上一帧）与固定延迟。

    语义与 SensorModel.observe 完全一致，保证视觉观测通道与本体观测通道
    服从同一套噪声模型：
      - latency_frames: 纯移位，前 latency 帧无延迟（缓冲区未填满时投递当前帧），
        之后投递 latency 帧之前渲染的画面。
      - dropout_prob: 以该概率丢弃本帧，改为重复上一次成功投递的帧；
        第一帧没有上一帧可重复，永远不会被丢。
    丢帧决策是 per-capture-tick 全局的（所有相机同步），保持帧索引对齐。
    """

    def __init__(self, dropout_prob: float, latency_frames: int,
                 rng: np.random.Generator):
        self.dropout_prob = float(dropout_prob)
        self.latency_frames = max(0, int(latency_frames))
        self.rng = rng
        self._buf: Deque = deque()
        self._last: Optional[_FramePacket] = None

    def step(self, packet):
        # 延迟：先入队，队列超过 latency 时取出最旧的一帧投递。
        self._buf.append(packet)
        if len(self._buf) > self.latency_frames:
            delivered = self._buf.popleft()
        else:
            delivered = packet

        # 丢帧：命中则重复上一次投递的帧。
        dropped = False
        if (
            self.dropout_prob
            and self._last is not None
            and self.rng.random() < self.dropout_prob
        ):
            delivered = self._last
            dropped = True
        else:
            self._last = delivered
        return delivered, dropped


def _quat_from_xyaxes(xyaxes) -> np.ndarray:
    """把 XML 的 xyaxes 换成 MjsCamera 支持的 quat（相机 z = x × y）。"""
    x = np.asarray(xyaxes[:3], dtype=float)
    y = np.asarray(xyaxes[3:], dtype=float)
    mat = np.stack([x, y, np.cross(x, y)], axis=1)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.ravel())
    return quat


def add_cameras(spec: mujoco.MjSpec) -> None:
    """MjSpec hook：在头部与左右手腕注入三台相机。"""
    for name, body_name, pos, xyaxes in CAMERAS:
        cam = spec.body(body_name).add_camera()
        cam.name = name
        cam.pos = list(pos)
        cam.quat = _quat_from_xyaxes(xyaxes)
        cam.fovy = FOVY


def _mat_to_quat(mat9) -> np.ndarray:
    """3x3 旋转矩阵（行主序 9 元素）转四元数 w,x,y,z。"""
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(mat9, dtype=float).ravel())
    return quat


class MultiCamRecorder:
    """按固定帧率渲染多路相机并落盘，图像与状态用 frame 对齐。"""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        out_dir,
        *,
        camera_names=CAMERA_NAMES,
        joint_names=(),
        joint_qpos_idx=(),
        width: int = 320,
        height: int = 240,
        fps: float = 30.0,
        task: str = "record",
        scene_id: str = "empty",
        seed: int = 0,
        policy=None,
        gt_object=None,
        ee_bodies=EE_BODIES,
        force_sensors=(),
        force_noise=None,
        success_vx_ratio: float = 0.5,
        extra_meta=None,
        camera_dropout_prob: float = 0.0,
        camera_latency_ms: float = 0.0,
        camera_rng=None,
    ):
        self.data = data
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.camera_names = tuple(camera_names)
        self.fps = float(fps)
        # 相机链路丢帧/延迟：与策略本体观测共用同一份采样参数，
        # 但用独立 RNG 流（采集帧率 != 控制帧率），保证可复现。
        self._dropped_frames = 0
        if camera_rng is not None and (camera_dropout_prob or camera_latency_ms):
            self._cam_link: Optional[CameraLink] = CameraLink(
                float(camera_dropout_prob),
                _latency_frames(camera_latency_ms, self.fps),
                camera_rng,
            )
            self._cam_link_meta = {
                "dropout_prob": float(camera_dropout_prob),
                "latency_ms": float(camera_latency_ms),
                "latency_frames": _latency_frames(camera_latency_ms, self.fps),
            }
        else:
            self._cam_link = None
            self._cam_link_meta = {
                "dropout_prob": 0.0,
                "latency_ms": 0.0,
                "latency_frames": 0,
            }
        self.width = int(width)
        self.height = int(height)
        self.frame = 0
        self.next_t = float(data.time)
        self.cmd = (0.0, 0.0, 0.0)
        self._joint_names = list(joint_names)
        self._joint_qpos_idx = np.asarray(joint_qpos_idx, dtype=int)
        self._success_vx_ratio = float(success_vx_ratio)

        self._cam_ids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            for name in self.camera_names
        ], dtype=int)
        if np.any(self._cam_ids < 0):
            missing = [n for n, i in zip(self.camera_names, self._cam_ids) if i < 0]
            raise RuntimeError(f"相机注入失败: {', '.join(missing)}")

        self._ee_bids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ee_bodies
        ], dtype=int)
        self._ee_names = tuple(ee_bodies)
        # 目标与每台相机的实际 fovy（可能已被 camera.fovy_scale 扰动）。
        self._has_obj = gt_object is not None
        self._targets = (["obj"] if self._has_obj else []) + ["ee_l", "ee_r"]
        self._cam_fovy = {
            name: float(model.cam_fovy[cid])
            for name, cid in zip(self.camera_names, self._cam_ids)
        }

        # force_noise：可选回调 (name, values) -> values，用于把传感器噪声
        # 加到落盘的足端力读数上；不传则记录干净读数。
        self._force_cols = []
        for name in tuple(force_sensors):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid < 0:
                raise RuntimeError(f"找不到力传感器: {name}")
            if int(model.sensor_dim[sid]) != 3:
                raise RuntimeError(
                    f"力传感器 {name} 维度应为 3，实际 {int(model.sensor_dim[sid])}"
                )
            self._force_cols.append((name, int(model.sensor_adr[sid])))
        self._force_noise = force_noise

        self._obj_bid = None
        self._obj_qadr = None
        if gt_object:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, gt_object)
            if bid < 0:
                raise RuntimeError(f"找不到真值物体: {gt_object}")
            self._obj_bid = bid
            if model.body_jntnum[bid] > 0:
                jid = model.body_jntadr[bid]
                if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                    self._obj_qadr = int(model.jnt_qposadr[jid])

        self._base_qadr = int(model.jnt_qposadr[0])
        self._t0 = float(data.time)
        self._x0 = float(data.qpos[self._base_qadr])
        self._h_min = float(data.qpos[self._base_qadr + 2])
        self._roll_max = 0.0
        self._pitch_max = 0.0

        self.renderer = mujoco.Renderer(model, self.height, self.width)
        self.frames_dir = self.out_dir / "frames"
        for name in self.camera_names:
            (self.frames_dir / name).mkdir(parents=True, exist_ok=True)

        self._states_file = open(self.out_dir / "states.csv", "w", newline="")
        self._states = csv.writer(self._states_file)
        self._states.writerow(
            ["frame", "time", "render_time", "vx_cmd", "vy_cmd", "yaw_cmd",
             "base_x", "base_y", "base_z", "qw", "qx", "qy", "qz",
             *(f"{name}_{axis}" for name, _ in self._force_cols
               for axis in ("x", "y", "z")),
             *(f"ctrl_{n}" for n in self._joint_names),
             *self._joint_names, "dropped"]
        )

        self._cam_file = open(self.out_dir / "cameras.csv", "w", newline="")
        self._cam_writer = csv.writer(self._cam_file)
        self._cam_writer.writerow(
            ["frame", "cam", "x", "y", "z",
             *(f"r{r}{c}" for r in range(3) for c in range(3)), "dropped"]
        )

        self._gt_file = open(self.out_dir / "ground_truth.csv", "w", newline="")
        self._gt_writer = csv.writer(self._gt_file)
        self._gt_writer.writerow(
            ["frame",
             "obj_x", "obj_y", "obj_z", "obj_qw", "obj_qx", "obj_qy", "obj_qz",
             *(f"ee_{side}_{axis}" for side in ("l", "r")
               for axis in ("x", "y", "z", "qw", "qx", "qy", "qz")), "dropped"]
        )

        self._proj_file = open(self.out_dir / "projections.csv", "w", newline="")
        self._proj_writer = csv.writer(self._proj_file)
        self._proj_writer.writerow(
            ["frame", "cam", "target", "u", "v", "depth", "in_view", "dropped"]
        )

        self.meta = {
            "schema_version": SCHEMA_VERSION,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": task,
            "scene_id": scene_id,
            "seed": int(seed),
            "policy": dict(policy or {}),
            "git_commit": git_commit(),
            "environment": environment_fingerprint(),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "timestep": float(model.opt.timestep),
            "cameras": [
                {"name": name, "body": body, "pos": list(pos),
                 "fovy": self._cam_fovy.get(name, FOVY)}
                for name, body, pos, _ in CAMERAS
                if name in self.camera_names
            ],
            "joints": self._joint_names,
            "force_sensors": [name for name, _ in self._force_cols],
            "gt_object": gt_object,
            "ee_bodies": list(self._ee_names),
            "projection": {"targets": list(self._targets), "format": "pixel_xy"},
            "frames": 0,
            "duration": 0.0,
            "success": False,
            "metrics": {},
            "camera_link": {
                **self._cam_link_meta,
                "dropped_frames": 0,
            },
        }
        if extra_meta:
            self.meta.update(extra_meta)
        self._write_meta()

    def set_command(self, cmd) -> None:
        self.cmd = tuple(float(c) for c in cmd)

    def due(self, t: float) -> bool:
        return t + 1e-9 >= self.next_t

    def capture_if_due(self, t: float) -> bool:
        """到点则渲染三路相机并写一行状态；返回本帧是否采集。"""
        if not self.due(t):
            return False
        self._capture()
        self.next_t += 1.0 / self.fps
        return True

    def _base_rpy_deg(self):
        quat = self.data.qpos[self._base_qadr + 3:self._base_qadr + 7]
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        rot = mat.reshape(3, 3)
        pitch = math.asin(max(-1.0, min(1.0, -rot[2, 0])))
        roll = math.atan2(rot[2, 1], rot[2, 2])
        return math.degrees(roll), math.degrees(pitch)

    def _object_pose(self):
        if self._obj_bid is None:
            return [float("nan")] * 7
        if self._obj_qadr is not None:
            q = self.data.qpos[self._obj_qadr:self._obj_qadr + 7]
            return [float(v) for v in q]
        return [*self.data.xpos[self._obj_bid], *_mat_to_quat(self.data.xmat[self._obj_bid])]

    def _ee_pose(self, bid: int):
        return [*self.data.xpos[bid], *_mat_to_quat(self.data.xmat[bid])]

    def _read_force(self, name: str, adr: int) -> np.ndarray:
        value = np.asarray(self.data.sensordata[adr:adr + 3], dtype=float)
        if self._force_noise is not None:
            value = np.asarray(self._force_noise(name, value), dtype=float)
        return value

    def _render_packet(self) -> _FramePacket:
        """渲染当前状态并快照全部标签；数组必须 copy，避免被后续步进覆盖。"""
        render_time = time.time()
        images: Dict[str, Image.Image] = {}
        cam_poses: List[Tuple[str, np.ndarray, np.ndarray]] = []
        for name, cid in zip(self.camera_names, self._cam_ids):
            self.renderer.update_scene(self.data, camera=name)
            images[name] = Image.fromarray(self.renderer.render())
            cam_poses.append((
                name,
                np.asarray(self.data.cam_xpos[cid], dtype=float).copy(),
                np.asarray(self.data.cam_xmat[cid], dtype=float).copy(),
            ))
        obj = self._object_pose()
        ee_l = self._ee_pose(int(self._ee_bids[0]))
        ee_r = self._ee_pose(int(self._ee_bids[1]))
        q = self.data.qpos
        forces: list = []
        for fname, adr in self._force_cols:
            forces.extend(float(v) for v in self._read_force(fname, adr))
        projections: list = []
        for cname, xpos, xmat in cam_poses:
            fovy = self._cam_fovy[cname]
            for tname, pose in (("obj", obj), ("ee_l", ee_l), ("ee_r", ee_r)):
                if tname == "obj" and not self._has_obj:
                    continue
                u, v, depth, inv = project_point(
                    pose[:3], xpos, xmat, fovy, self.width, self.height
                )
                projections.append((cname, tname, u, v, depth, inv))
        return _FramePacket(
            images=images,
            cam_poses=cam_poses,
            obj=obj,
            ee_l=ee_l,
            ee_r=ee_r,
            sim_time=float(self.data.time),
            render_time=render_time,
            cmd=tuple(self.cmd),
            base=[float(q[self._base_qadr + i]) for i in range(7)],
            forces=forces,
            ctrl=[float(v) for v in self.data.ctrl[:len(self._joint_names)]],
            joints=[float(v) for v in q[self._joint_qpos_idx]],
            projections=projections,
        )

    def _write_packet(self, pkt: _FramePacket, dropped: bool) -> None:
        """把投递帧（可能是延迟/重复后的）落盘；标签与图像严格对齐。"""
        for name in self.camera_names:
            pkt.images[name].save(self.frames_dir / name / f"{self.frame:06d}.png")

        for name, xpos, xmat in pkt.cam_poses:
            self._cam_writer.writerow(
                [self.frame, name,
                 *(f"{v:.6f}" for v in xpos),
                 *(f"{v:.6f}" for v in xmat),
                 int(dropped)]
            )

        self._gt_writer.writerow(
            [self.frame,
             *(f"{v:.6f}" for v in pkt.obj),
             *(f"{v:.6f}" for v in pkt.ee_l),
             *(f"{v:.6f}" for v in pkt.ee_r),
             int(dropped)]
        )

        for cname, tname, u, v, depth, inv in pkt.projections:
            self._proj_writer.writerow(
                [self.frame, cname, tname,
                 f"{u:.3f}", f"{v:.3f}", f"{depth:.6f}", int(inv), int(dropped)]
            )

        self._states.writerow(
            [self.frame, f"{pkt.sim_time:.6f}", f"{pkt.render_time:.6f}", *pkt.cmd,
             *(f"{v:.6f}" for v in pkt.base),
             *(f"{v:.6f}" for v in pkt.forces),
             *(f"{v:.6f}" for v in pkt.ctrl),
             *(f"{v:.6f}" for v in pkt.joints),
             int(dropped)]
        )

        self._cam_file.flush()
        self._gt_file.flush()
        self._proj_file.flush()
        self._states_file.flush()

    def _capture(self) -> None:
        # 指标用真实当前状态，不受相机链路延迟/丢帧影响。
        q = self.data.qpos
        self._h_min = min(self._h_min, float(q[self._base_qadr + 2]))
        roll, pitch = self._base_rpy_deg()
        self._roll_max = max(self._roll_max, abs(roll))
        self._pitch_max = max(self._pitch_max, abs(pitch))

        packet = self._render_packet()
        if self._cam_link is not None:
            delivered, dropped = self._cam_link.step(packet)
        else:
            delivered, dropped = packet, False
        if dropped:
            self._dropped_frames += 1

        self._write_packet(delivered, dropped)
        self.frame += 1

    def _metrics(self) -> dict:
        duration = float(self.data.time - self._t0)
        final_dx = float(self.data.qpos[self._base_qadr] - self._x0)
        mean_vx = final_dx / duration if duration > 0 else 0.0
        fell = bool(
            self._h_min < FELL_HEIGHT
            or self._roll_max > FELL_ANGLE_DEG
            or self._pitch_max > FELL_ANGLE_DEG
        )
        cmd_vx = float(self.cmd[0])
        success = (not fell) and (
            cmd_vx <= 0.0 or mean_vx >= self._success_vx_ratio * cmd_vx
        )
        return {
            "duration": duration,
            "final_dx": final_dx,
            "mean_vx": mean_vx,
            "h_min": self._h_min,
            "roll_max": self._roll_max,
            "pitch_max": self._pitch_max,
            "fell": fell,
            "success": success,
        }

    def close(self) -> Path:
        self.renderer.close()
        for f in (self._states_file, self._cam_file, self._gt_file, self._proj_file):
            f.close()
        metrics = self._metrics()
        self.meta["frames"] = self.frame
        self.meta["duration"] = metrics["duration"]
        self.meta["success"] = metrics["success"]
        self.meta["camera_link"]["dropped_frames"] = self._dropped_frames
        self.meta["metrics"] = metrics
        self._write_meta()
        return self.out_dir

    def _write_meta(self) -> None:
        (self.out_dir / "meta.json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2)
        )
