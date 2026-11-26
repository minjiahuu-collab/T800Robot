"""多路相机注入与逐帧落盘（数据资产 schema v1）。

头部、左右手腕各挂一台相机：通过 MjSpec 在运行时注入（不改 SDK 文件）。
MultiCamRecorder 按固定帧率渲染每台相机并写成 PNG，同时落盘四份按 frame
对齐的记录：

    states.csv        本体状态（仿真时间、墙钟时间、指令、控制量、基座位姿、关节角）
    cameras.csv       每台相机每帧的世界位姿
    ground_truth.csv  物体位姿 + 左右末端位姿（自动真值）
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
from pathlib import Path

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
FOVY = 70.0
SCHEMA_VERSION = "1.0"

# 跌倒判据（与 batch.py 保持一致）
FELL_HEIGHT = 0.6
FELL_ANGLE_DEG = 45.0


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
        success_vx_ratio: float = 0.5,
        extra_meta=None,
    ):
        self.data = data
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.camera_names = tuple(camera_names)
        self.fps = float(fps)
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
             *(f"ctrl_{n}" for n in self._joint_names),
             *self._joint_names]
        )

        self._cam_file = open(self.out_dir / "cameras.csv", "w", newline="")
        self._cam_writer = csv.writer(self._cam_file)
        self._cam_writer.writerow(
            ["frame", "cam", "x", "y", "z",
             *(f"r{r}{c}" for r in range(3) for c in range(3))]
        )

        self._gt_file = open(self.out_dir / "ground_truth.csv", "w", newline="")
        self._gt_writer = csv.writer(self._gt_file)
        self._gt_writer.writerow(
            ["frame",
             "obj_x", "obj_y", "obj_z", "obj_qw", "obj_qx", "obj_qy", "obj_qz",
             *(f"ee_{side}_{axis}" for side in ("l", "r")
               for axis in ("x", "y", "z", "qw", "qx", "qy", "qz"))]
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
                {"name": name, "body": body, "pos": list(pos), "fovy": FOVY}
                for name, body, pos, _ in CAMERAS
                if name in self.camera_names
            ],
            "joints": self._joint_names,
            "gt_object": gt_object,
            "ee_bodies": list(self._ee_names),
            "frames": 0,
            "duration": 0.0,
            "success": False,
            "metrics": {},
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

    def _capture(self) -> None:
        render_time = time.time()
        for name in self.camera_names:
            self.renderer.update_scene(self.data, camera=name)
            image = Image.fromarray(self.renderer.render())
            image.save(self.frames_dir / name / f"{self.frame:06d}.png")

        for name, cid in zip(self.camera_names, self._cam_ids):
            xmat = self.data.cam_xmat[cid]
            self._cam_writer.writerow(
                [self.frame, name,
                 *(f"{v:.6f}" for v in self.data.cam_xpos[cid]),
                 *(f"{v:.6f}" for v in xmat)]
            )

        obj = self._object_pose()
        ee_l = self._ee_pose(int(self._ee_bids[0]))
        ee_r = self._ee_pose(int(self._ee_bids[1]))
        self._gt_writer.writerow(
            [self.frame,
             *(f"{v:.6f}" for v in obj),
             *(f"{v:.6f}" for v in ee_l),
             *(f"{v:.6f}" for v in ee_r)]
        )

        q = self.data.qpos
        self._states.writerow(
            [self.frame, f"{self.data.time:.6f}", f"{render_time:.6f}", *self.cmd,
             f"{q[self._base_qadr]:.6f}", f"{q[self._base_qadr + 1]:.6f}",
             f"{q[self._base_qadr + 2]:.6f}",
             f"{q[self._base_qadr + 3]:.6f}", f"{q[self._base_qadr + 4]:.6f}",
             f"{q[self._base_qadr + 5]:.6f}", f"{q[self._base_qadr + 6]:.6f}",
             *(f"{v:.6f}" for v in self.data.ctrl[:len(self._joint_names)]),
             *(f"{v:.6f}" for v in q[self._joint_qpos_idx])]
        )

        self._cam_file.flush()
        self._gt_file.flush()
        self._states_file.flush()

        self._h_min = min(self._h_min, float(q[self._base_qadr + 2]))
        roll, pitch = self._base_rpy_deg()
        self._roll_max = max(self._roll_max, abs(roll))
        self._pitch_max = max(self._pitch_max, abs(pitch))
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
        for f in (self._states_file, self._cam_file, self._gt_file):
            f.close()
        metrics = self._metrics()
        self.meta["frames"] = self.frame
        self.meta["duration"] = metrics["duration"]
        self.meta["success"] = metrics["success"]
        self.meta["metrics"] = metrics
        self._write_meta()
        return self.out_dir

    def _write_meta(self) -> None:
        (self.out_dir / "meta.json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2)
        )
