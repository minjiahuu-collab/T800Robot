"""MuJoCo 被动 viewer 后端。

封装三件每个脚本都要重复的事：
  1. launch_passive / close 的生命周期
  2. 每帧恢复 opt.flags 与 geomgroup，抵消 viewer 快捷键（如 W 切线框）的副作用
  3. key_callback -> 队列，主循环里用 poll_key() 取，避免回调线程直接改仿真状态
"""

import collections
from typing import Callable, Optional

import mujoco


class ViewerBackend:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        key_callback: Optional[Callable[[int], None]] = None,
    ):
        self._model = model
        self._data = data
        self._user_key_cb = key_callback
        self._keys = collections.deque()
        self._viewer = None
        self._flags0 = None
        self._groups0 = None

    def __enter__(self) -> "ViewerBackend":
        import mujoco.viewer

        self._viewer = mujoco.viewer.launch_passive(
            self._model, self._data, key_callback=self._on_key
        )
        self._flags0 = self._viewer.opt.flags.copy()
        self._groups0 = self._viewer.opt.geomgroup.copy()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._viewer.close()

    def _on_key(self, keycode: int) -> None:
        self._keys.append(keycode)
        if self._user_key_cb is not None:
            self._user_key_cb(keycode)

    def poll_key(self) -> Optional[int]:
        return self._keys.popleft() if self._keys else None

    def is_running(self) -> bool:
        return self._viewer.is_running()

    def begin_frame(self) -> None:
        self._viewer.opt.flags[:] = self._flags0
        self._viewer.opt.geomgroup[:] = self._groups0

    def sync(self) -> None:
        self._viewer.sync()

    def set_texts(self, texts) -> None:
        self._viewer.set_texts(texts)

    def set_camera(self, distance: float, azimuth: float, elevation: float, lookat) -> None:
        cam = self._viewer.cam
        cam.distance = distance
        cam.azimuth = azimuth
        cam.elevation = elevation
        cam.lookat[:] = lookat
