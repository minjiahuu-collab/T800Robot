"""固定步长主循环与日志节流。

后端只需要实现 is_running / begin_frame / sync：
  - ViewerBackend（viewer.py）：真实窗口
  - 后续批量仿真可提供 HeadlessBackend（is_running 由步数控制，sync 为空）
"""

import time
from typing import Callable, Protocol


class LoopBackend(Protocol):
    def is_running(self) -> bool: ...
    def begin_frame(self) -> None: ...
    def sync(self) -> None: ...


class LogThrottle:
    """按仿真时间均匀触发日志，避免逐帧打印。"""

    def __init__(self, dt: float, start: float = 0.0):
        self.dt = dt
        self.next_t = start

    def due(self, t: float) -> bool:
        if t + 1e-9 < self.next_t:
            return False
        self.next_t += self.dt
        return True


def run(step: Callable[[LoopBackend], None], backend: LoopBackend, dt_ctrl: float) -> None:
    """begin_frame -> step -> sync -> 补齐实时节拍。step 内负责物理步进。"""
    while backend.is_running():
        t0 = time.time()
        backend.begin_frame()
        step(backend)
        backend.sync()
        remain = dt_ctrl - (time.time() - t0)
        if remain > 0:
            time.sleep(remain)
