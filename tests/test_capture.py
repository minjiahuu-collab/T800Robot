"""CameraLink（相机链路丢帧/延迟）单元测试。

CameraLink 是纯逻辑，不依赖 GL/渲染，可直接用任意 packet 验证语义。
与 SensorModel.observe 的丢帧/延迟语义保持一致：
  - latency_frames 帧纯移位，前 latency 帧无延迟；
  - dropout 命中则重复上一次投递帧，第一帧永不丢。
"""

import numpy as np
import pytest

from t800_sim.capture import CameraLink, _latency_frames


def test_latency_frames_conversion():
    assert _latency_frames(0, 30) == 0
    assert _latency_frames(-1, 30) == 0          # 非正视为 0
    assert _latency_frames(10, 30) == 1          # 10 ms @ 30 fps -> 0.3 帧，至少 1
    assert _latency_frames(20, 30) == 1          # 20 ms -> 0.6 帧
    assert _latency_frames(34, 30) == 2          # 34 ms -> 1.02 帧 -> 2
    assert _latency_frames(100, 10) == 1         # 100 ms @ 10 fps -> 1 帧
    assert _latency_frames(200, 10) == 2


def test_link_noop_when_disabled():
    link = CameraLink(0.0, 0, np.random.default_rng(0))
    assert link.step("A") == ("A", False)
    assert link.step("B") == ("B", False)
    assert link.step("C") == ("C", False)


def test_first_frame_never_dropped():
    # dropout=1.0：第一帧没有上一帧可重复，必须成功投递。
    link = CameraLink(1.0, 0, np.random.default_rng(0))
    delivered, dropped = link.step("first")
    assert dropped is False
    assert delivered == "first"


def test_dropout_repeats_last_delivered():
    link = CameraLink(1.0, 0, np.random.default_rng(0))
    link.step("A")                       # 第 0 帧：成功，_last = A
    delivered, dropped = link.step("B")  # 命中丢帧 -> 重复 A
    assert dropped is True
    assert delivered == "A"
    delivered, dropped = link.step("C")  # 继续重复 A
    assert dropped is True
    assert delivered == "A"


def test_dropout_partial_probability_is_bounded():
    # dropout=0.5：丢帧比例应接近 0.5，且第一帧永不丢。
    n = 2000
    link = CameraLink(0.5, 0, np.random.default_rng(123))
    drops = 0
    for i in range(n):
        _, dropped = link.step(i)
        drops += int(dropped)
    assert drops > 0
    # 第一帧不丢，所以最多丢 n-1 帧；比例落在合理区间。
    assert 0.35 < drops / (n - 1) < 0.65


def test_latency_shifts_stream():
    # latency=2：前 2 帧投递当前，之后投递 2 帧前的画面。
    link = CameraLink(0.0, 2, np.random.default_rng(0))
    seq = ["A", "B", "C", "D", "E", "F", "G"]
    out = [link.step(p)[0] for p in seq]
    assert out == ["A", "B", "A", "B", "C", "D", "E"]


def test_latency_zero_identity():
    link = CameraLink(0.0, 0, np.random.default_rng(0))
    out = [link.step(p)[0] for p in ["A", "B", "C"]]
    assert out == ["A", "B", "C"]


def test_combined_dropout_and_latency():
    # 延迟 1 帧 + 必丢：投递序列应为 [A(当前), A(延迟后本应投A但丢->重复A), A, ...]
    link = CameraLink(1.0, 1, np.random.default_rng(0))
    seq = ["A", "B", "C", "D"]
    out = []
    for p in seq:
        delivered, _ = link.step(p)
        out.append(delivered)
    # 第0帧: buf=[A], len1不>1 -> 投A, _last=A
    # 第1帧: buf=[A,B], len2>1 -> pop A, 候选A; 丢->重复_last=A -> 投A
    # 第2帧: buf=[B,C] -> pop B, 候选B; 丢->重复A -> 投A
    # 第3帧: buf=[C,D] -> pop C, 候选C; 丢->重复A -> 投A
    assert out == ["A", "A", "A", "A"]


def test_link_is_reproducible():
    def run():
        link = CameraLink(0.5, 1, np.random.default_rng(42))
        return [(link.step(i)[0], link.step(i)[1]) for i in range(30)]
    assert run() == run()


def test_link_does_not_mutate_packet_identity():
    # 同一个 packet 引用被重复投递时，不应被复制或改写（写入方只读取）。
    link = CameraLink(1.0, 0, np.random.default_rng(0))
    pkt = object()
    first, _ = link.step(pkt)
    second, dropped = link.step(object())
    assert first is pkt
    assert second is pkt          # 丢帧后返回的是同一对象引用
    assert dropped is True
