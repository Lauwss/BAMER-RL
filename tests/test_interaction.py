# -*- coding: utf-8 -*-
"""交互协议单测（M1.1 验收）：上报/图像下发/到达事件按规则触发。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bamer_env import Maze, build_scene
from bamer_env.interaction import (
    EVENT_ARRIVED,
    EVENT_COLLISION,
    EVENT_IMAGE_FEED,
    EVENT_REPORTED,
    EVENT_TASK_COMPLETE,
    EVENT_TIMEOUT,
    InteractionManager,
    image_asset,
)


def empty_maze(n: int = 3) -> Maze:
    """无内墙迷宫（外边框由 Maze 构造自动封闭）。"""
    return Maze([[False] * n for _ in range(n)], [[False] * n for _ in range(n)])


def make_manager():
    scene = build_scene(
        maze=empty_maze(3),
        state_machines={"quyuan": (1, 1), "zhaojun": (2, 2)},
        destinations={"chuguo": (0, 2), "xihan": (2, 0)},
        start=(0, 0),
    )
    return InteractionManager(scene)


def _by_type(events, etype):
    return [e for e in events if e.type == etype]


def test_report():
    """每步实时上报自身格坐标（题目规则）。"""
    m = make_manager()
    ev = m.report(7, (3, 4))
    assert ev.type == EVENT_REPORTED and ev.step == 7
    assert ev.payload == {"x": 3, "y": 4}
    print("[上报] OK：Reported 事件携带自身格坐标")


def test_image_feed_on_entry():
    """进入状态机格下发图像（first 标记识别）；停留不重复下发；再进入重新下发。"""
    m = make_manager()
    # 从 (0,1) 走进屈原格 (1,1)
    evs = m.enter_events(5, (1, 1), (0, 1))
    feeds = _by_type(evs, EVENT_IMAGE_FEED)
    assert len(feeds) == 1 and feeds[0].payload["machine"] == "quyuan"
    assert feeds[0].payload["first"] is True
    assert m.identified == {"quyuan"}
    # 同格停留（原地转身）不下发
    assert _by_type(m.enter_events(6, (1, 1), (1, 1)), EVENT_IMAGE_FEED) == []
    # 离开后再次进入重新下发（"经过即反馈"），first=False
    evs = m.enter_events(9, (1, 1), (2, 1))
    feeds = _by_type(evs, EVENT_IMAGE_FEED)
    assert len(feeds) == 1 and feeds[0].payload["first"] is False
    print("[图像下发] OK：进入即下发 / 停留不重复 / 再进入重新下发")


def test_delivery_requires_identification():
    """识别前经过终点格不派送；识别后到达映射终点才 Arrived；双派送达成 TaskComplete 仅一次。"""
    m = make_manager()
    # 未识别时进入楚国 (0,2)：无任何派送
    assert m.enter_events(2, (0, 2), (0, 1)) == []
    assert not m.task_complete
    # 识别屈原后到达楚国 → Arrived(quyuan→chuguo)
    m.enter_events(3, (1, 1), (0, 1))
    evs = m.enter_events(6, (0, 2), (1, 2))
    arrived = _by_type(evs, EVENT_ARRIVED)
    assert len(arrived) == 1
    assert arrived[0].payload == {"machine": "quyuan", "destination": "chuguo", "x": 0, "y": 2}
    assert m.delivered == {"quyuan"} and not m.task_complete
    # 识别昭君后到达西汉 → Arrived + TaskComplete（恰一次）
    m.enter_events(8, (2, 2), (2, 1))
    evs = m.enter_events(11, (2, 0), (2, 1))
    assert len(_by_type(evs, EVENT_ARRIVED)) == 1
    completes = _by_type(evs, EVENT_TASK_COMPLETE)
    assert len(completes) == 1 and completes[0].payload == {"steps": 11}
    assert m.task_complete
    # 再次进入终点：不重复 TaskComplete
    assert _by_type(m.enter_events(12, (2, 0), (2, 1)), EVENT_TASK_COMPLETE) == []
    print("[派送] OK：识别前不派送 / 识别后到达即派送 / TaskComplete 仅在达成时发一次")


def test_collision_and_timeout_events():
    m = make_manager()
    ev = m.collision_event(4, (0, 0), "N")
    assert ev.type == EVENT_COLLISION and ev.payload == {"x": 0, "y": 0, "direction": "N"}
    ev = m.timeout_event(100, 100)
    assert ev.type == EVENT_TIMEOUT and ev.payload == {"max_steps": 100}
    print("[碰撞/超时] OK：事件载荷正确")


def test_image_asset():
    """ImageFeed 的图像资产引用指向 PPT 素材。"""
    path = image_asset("quyuan")
    assert path and os.path.exists(path), f"屈原图像素材缺失: {path}"
    assert image_asset("nonexistent") is None
    print(f"[图像资产] OK：{path}")


if __name__ == "__main__":
    test_report()
    test_image_feed_on_entry()
    test_delivery_requires_identification()
    test_collision_and_timeout_events()
    test_image_asset()
    print("\n交互协议单测全部通过 [PASS]")
