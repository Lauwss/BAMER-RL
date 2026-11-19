# -*- coding: utf-8 -*-
"""交互协议单测（M1.1 验收 + 2026-11-20 契约定稿语义）：上报/图像下发/拾取/终点发现/派送事件按规则触发。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bamer_env import Maze, build_scene
from bamer_env.interaction import (
    EVENT_ARRIVED,
    EVENT_COLLISION,
    EVENT_DESTINATION_FEED,
    EVENT_IMAGE_FEED,
    EVENT_PICKUP,
    EVENT_REPORTED,
    EVENT_TASK_COMPLETE,
    EVENT_TIMEOUT,
    InteractionManager,
    destination_asset,
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
    # 从 (0,1) 走进屈原格 (1,1)：ImageFeed + Pickup（首达即识别+拾取）
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


def test_pickup_semantics():
    """拾取语义（2026-11-20 定稿）：首达即识别+拾取 / 手持时仅识别 / 送达后回取 / 已送达不重复拾取。"""
    m = make_manager()
    # 首达屈原格 (1,1)：ImageFeed(first) + Pickup 同步发生，carrying 生效
    evs = m.enter_events(1, (1, 1), (0, 1))
    feeds, pickups = _by_type(evs, EVENT_IMAGE_FEED), _by_type(evs, EVENT_PICKUP)
    assert len(feeds) == 1 and len(pickups) == 1
    assert pickups[0].payload == {"machine": "quyuan", "x": 1, "y": 1}
    assert m.carrying == "quyuan"
    # 手持屈原进入昭君格 (2,2)：仅识别（单机负载），不拾取
    evs = m.enter_events(3, (2, 2), (2, 1))
    assert len(_by_type(evs, EVENT_IMAGE_FEED)) == 1
    assert _by_type(evs, EVENT_PICKUP) == []
    assert m.identified == {"quyuan", "zhaojun"} and m.carrying == "quyuan"
    # 派送屈原到楚国 (0,2)：Arrived 自动完成，carrying 清空
    evs = m.enter_events(6, (0, 2), (1, 2))
    assert len(_by_type(evs, EVENT_ARRIVED)) == 1 and m.carrying is None
    # 空手再进昭君格：Pickup 生效（显式 Pick option 的 env 层表现 = 到达拾取）
    evs = m.enter_events(9, (2, 2), (2, 1))
    pickups = _by_type(evs, EVENT_PICKUP)
    assert len(pickups) == 1 and pickups[0].payload["machine"] == "zhaojun"
    assert m.carrying == "zhaojun"
    # 已送达机器重过其格：不再拾取（先放下昭君不可行——单机负载，此处直接派送后验证）
    m.enter_events(11, (2, 0), (2, 1))             # 昭君→西汉，carrying 清空
    assert m.carrying is None and m.delivered == {"quyuan", "zhaojun"}
    assert _by_type(m.enter_events(13, (1, 1), (1, 0)), EVENT_PICKUP) == []
    print("[拾取] OK：首达即拾取 / 单机负载 / 送达后可回取 / 已送达不重复拾取")


def test_destination_feed():
    """终点发现事件（2026-11-20 定稿）：进入即下发图像，first 标记首次发现；终点封闭须自主发现。"""
    m = make_manager()
    # 未携带时进终点格 = 仅发现（DestinationFeed），不派送
    evs = m.enter_events(2, (0, 2), (0, 1))        # 楚国 (0,2)
    feeds = _by_type(evs, EVENT_DESTINATION_FEED)
    assert len(feeds) == 1 and feeds[0].payload["destination"] == "chuguo"
    assert feeds[0].payload["first"] is True
    assert feeds[0].payload["image"] == destination_asset("chuguo")
    assert _by_type(evs, EVENT_ARRIVED) == []
    assert m.discovered == {"chuguo"}
    # 再进入重新下发（first=False）
    evs = m.enter_events(4, (0, 2), (0, 1))
    feeds = _by_type(evs, EVENT_DESTINATION_FEED)
    assert len(feeds) == 1 and feeds[0].payload["first"] is False
    print("[终点发现] OK：进入即发现 / first 标记 / 未携带不派送")


def test_delivery_requires_carrying():
    """派送前置（2026-11-20 定稿）：未携带不派送；携带映射机器到达即派送；双派送达成 TaskComplete 仅一次。"""
    m = make_manager()
    # 未携带进入楚国：仅 DestinationFeed，无派送
    evs = m.enter_events(2, (0, 2), (0, 1))
    assert _by_type(evs, EVENT_ARRIVED) == []
    assert not m.task_complete
    # 识别+拾取屈原后到达楚国 → Arrived(quyuan→chuguo)
    m.enter_events(3, (1, 1), (0, 1))
    assert m.carrying == "quyuan"
    evs = m.enter_events(6, (0, 2), (1, 2))
    arrived = _by_type(evs, EVENT_ARRIVED)
    assert len(arrived) == 1
    assert arrived[0].payload == {"machine": "quyuan", "destination": "chuguo", "x": 0, "y": 2}
    assert m.carrying is None and m.delivered == {"quyuan"} and not m.task_complete
    # 识别+拾取昭君后到达西汉 → Arrived + TaskComplete（恰一次）
    m.enter_events(8, (2, 2), (2, 1))
    assert m.carrying == "zhaojun"
    evs = m.enter_events(11, (2, 0), (2, 1))
    assert len(_by_type(evs, EVENT_ARRIVED)) == 1
    completes = _by_type(evs, EVENT_TASK_COMPLETE)
    assert len(completes) == 1 and completes[0].payload == {"steps": 11}
    assert m.task_complete
    # 再次进入终点：不重复 TaskComplete
    evs = m.enter_events(12, (2, 0), (2, 1))
    assert _by_type(evs, EVENT_TASK_COMPLETE) == []
    print("[派送] OK：未携带不派送 / 携带映射机器到达即派送 / TaskComplete 仅在达成时发一次")


def test_search_complete():
    """课程臂 A 阶段 0 判据：2 机器识别 + 2 终点发现 = search_complete（与派送无关）。"""
    m = make_manager()
    assert not m.search_complete
    m.enter_events(2, (0, 2), (0, 1))              # 发现楚国
    assert not m.search_complete
    m.enter_events(3, (1, 1), (0, 1))              # 识别+拾取屈原
    assert not m.search_complete
    m.enter_events(6, (2, 2), (2, 1))              # 识别昭君（手持屈原）
    assert not m.search_complete
    m.enter_events(8, (2, 0), (2, 1))              # 发现西汉 → 4 POI 齐了（尽管未派送）
    assert m.search_complete
    print("[搜索课程] OK：search_complete = 双识别 + 双终点发现，与派送解耦")


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
    test_pickup_semantics()
    test_destination_feed()
    test_delivery_requires_carrying()
    test_search_complete()
    test_collision_and_timeout_events()
    test_image_asset()
    print("\n交互协议单测全部通过 [PASS]")
