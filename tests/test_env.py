# -*- coding: utf-8 -*-
"""Env 单测（M1.2 验收）：reset 种子可复现、终止/截断语义正确、计步含旋转、信息封闭。"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from bamer_env import BamerEnv, Maze
from bamer_env.env import HEADINGS
from bamer_env.interaction import (
    EVENT_ARRIVED,
    EVENT_COLLISION,
    EVENT_IMAGE_FEED,
    EVENT_REPORTED,
    EVENT_TASK_COMPLETE,
    EVENT_TIMEOUT,
)


def empty_maze(n: int = 5) -> Maze:
    """无内墙迷宫（外边框由 Maze 构造自动封闭）。"""
    return Maze([[False] * n for _ in range(n)], [[False] * n for _ in range(n)])


def task_env(**kwargs):
    """5×5 空迷宫任务场景：屈原(0,2)→楚国(4,2)，昭君(2,2)→西汉(4,3)，起点 (0,0) 朝北。"""
    params = dict(
        maze=empty_maze(5),
        state_machines={"quyuan": (0, 2), "zhaojun": (2, 2)},
        destinations={"chuguo": (4, 2), "xihan": (4, 3)},
        max_steps=100,
    )
    params.update(kwargs)
    return BamerEnv(**params)


def _by_step(info):
    return info["events"][-1].step


def _of_type(events, etype):
    return [e for e in events if e.type == etype]


def test_obs_and_spaces():
    env = task_env()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs), f"观测不在占位空间内: {obs}"
    assert obs["cell"] == 0 and obs["heading"] == 0          # (0,0) 朝北
    assert set(info.keys()) == {"steps", "events", "identified", "delivered"}
    assert info["steps"] == 0
    print("[观测/空间] OK：占位 Dict 空间合法，info 键固定")


def test_rotation_step_counting():
    """旋转计入步数（R4）：4 次 TL = 4 步且朝向复原；再 F = 第 5 步前进一格。"""
    env = task_env()
    env.reset(seed=0)
    for k in range(1, 5):
        _, _, _, _, info = env.step("TL")
        assert info["steps"] == k and _by_step(info) == k
    _, _, _, _, info = env.step("F")
    assert info["steps"] == 5
    assert env.heading == "N"
    obs, *_ = env.step("TR")   # 第 6 步：转向东
    assert obs["heading"] == HEADINGS.index("E")
    obs, _, _, _, info = env.step("F")   # 第 7 步：东进一格 → (1,1)
    assert obs["cell"] == 1 * 5 + 1 and info["steps"] == 7
    # 每步首事件均为实时上报，步号一致
    for _ in range(3):
        _, _, _, _, info = env.step("F")
        assert info["events"][0].type == EVENT_REPORTED
        assert info["events"][0].step == info["steps"]
    print("[计步] OK：旋转计步 / 朝向状态机 / 上报步号一致")


def test_collision_keeps_position():
    """撞墙：位置不变、计 1 步、Collision 事件，不终止。"""
    env = task_env()
    env.reset(seed=0)
    _, _, terminated, truncated, info = env.step("TB")   # 转向南
    assert not terminated and not truncated and env.heading == "S"
    _, _, terminated, truncated, info = env.step("F")    # (0,0) 南边是外边框
    assert env.pose_cell == (0, 0) and info["steps"] == 2
    assert _of_type(info["events"], EVENT_COLLISION)[0].payload == {
        "x": 0, "y": 0, "direction": "S"}
    assert not terminated and not truncated
    print("[碰撞] OK：位置不变 + 事件 + 计步，不单方面终止")


def test_full_task_flow():
    """完整任务流：识别→派送→TaskComplete，事件序列与步数逐点核对。"""
    env = task_env()
    obs, info = env.reset(seed=0)
    actions = ["F", "F", "TR", "F", "F", "F", "F", "TL", "F"]
    # 步 2 进入 (0,2) 识别屈原；步 5 进入 (2,2) 识别昭君；
    # 步 7 进入 (4,2) 派送屈原→楚国；步 9 进入 (4,3) 派送昭君→西汉 → TaskComplete
    events_log, flags = [info["events"]], []
    for a in actions:
        obs, reward, terminated, truncated, info = env.step(a)
        events_log.append(info["events"])
        flags.append((terminated, truncated))
    feeds = [(e.step, e.payload["machine"]) for evs in events_log
             for e in _of_type(evs, EVENT_IMAGE_FEED)]
    arrived = [(e.step, e.payload["machine"], e.payload["destination"])
               for evs in events_log for e in _of_type(evs, EVENT_ARRIVED)]
    completes = [e.step for evs in events_log for e in _of_type(evs, EVENT_TASK_COMPLETE)]
    assert feeds == [(2, "quyuan"), (5, "zhaojun")]
    assert arrived == [(7, "quyuan", "chuguo"), (9, "zhaojun", "xihan")]
    assert completes == [9]
    assert flags[-1] == (True, False) and all(not t for t, _ in flags[:-1])
    assert info["steps"] == 9 and env.pose_cell == (4, 3)
    assert obs["identified"].tolist() == [1, 1] and obs["delivered"].tolist() == [1, 1]
    # 任务达成后环境仍可交互，terminated 保持完成态
    obs, _, terminated, truncated, info = env.step("F")
    assert terminated is True and truncated is False
    print("[完整任务流] OK：识别@2/5 派送@7/9 TaskComplete@9（共 9 步）")


def test_delivery_requires_identification():
    """识别前经过终点格不派送；单侧派送不终止。"""
    env = task_env()
    env.reset(seed=0)
    for a in ["TR", "F", "F", "F", "F", "TL", "F", "F"]:    # 东行至 (4,0) 再北上 (4,2)
        obs, _, terminated, truncated, info = env.step(a)
    assert not _of_type(info["events"], EVENT_ARRIVED)      # 未识别任何状态机 → 不派送
    for a in ["TL", "F", "F", "TB", "F", "F", "TL", "F"]:   # 去识别昭君(2,2) 再折返 (4,3)
        obs, _, terminated, truncated, info = env.step(a)
    assert ("zhaojun", "xihan") in [(e.payload["machine"], e.payload["destination"])
                                    for e in _of_type(info["events"], EVENT_ARRIVED)]
    assert info["delivered"] == ["zhaojun"]                  # 单侧派送
    assert not _of_type(info["events"], EVENT_TASK_COMPLETE)
    assert terminated is False
    print("[派送前置] OK：识别前不派送 / 单侧派送不终止")


def test_timeout_truncation():
    """超时截断：达到 max_steps → truncated + Timeout 事件，terminated 保持 False。"""
    env = task_env(max_steps=3)
    env.reset(seed=0)
    last = None
    for _ in range(3):
        last = env.step("F")
    _, _, terminated, truncated, info = last
    assert terminated is False and truncated is True
    assert info["steps"] == 3
    timeouts = _of_type(info["events"], EVENT_TIMEOUT)
    assert len(timeouts) == 1 and timeouts[0].payload == {"max_steps": 3}
    print("[超时] OK：truncated + Timeout 事件")


def test_terminated_wins_at_max_steps():
    """恰好在第 max_steps 步完成任务：terminated 优先，不发 Timeout。"""
    env = task_env(max_steps=9)
    env.reset(seed=0)
    for a in ["F", "F", "TR", "F", "F", "F", "F", "TL", "F"]:
        obs, _, terminated, truncated, info = env.step(a)
    assert terminated is True and truncated is False
    assert _of_type(info["events"], EVENT_TIMEOUT) == []
    assert _of_type(info["events"], EVENT_TASK_COMPLETE)
    print("[终止优先] OK：同步到达时 terminated 优先于 truncated")


def _random_trajectory(seed: int):
    env = BamerEnv(randomize=True, max_steps=60)
    obs, info = env.reset(seed=seed)
    rng = random.Random(0)
    traj = [obs["cell"]]
    for _ in range(50):
        obs, _, terminated, truncated, info = env.step(rng.randrange(4))
        traj.append(obs["cell"])
        if terminated or truncated:
            break
    return traj, env.scene.state_machines, env.scene.maze


def test_seed_reproducibility():
    """同种子两次 reset + 同动作序列 → 轨迹/布点/迷宫逐位一致（可复现）。"""
    t1, m1, maze1 = _random_trajectory(123)
    t2, m2, maze2 = _random_trajectory(123)
    assert t1 == t2 and m1 == m2
    assert maze1.wall_south == maze2.wall_south and maze1.wall_west == maze2.wall_west
    print(f"[可复现] OK：seed=123 随机场景 50 步轨迹一致，布点 {m1}")


def test_100_random_seeds():
    """验收项：默认场景 + 100 个随机种子场景，任务点互相可达（scene.validate()）。"""
    build = BamerEnv()
    obs, _ = build.reset(seed=0)
    build.scene.validate()
    for seed in range(100):
        env = BamerEnv(randomize=True, max_steps=10)
        obs, info = env.reset(seed=seed)
        env.scene.validate()                     # 起点可达全部状态机与终点
        sm = env.scene.state_machines
        assert len(set(sm.values())) == 2        # 两状态机不重合
        assert all(v not in env.scene.destinations.values() for v in sm.values())
        assert env.observation_space.contains(obs)
    print("[100 种子] OK：100 个随机种子场景任务点全部可达，观测合法")


def test_reset_options_injection():
    """R2 代码冻结接口：reset(options=...) 注入迷宫与布点，不改任务代码。

    注入迷宫尺寸须与 observation_space 一致（构造时固定，默认 16×16）。
    """
    env = BamerEnv()
    env.reset(seed=0)
    obs, info = env.reset(options={
        "maze": empty_maze(16),
        "state_machines": {"quyuan": (0, 2), "zhaojun": (2, 2)},
        "destinations": {"chuguo": (4, 2), "xihan": (4, 3)},
    })
    assert env.scene.state_machines == {"quyuan": (0, 2), "zhaojun": (2, 2)}
    for a in ["F", "F", "TR", "F", "F", "F", "F", "TL", "F"]:
        obs, _, terminated, truncated, info = env.step(a)
    assert terminated is True
    print("[注入接口] OK：reset(options=...) 注入新场景并完成任务")


def test_info_closure():
    """R1 信息封闭：obs/info 不含迷宫结构、状态机位置、派送映射。"""
    env = task_env()
    obs, info = env.reset(seed=0)
    for a in ["F", "TR", "F", "F", "F", "TL"]:
        obs, _, _, _, info = env.step(a)
    forbidden = ("maze", "wall", "state_machine", "delivery_map", "destination_cell",
                 "machine_cell", "scene")
    assert set(info.keys()) == {"steps", "events", "identified", "delivered"}
    for ev in info["events"]:
        for key in ev.payload:
            assert not any(f in key.lower() for f in forbidden), f"事件载荷泄漏字段: {key}"
    assert set(obs.keys()) == {"cell", "heading", "identified", "delivered"}
    # obs 中不含任何墙线/布点信息（cell 与 heading 是机器鼠自身状态）
    assert obs["cell"] < 25 and obs["heading"] < 4
    print("[信息封闭] OK：obs/info 无场景内部字段")


def test_action_validation():
    env = task_env()
    env.reset(seed=0)
    try:
        env.step(4)
        raise AssertionError("动作越界未报错")
    except ValueError:
        pass
    try:
        env.step("XX")
        raise AssertionError("未知原语未报错")
    except ValueError:
        pass
    env2 = task_env()
    try:
        env2.step("F")
        raise AssertionError("reset 前 step 未报错")
    except RuntimeError:
        pass
    print("[动作校验] OK：越界/未知原语/未 reset 均报错")


def test_action_space_smoke():
    """动作空间随机采样冒烟：30 步无异常。"""
    env = task_env()
    env.reset(seed=0)
    for _ in range(30):
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset(seed=0)
    print("[冒烟] OK：action_space.sample() 30 步无异常")


def test_render_rgb_array():
    env = task_env(render_mode="rgb_array")
    env.reset(seed=0)
    for _ in range(3):
        env.step("F")
    frame = env.render()
    assert isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[2] == 3
    assert frame.dtype == np.uint8
    print(f"[渲染] OK：rgb_array {frame.shape[1]}×{frame.shape[0]}×3")


if __name__ == "__main__":
    test_obs_and_spaces()
    test_rotation_step_counting()
    test_collision_keeps_position()
    test_full_task_flow()
    test_delivery_requires_identification()
    test_timeout_truncation()
    test_terminated_wins_at_max_steps()
    test_seed_reproducibility()
    test_100_random_seeds()
    test_reset_options_injection()
    test_info_closure()
    test_action_validation()
    test_action_space_smoke()
    test_render_rgb_array()
    print("\nEnv 单测全部通过 [PASS]")
