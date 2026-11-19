# -*- coding: utf-8 -*-
"""数据 IO 单测（M2.4 验收）：写→读往返字段一致；迷宫哈希稳定；无雷达数据兼容。"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from bamer_env import BamerEnv, Maze
from bamer_mouse import EpisodeRecord, LidarSpec, StepRecord, maze_hash, pose_at


def empty_maze(n: int = 5) -> Maze:
    return Maze([[False] * n for _ in range(n)], [[False] * n for _ in range(n)])


def task_env(**kwargs):
    params = dict(
        maze=empty_maze(5),
        state_machines={"quyuan": (0, 2), "zhaojun": (2, 2)},
        destinations={"chuguo": (4, 2), "xihan": (4, 3)},
        max_steps=50,
    )
    params.update(kwargs)
    return BamerEnv(**params)


def _event_to_dict(ev):
    from bamer_mouse.dataio import _event_to_dict as f
    return f(ev)


def test_maze_hash():
    m1 = empty_maze(5)
    m2 = empty_maze(5)
    assert maze_hash(m1) == maze_hash(m2)           # 同结构同哈希
    assert len(maze_hash(m1)) == 64
    m3 = empty_maze(5)
    m3.wall_south[2][2] = True
    assert maze_hash(m3) != maze_hash(m1)           # 改墙变哈希
    assert maze_hash(Maze.default()) != maze_hash(m1)   # 官方迷宫 ≠ 5×5 空图
    print("[迷宫哈希] OK：同结构同哈希 / 改墙变哈希（sha256）")


def test_roundtrip_with_sensor():
    """含雷达数据的完整往返：meta/位姿/动作/量测/点云/事件逐字段一致。"""
    env = task_env(lidar=LidarSpec(noise_sigma=0.02))
    obs, info = env.reset(seed=3)
    rec = EpisodeRecord(meta={
        "maze_hash": maze_hash(env.scene.maze),
        "reset_seed": 3,
        "scene": {
            "start": list(env.scene.start),
            "state_machines": {k: list(v) for k, v in env.scene.state_machines.items()},
            "destinations": {k: list(v) for k, v in env.scene.destinations.items()},
            "delivery_map": dict(env.scene.delivery_map),
        },
        "lidar": LidarSpec(noise_sigma=0.02).to_dict(),
        "created_at": "2026-11-19T00:00:00",
    })
    pose = pose_at(env.pose_cell, env.heading)
    rec.add_step(StepRecord(step=0, x=pose.x, y=pose.y, theta_deg=pose.theta_deg,
                            action="RESET", ranges=info["ranges"], angles=info["angles"],
                            points=info["points"], events=info["events"]))
    terminated = truncated = False
    while not (terminated or truncated):
        obs, _, terminated, truncated, info = env.step("F")     # 一路北进撞墙也不影响 IO 测试
        pose = pose_at(env.pose_cell, env.heading)
        rec.add_step(StepRecord(step=info["steps"], x=pose.x, y=pose.y,
                                theta_deg=pose.theta_deg, action="F",
                                ranges=info["ranges"], angles=info["angles"],
                                points=info["points"], events=info["events"]))
        if rec.steps[-1].step >= 8:
            break

    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "ep_test")
        paths = rec.save(base)
        assert set(paths.values()) == {base + ".npz", base + ".json"}
        rec2 = EpisodeRecord.load(base)

    assert rec2.meta == rec.meta
    assert len(rec2.steps) == len(rec.steps)
    for s1, s2 in zip(rec.steps, rec2.steps):
        assert s2.step == s1.step and (s2.x, s2.y, s2.theta_deg) == (s1.x, s1.y, s1.theta_deg)
        assert s2.action == s1.action
        assert np.array_equal(s2.ranges, s1.ranges, equal_nan=True)
        assert np.array_equal(s2.angles, s1.angles, equal_nan=True)
        assert np.array_equal(s2.points, s1.points, equal_nan=True)
        assert s2.events == [_event_to_dict(e) for e in s1.events]
    n_sensor = sum(1 for s in rec2.steps if s.ranges is not None)
    assert n_sensor == len(rec2.steps)
    print(f"[往返·带雷达] OK：{len(rec.steps)} 步 meta/位姿/动作/量测/点云/事件逐字段一致")


def test_roundtrip_sensorless():
    """无雷达 episode 往返：ranges/points 缺省 → 读回为 None，不报错。"""
    rec = EpisodeRecord(meta={"maze_hash": "x", "note": "sensorless"})
    rec.add_step(StepRecord(step=0, x=0.5, y=0.5, theta_deg=90.0, action="RESET"))
    rec.add_step(StepRecord(step=1, x=0.5, y=1.5, theta_deg=90.0, action="F"))
    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "ep_plain")
        rec.save(base)
        rec2 = EpisodeRecord.load(base)
    assert len(rec2.steps) == 2
    assert rec2.steps[1].ranges is None and rec2.steps[1].points is None
    assert rec2.steps[1].action == "F" and rec2.meta["note"] == "sensorless"
    print("[往返·无雷达] OK：缺省传感器字段读回 None")


def test_episode_dir_convention():
    """目录规范：save 自动创建 data/episodes/，写读一致（冒烟文件即存即删）。"""
    from bamer_mouse.dataio import EPISODE_DIR
    rec = EpisodeRecord(meta={"smoke": True})
    rec.add_step(StepRecord(step=0, x=0.5, y=0.5, theta_deg=90.0, action="RESET"))
    base = os.path.join(EPISODE_DIR, "_smoke_roundtrip")
    rec.save(base)
    rec2 = EpisodeRecord.load(base)
    assert rec2.meta["smoke"] is True
    os.remove(base + ".npz")
    os.remove(base + ".json")
    print(f"[目录规范] OK：data/episodes/ 自动创建并可写可读（冒烟文件已清理）")


if __name__ == "__main__":
    test_maze_hash()
    test_roundtrip_with_sensor()
    test_roundtrip_sensorless()
    test_episode_dir_convention()
    print("\n数据 IO 单测全部通过 [PASS]")
