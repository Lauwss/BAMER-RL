# -*- coding: utf-8 -*-
"""固定迷宫诊断单测（2026-11-21）：maze_fixed 语义——拓扑跨集钉死、机器布点每集
随机、终点固定默认；generated 实例种子确定；训练器两 stage 共享同一张实例。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bamer_env import BamerEnv, Maze
from bamer_env.scene import DEFAULT_DESTINATIONS

from algo.trainer import TrainConfig, build_fixed_maze


def _walls(maze: Maze):
    return maze.wall_south, maze.wall_west


def test_fixed_maze_env_semantics():
    """固定迷宫 + randomize：跨 reset 拓扑不变、机器布点随 seed 变化且合法。"""
    maze = Maze.random_maze(n=16, seed=200, braiding=0.8, target_loops=37)
    env = BamerEnv(maze=maze, randomize=True, max_steps=100)

    _, info1 = env.reset(seed=1)
    maze1 = env.scene.maze
    machines1 = dict(env.scene.state_machines)
    assert _walls(maze1) == _walls(maze), "reset 后迷宫拓扑与固定实例不一致"
    assert env.scene.destinations == DEFAULT_DESTINATIONS, "终点应为固定默认位置"
    # 布点合法性：不占起点/终点格
    assert env.scene.start not in machines1.values()
    for cell in machines1.values():
        assert cell not in DEFAULT_DESTINATIONS.values()

    _, info2 = env.reset(seed=2)
    assert _walls(env.scene.maze) == _walls(maze), "跨 reset 迷宫拓扑变化（应为固定）"
    assert env.scene.destinations == DEFAULT_DESTINATIONS
    assert env.scene.state_machines != machines1, \
        "不同 seed 机器布点应不同（seed=1/2 恰好同布点的概率可忽略）"

    # 同 seed 可复现
    env.reset(seed=1)
    assert env.scene.state_machines == machines1, "同 seed 布点应逐位可复现"
    print("[固定迷宫语义] OK：拓扑钉死、布点随机且合法")


def test_build_fixed_maze_deterministic():
    """generated：同 cfg 两次构建逐位一致；official：尺寸 16 且两 stage 共用校验。"""
    cfg = TrainConfig(maze_fixed="generated", seed=0, maze_braiding_stage1=0.8)
    m1, m2 = build_fixed_maze(cfg), build_fixed_maze(cfg)
    assert _walls(m1) == _walls(m2), "generated 固定迷宫应种子确定"

    cfg_official = TrainConfig(maze_fixed="official")
    m3 = build_fixed_maze(cfg_official)
    assert m3.n == 16, "官方迷宫应为 16×16"

    try:
        build_fixed_maze(TrainConfig(maze_fixed="bogus"))
        raise AssertionError("非法 maze_fixed 应报错")
    except ValueError:
        pass
    print("[固定迷宫构建] OK：generated 确定性与非法值校验")


def test_trainer_shares_maze_across_stages():
    """训练器：两 stage 构建的环境共享同一固定迷宫实例（课程只剩奖励切换）。"""
    from algo.trainer import Trainer

    cfg = TrainConfig(
        arm="A", stage0_options=10, stage1_options=10, max_episodes_per_stage=2,
        env_max_steps=100, lidar_n_beams=60, maze_fixed="official",
        warmup_options=5, batch_size=8, target_sync_options=15,
        eval_every_options=0, swanlab=False, device="cpu", smoke=True,
        out_dir="algo/runs/_test_fixed_maze_stage",
    )
    trainer = Trainer(cfg)
    env0 = trainer._build_env("search", "stage0")
    env1 = trainer._build_env(None, "stage1")
    assert env0.scene is None and env1.scene is None  # reset 前场景未建
    assert trainer._fixed_maze is not None
    assert env0._maze_fixed is trainer._fixed_maze
    assert env1._maze_fixed is trainer._fixed_maze
    env0.close()
    env1.close()
    print("[两 stage 共享] OK：stage0/stage1 环境共用同一固定迷宫实例")


if __name__ == "__main__":
    test_fixed_maze_env_semantics()
    test_build_fixed_maze_deterministic()
    test_trainer_shares_maze_across_stages()
    print("全部通过")
