# -*- coding: utf-8 -*-
"""frontier 引导单测（2026-11-21 探索塌缩修复§改动1）：前沿格语义、掩码收窄
一致性、完全观测回退。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from bamer_env import BamerEnv, Maze
from bamer_mouse.lidar import LidarSpec

from algo.actions import explore_index, frontier_cells, legal_mask, reachable_cells
from algo.belief import BeliefState, W_FREE, W_UNKNOWN, W_WALL


def scanned_env_belief(seed: int, n: int = 16):
    """官方图 + 雷达：reset 后扫描一帧，返回 (env, belief)。"""
    env = BamerEnv(maze=Maze.default(), randomize=True, max_steps=100,
                   lidar=LidarSpec(n_beams=180, max_range=2.0, noise_sigma=0.0))
    obs, info = env.reset(seed=seed)
    belief = BeliefState(n=n, max_range=2.0)
    belief.observe(obs, info)
    return env, belief


def test_frontier_cell_properties():
    """前沿格 = 可达 且 邻接 ≥1 unknown 槽；当前格必然不是前沿（本格四槽已揭示）。"""
    env, belief = scanned_env_belief(seed=7)
    fron = frontier_cells(belief)
    reach = reachable_cells(belief, belief.cell)
    assert fron <= reach, "frontier 必须是可达集子集"
    assert belief.cell not in fron, "当前格四槽已由本格扫描揭示，不应为前沿"
    for (x, y) in fron:
        slots = [belief.ledger.wall_value(x, y, d) for d in ("S", "W", "N", "E")]
        assert W_UNKNOWN in slots, f"前沿格 {(x, y)} 必须邻接 unknown 槽，实测 {slots}"
    env.close()
    print("[前沿格语义] OK：可达子集 + unknown 邻接 + 当前格排除")


def test_legal_mask_frontier_narrowing():
    """frontier_only=True 的掩码是全掩码的子集：Explore 收窄、Pick/Deliver 不变、非空。"""
    env, belief = scanned_env_belief(seed=11)
    full = legal_mask(belief)
    narrow = legal_mask(belief, frontier_only=True)
    assert narrow.any() and full.any(), "两种掩码都必须非空"
    # Pick/Deliver 位完全一致
    assert np.array_equal(full[256:], narrow[256:]), "Pick/Deliver 合法性不受收窄影响"
    # Explore 位收窄为子集
    explore_full = {i for i in np.flatnonzero(full[:256])}
    explore_narrow = {i for i in np.flatnonzero(narrow[:256])}
    assert explore_narrow <= explore_full, "Explore 掩码应收窄为子集"
    assert explore_narrow, "非完全观测时前沿 Explore 候选不应为空"
    # 收窄后的候选恰好是 frontier 格（排除当前格）
    fron = frontier_cells(belief)
    fron.discard(belief.cell)
    expect = {explore_index(c, belief.n) for c in fron}
    assert explore_narrow == expect, "收窄 Explore 位应精确等于前沿格索引集"
    env.close()
    print("[掩码收窄] OK：Explore 子集精确匹配、任务位不变、非空保证")


def test_frontier_empty_fallback():
    """可达区域完全观测（账本全 W_FREE）→ frontier 空 → 回退掩码与全掩码一致。"""
    belief = BeliefState(n=5, max_range=2.0)
    belief.ledger.sensed_south[:] = W_FREE
    belief.ledger.sensed_west[:] = W_FREE
    belief.cell = (2, 2)
    assert frontier_cells(belief) == set(), "全开放账本下前沿应为空"
    full = legal_mask(belief)
    narrow = legal_mask(belief, frontier_only=True)
    assert np.array_equal(full, narrow), "frontier 为空时应回退到全可达集掩码"
    print("[完全观测回退] OK：frontier 空 → 掩码等价于全可达集")


def test_frontier_narrowing_end_to_end_episode():
    """端到端：official 图上跑一条 greedy episode（frontier_only=True），
    验证合法掩码每步非空且轨迹推进（不抛异常即通过基本健壮性）。"""
    from algo.executor import Executor

    env, belief = scanned_env_belief(seed=3)
    executor = Executor(belief, max_option_steps=100)
    steps = 0
    for _ in range(20):
        mask = legal_mask(belief, frontier_only=True)
        assert mask.any(), "每步合法集非空（回退保证）"
        explore_cnt = int(np.flatnonzero(mask[:256]).size)
        assert explore_cnt > 0, "frontier_only 下每步都应有前沿 Explore 候选"
        result = executor.run(env, int(np.flatnonzero(mask)[0]))  # 任意合法 option
        belief.commit_option(result.path, result.start_cell)
        steps += result.steps
        if result.terminated or result.truncated:
            break
    assert steps > 0
    env.close()
    print("[端到端健壮性] OK：frontier 掩码下 20 个 option 全程合法非空")


if __name__ == "__main__":
    test_frontier_cell_properties()
    test_legal_mask_frontier_narrowing()
    test_frontier_empty_fallback()
    test_frontier_narrowing_end_to_end_episode()
    print("全部通过")
