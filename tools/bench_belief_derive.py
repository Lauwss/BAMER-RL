# -*- coding: utf-8 -*-
"""Belief tensor 派生速度基准：每步 16×16×4 = 1024 次墙位查询。

测量两件事：
1. 逐格循环派生 (4,16,16) uint8 的单步耗时（贴近 maze.py 的 list-of-list 风格）
2. numpy 向量化版本耗时（对比），并验证两者结果一致
"""
import random
import timeit

import numpy as np

N = 16
rng = random.Random(0)

# ground truth 墙线（list-of-list，与 maze.py 同构，外边框封闭）
wall_south = [[rng.random() < 0.4 for _ in range(N)] for _ in range(N)]
wall_west = [[rng.random() < 0.4 for _ in range(N)] for _ in range(N)]
for x in range(N):
    wall_south[x][0] = True
for y in range(N):
    wall_west[0][y] = True

# 探测记录：假设已探明约 60% 的墙位
sensed_south = [[rng.random() < 0.6 for _ in range(N)] for _ in range(N)]
sensed_west = [[rng.random() < 0.6 for _ in range(N)] for _ in range(N)]


def derive_loop():
    """逐格派生 (4,N,N) uint8：0=unknown, 1=open(已探明无墙), 2=wall。"""
    B = np.zeros((4, N, N), dtype=np.uint8)
    for x in range(N):
        for y in range(N):
            # N: 规范形式 = south[x][y+1]；y=N-1 为外边框，恒有墙
            if y + 1 < N:
                if sensed_south[x][y + 1]:
                    B[0, x, y] = 2 if wall_south[x][y + 1] else 1
            else:
                B[0, x, y] = 2
            # S: south[x][y]
            if sensed_south[x][y]:
                B[1, x, y] = 2 if wall_south[x][y] else 1
            # W: west[x][y]
            if sensed_west[x][y]:
                B[2, x, y] = 2 if wall_west[x][y] else 1
            # E: 规范形式 = west[x+1][y]；x=N-1 为外边框
            if x + 1 < N:
                if sensed_west[x + 1][y]:
                    B[3, x, y] = 2 if wall_west[x + 1][y] else 1
            else:
                B[3, x, y] = 2
    return B


def derive_numpy():
    """向量化版本。注意 maze.py 的索引是 [x][y]，asarray 后 shape=(N,N) 即 [x,y]。"""
    ws = np.asarray(wall_south)
    ww = np.asarray(wall_west)
    ss = np.asarray(sensed_south)
    sw = np.asarray(sensed_west)
    B = np.zeros((4, N, N), dtype=np.uint8)
    B[0, :, :-1] = np.where(ss[:, 1:], np.where(ws[:, 1:], 2, 1), 0)
    B[0, :, -1] = 2                      # 北外边框
    B[1] = np.where(ss, np.where(ws, 2, 1), 0)
    B[2] = np.where(sw, np.where(ww, 2, 1), 0)
    B[3, :-1, :] = np.where(sw[1:, :], np.where(ww[1:, :], 2, 1), 0)
    B[3, -1, :] = 2                      # 东外边框
    return B


# 正确性：两版必须逐元素一致
assert np.array_equal(derive_loop(), derive_numpy()), "loop 与 numpy 版结果不一致！"
print("正确性: loop 版 == numpy 版  [OK]")

REPS = 2000
t_loop = timeit.timeit(derive_loop, number=REPS) / REPS
t_np = timeit.timeit(derive_numpy, number=REPS) / REPS
print(f"循环版:   {t_loop*1e6:8.1f} us/次   ({1/t_loop:,.0f} 次/秒)")
print(f"numpy版:  {t_np*1e6:8.1f} us/次   ({1/t_np:,.0f} 次/秒)")

# 训练规模外推（论文量级：600 episodes × ~150 步 ≈ 9 万决策步）
steps = 600 * 150
print(f"\n训练全程外推（600 episodes × 150 步 = {steps:,} 步）:")
print(f"  循环版总开销 ≈ {t_loop*steps:6.2f} 秒")
print(f"  numpy版总开销 ≈ {t_np*steps:6.2f} 秒")

# 显存侧账：replay buffer 存 uint8 观测
obs_bytes = 6 * N * N  # 6 通道（4 墙位 + 2 位置），uint8
print(f"\n观测尺寸: {obs_bytes} B/步 = {6*N*N} uint8")
print(f"  replay buffer 10 万步 ≈ {obs_bytes*100000/1e6:.1f} MB")
