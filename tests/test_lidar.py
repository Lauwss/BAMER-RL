# -*- coding: utf-8 -*-
"""激光雷达单测（M2.2 验收）：raycast 与解析解对拍（噪声关/开）、开口穿透、噪声可复现。"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from bamer_env import Maze
from bamer_mouse import Lidar, LidarSpec, pose_at, raycast


def empty_maze(n: int = 5) -> Maze:
    return Maze([[False] * n for _ in range(n)], [[False] * n for _ in range(n)])


def test_analytic_empty_room():
    """空房间解析解：直射四向距离 = 到外边框距离，命中点在墙线上。"""
    maze = empty_maze(5)
    cases = [
        (pose_at((2, 2), "N"), 90.0, 2.5, (2.5, 5.0)),    # 北上边框
        (pose_at((2, 2), "N"), 270.0, 2.5, (2.5, 0.0)),   # 正后方（南边框）
        (pose_at((2, 2), "E"), 0.0, 2.5, (5.0, 2.5)),     # 东右边框
        (pose_at((2, 2), "W"), 180.0, 2.5, (0.0, 2.5)),   # 西左边框
    ]
    for pose, world_theta, exp_r, exp_hit in cases:
        r, hx, hy = raycast(maze.has_wall, pose.x, pose.y, world_theta, max_range=10.0)
        assert math.isclose(r, exp_r, abs_tol=1e-12), f"theta={world_theta}: r={r} ≠ {exp_r}"
        assert math.isclose(hx, exp_hit[0], abs_tol=1e-9)
        assert math.isclose(hy, exp_hit[1], abs_tol=1e-9)
    # 量程截断：距离 2.5 > 量程 2.0 → 无回波
    r, hx, hy = raycast(maze.has_wall, 2.5, 2.5, 90.0, max_range=2.0)
    assert math.isnan(r) and math.isnan(hx) and math.isnan(hy)
    print("[解析解·空房] OK：四向直射距离/命中点与解析解一致，量程截断=NaN")


def test_analytic_wall_and_angled():
    """内墙解析解 + 斜射束（30°/45°）与闭式解对拍。"""
    maze = empty_maze(5)
    maze.wall_south[2][2] = True        # (2,2) 与 (2,1) 之间
    r, hx, hy = raycast(maze.has_wall, 2.5, 2.5, 270.0, max_range=10.0)
    assert math.isclose(r, 0.5, abs_tol=1e-12) and math.isclose(hy, 2.0, abs_tol=1e-12)
    # 30° 斜射：命中右边框 x=5，t = 2.5/cos30°，y = 2.5 + t·sin30°
    r, hx, hy = raycast(maze.has_wall, 2.5, 2.5, 30.0, max_range=10.0)
    t_exp = 2.5 / math.cos(math.radians(30.0))
    assert math.isclose(r, t_exp, abs_tol=1e-12)
    assert math.isclose(hx, 5.0, abs_tol=1e-9)
    assert math.isclose(hy, 2.5 + t_exp * math.sin(math.radians(30.0)), abs_tol=1e-9)
    # 45° 斜射：命中角点 (5,5)，t = 2.5·√2（角点 tie 约定先竖直后水平）
    r, hx, hy = raycast(maze.has_wall, 2.5, 2.5, 45.0, max_range=10.0)
    assert math.isclose(r, 2.5 * math.sqrt(2.0), abs_tol=1e-12)
    assert math.isclose(hx, 5.0, abs_tol=1e-9) and math.isclose(hy, 5.0, abs_tol=1e-9)
    print("[解析解·斜射] OK：内墙 0.5 格、30°/45° 闭式解与命中点一致")


def test_opening_penetration():
    """开口穿透：缺失的墙边界上射线自然穿过（M2.2 核心语义）。"""
    maze = empty_maze(5)
    # y=2|3 边界整排墙，仅 x=2 处留开口
    for x in (0, 1, 3, 4):
        maze.wall_south[x][3] = True
    # 正对开口：(2,2) 朝北 → 穿过开口直达外边框 y=5
    r, hx, hy = raycast(maze.has_wall, 2.5, 2.5, 90.0, max_range=10.0)
    assert math.isclose(r, 2.5, abs_tol=1e-12) and math.isclose(hy, 5.0, abs_tol=1e-9)
    # 默认量程 2.0：开口后下一堵墙 2.5 格 > 量程 → 无回波（豁口可见性符合 D4 量程设计）
    r, *_ = raycast(maze.has_wall, 2.5, 2.5, 90.0, max_range=2.0)
    assert math.isnan(r)
    # 开口两侧邻位朝北：墙在 0.5 格处
    r, hx, hy = raycast(maze.has_wall, 1.5, 2.5, 90.0, max_range=10.0)
    assert math.isclose(r, 0.5, abs_tol=1e-12)
    print("[开口穿透] OK：缺口射线直达 2.5 格外框，邻位 0.5 格命中")


def test_scan_machine_frame():
    """全扫描输出：形状/束角/机体系笛卡尔自洽；NaN 语义一致。"""
    maze = empty_maze(5)
    lidar = Lidar()                     # D4 默认：180 束 / 360° / 2.0 格 / 无噪声
    scan = lidar.scan(maze.has_wall, pose_at((0, 0), "N"))
    assert scan.ranges.shape == (180,) and scan.angles.shape == (180,)
    assert scan.hit_xy.shape == (180, 2)
    assert np.allclose(scan.angles, -180.0 + 2.0 * np.arange(180))
    assert np.array_equal(scan.echo_mask, ~np.isnan(scan.ranges))
    # 机体系自洽：有回波束 hit_xy = r·(cos a, sin a)
    echo = scan.echo_mask
    expect = scan.ranges[echo, None] * np.stack(
        [np.cos(np.radians(scan.angles[echo])), np.sin(np.radians(scan.angles[echo]))], axis=1)
    assert np.allclose(scan.hit_xy[echo], expect, atol=1e-12)
    assert np.isnan(scan.hit_xy[~echo]).all()
    # 束 0 = 正后方（机体系 -180°，世界系朝南）→ 南边框 0.5 格
    assert math.isclose(scan.ranges[0], 0.5, abs_tol=1e-12)
    # 正前方束（机体系 0° = 束 90）朝北：边框 2.5 > 量程 → 无回波
    assert math.isnan(scan.ranges[90])
    # 正左方束（机体系 +90° = 束 135）朝西：边框 0.5 格
    assert math.isclose(scan.ranges[135], 0.5, abs_tol=1e-12)
    print("[扫描输出] OK：形状/束角/机体系自洽/NaN 语义全部正确")


def test_noise_reproducibility():
    """噪声模式：同 seed 逐位可复现；带噪距离与解析解偏差有界；越界截断。"""
    maze = empty_maze(5)
    pose = pose_at((1, 1), "E")
    spec = LidarSpec(noise_sigma=0.02)
    noisy1 = Lidar(spec, seed=42).scan(maze.has_wall, pose)
    noisy2 = Lidar(spec, seed=42).scan(maze.has_wall, pose)
    assert np.array_equal(noisy1.ranges, noisy2.ranges, equal_nan=True)
    assert np.array_equal(noisy1.hit_xy, noisy2.hit_xy, equal_nan=True)
    clean = Lidar(LidarSpec(), seed=0).scan(maze.has_wall, pose)
    echo = clean.echo_mask
    assert echo.any(), "测试位姿应有回波束"
    assert np.all(np.abs(noisy1.ranges[echo] - clean.ranges[echo]) <= 5 * spec.noise_sigma)
    assert np.all((noisy1.ranges[echo] > 0) & (noisy1.ranges[echo] <= spec.max_range))
    diff = Lidar(spec, seed=7).scan(maze.has_wall, pose)
    assert not np.allclose(diff.ranges[echo], noisy1.ranges[echo])
    print(f"[噪声] OK：seed 复现逐位一致；{int(echo.sum())} 束回波偏差 ≤5σ 且截断于量程内")


def test_scan_perf_smoke():
    """性能冒烟：默认 180 束全扫描应为微秒~毫秒级（阈值放宽到 100ms 防抖）。"""
    maze = Maze.default()
    lidar = Lidar()
    pose = pose_at((2, 11), "N")
    lidar.scan(maze.has_wall, pose)     # 预热
    t0 = time.perf_counter()
    for _ in range(50):
        lidar.scan(maze.has_wall, pose)
    dt_ms = (time.perf_counter() - t0) / 50 * 1000
    assert dt_ms < 100.0, f"单帧扫描 {dt_ms:.2f} ms 超阈值"
    print(f"[性能] OK：官方迷宫 180 束单帧扫描 {dt_ms:.3f} ms")


def test_interior_guard():
    """射线起点在格边界上时几何退化 → 显式报错（插值位姿不得用于扫描）。"""
    try:
        raycast(empty_maze(5).has_wall, 1.0, 2.5, 0.0, 2.0)
        raise AssertionError("边界起点未报错")
    except ValueError:
        pass
    print("[退化防护] OK：格边界起点显式报错")


if __name__ == "__main__":
    test_analytic_empty_room()
    test_analytic_wall_and_angled()
    test_opening_penetration()
    test_scan_machine_frame()
    test_noise_reproducibility()
    test_scan_perf_smoke()
    test_interior_guard()
    print("\n激光雷达单测全部通过 [PASS]")
