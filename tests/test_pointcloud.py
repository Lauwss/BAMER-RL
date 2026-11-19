# -*- coding: utf-8 -*-
"""点云/占据栅格单测（M2.3 验收）：坐标链、占据栅格语义、量测数据与墙查询等价（R1 自洽）。"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from bamer_env import Maze
from bamer_mouse import (
    OCC_BLOCKED,
    OCC_FREE,
    OCC_UNKNOWN,
    Lidar,
    LidarSpec,
    Pose,
    ghost_alphas,
    occupancy_grid,
    polar_to_cartesian,
    pose_at,
    raycast,
    scan_to_world,
    to_world,
    walk_cells,
)


def empty_maze(n: int = 7) -> Maze:
    return Maze([[False] * n for _ in range(n)], [[False] * n for _ in range(n)])


# ---------- 坐标链 ----------
def test_polar_to_cartesian():
    xy = polar_to_cartesian(np.array([2.0, np.nan, 1.0]), np.array([90.0, 0.0, 180.0]))
    assert np.allclose(xy[0], [0.0, 2.0], atol=1e-12)
    assert np.isnan(xy[1]).all()                       # NaN 距离 → NaN 行
    assert np.allclose(xy[2], [-1.0, 0.0], atol=1e-12)
    print("[极坐标→笛卡尔] OK：正交变换正确，NaN 行保持")


def test_to_world_chain():
    """机体系 → 世界系：平移/旋转/NaN 保持；与 scan_to_world 全链一致。"""
    pose = Pose(1.5, 2.5, 90.0)
    local = np.array([[1.0, 0.0], [0.0, 1.0], [np.nan, np.nan]])
    world = to_world(local, pose)
    assert np.allclose(world[0], [1.5, 3.5], atol=1e-12)   # 正前方 1 格
    assert np.allclose(world[1], [0.5, 2.5], atol=1e-12)   # 左方 1 格
    assert np.isnan(world[2]).all()
    # 全链：雷达扫描（无噪声）→ 世界系点云必与 pose + R·hit_xy 一致
    maze = empty_maze(7)
    scan = Lidar(LidarSpec(max_range=10.0)).scan(maze.has_wall, pose_at((1, 1), "E"))
    w1 = scan_to_world(scan, pose_at((1, 1), "E"), drop_no_echo=True)
    w2 = to_world(scan.hit_xy[scan.echo_mask], pose_at((1, 1), "E"))
    assert np.allclose(w1, w2, atol=1e-12)
    assert len(w1) == int(scan.echo_mask.sum())
    print("[坐标链] OK：极坐标→机体系→世界系齐次变换与全链一致，无回波可丢弃")


# ---------- 测距数据格穿越 ↔ 墙查询参考实现（R1 等价性） ----------
def _ref_walk(wall_query, ox, oy, theta_deg, max_range):
    """参考实现：D7 样例工具的墙查询逐格穿越（先竖直后水平，同一 tie 约定）。"""
    a = math.radians(theta_deg)
    dx, dy = math.cos(a), math.sin(a)
    ix, iy = int(math.floor(ox)), int(math.floor(oy))
    free = [(ix, iy)]
    while True:
        if dx > 1e-12:
            tx = (ix + 1 - ox) / dx
        elif dx < -1e-12:
            tx = (ix - ox) / dx
        else:
            tx = float("inf")
        if dy > 1e-12:
            ty = (iy + 1 - oy) / dy
        elif dy < -1e-12:
            ty = (iy - oy) / dy
        else:
            ty = float("inf")
        t = min(tx, ty)
        if t > max_range:
            return free, None
        if tx <= ty:
            if wall_query(ix, iy, "E" if dx > 0 else "W"):
                return free, (ix + 1 if dx > 0 else ix - 1, iy)
            ix += 1 if dx > 0 else -1
        else:
            if wall_query(ix, iy, "N" if dy > 0 else "S"):
                return free, (ix, iy + 1 if dy > 0 else iy - 1)
            iy += 1 if dy > 0 else -1
        free.append((ix, iy))


def test_walk_cells_matches_raycast():
    """walk_cells（纯量测距离）与墙查询参考穿越在噪声关闭时逐格等价。"""
    maze = Maze.default()
    rng = np.random.default_rng(0)
    n_checked = 0
    for cx in (0, 2, 7, 11, 15):
        for cy in (0, 5, 8, 12, 15):
            pose = pose_at((cx, cy), "N")
            for i in range(0, 180, 7):                  # 采样束角
                theta = pose.theta_deg + scan_angles()[i]
                r, _, _ = raycast(maze.has_wall, pose.x, pose.y, theta, 2.0)
                dist = 2.0 if math.isnan(r) else r
                got_free, got_nb = walk_cells(pose.x, pose.y, theta, dist)
                ref_free, ref_nb = _ref_walk(maze.has_wall, pose.x, pose.y, theta, dist)
                if not math.isnan(r):
                    # 有回波：距离 = 首个有墙边界 → 两条实现必须逐格一致
                    assert got_free == ref_free, f"({cx},{cy}) θ={theta:.0f} free 不一致"
                    assert got_nb == ref_nb, f"({cx},{cy}) θ={theta:.0f} blocked 不一致"
                    n_checked += 1
                else:
                    # 无回波：穿越路径一致（blocked 参照可能停在量程边界，容许 None）
                    assert got_free == ref_free
    assert n_checked > 50
    print(f"[量测↔墙查询等价] OK：官方迷宫 {n_checked} 条有回波束逐格一致（R1 自洽）")


def scan_angles():
    return -180.0 + 2.0 * np.arange(180)


def test_occupancy_grid_semantics():
    """占据栅格：free/blocked/unknown 三值语义与窗口映射（世界对齐，行 0=北）。"""
    maze = empty_maze(7)
    pose = pose_at((3, 3), "N")
    scan = Lidar(LidarSpec(max_range=2.0)).scan(maze.has_wall, pose)
    grid = occupancy_grid(scan.ranges, scan.angles, pose, max_range=2.0, half=3)
    assert grid.shape == (7, 7) and grid.dtype == np.int8
    rc = lambda x, y: ((3 + 3) - y, x - (3 - 3))        # (row, col)
    assert grid[rc(3, 3)] == OCC_FREE                    # 自身格
    assert grid[rc(3, 4)] == OCC_FREE                    # 北邻（纵深 2.5>量程→partial free）
    assert grid[rc(3, 1)] == OCC_FREE                    # 南向 1.5 格内可穿
    assert grid[rc(2, 2)] == OCC_FREE                    # 斜向可达
    assert grid[rc(4, 4)] == OCC_FREE                    # 45° 束可达
    assert grid[rc(1, 1)] == OCC_UNKNOWN                 # 量程外不可达
    # 加北墙后：blocked 标注回波边界另一侧；侧旁仍 free
    maze2 = empty_maze(7)
    maze2.wall_south[3][4] = True
    scan2 = Lidar(LidarSpec(max_range=2.0)).scan(maze2.has_wall, pose)
    grid2 = occupancy_grid(scan2.ranges, scan2.angles, pose, max_range=2.0, half=3)
    assert grid2[rc(3, 4)] == OCC_BLOCKED
    assert grid2[rc(4, 4)] == OCC_FREE                   # 45° 束绕过墙端进入
    assert grid2[rc(3, 5)] == OCC_UNKNOWN                # 墙后纵深不可达
    print("[占据栅格] OK：free/blocked/unknown 三值语义与窗口映射正确")


def test_free_overrides_blocked():
    """free 覆盖 blocked：一格被某束标 blocked、又被另一束经开口穿入 → free。"""
    maze = empty_maze(7)
    maze.wall_south[3][4] = True                        # (3,3) 北墙
    pose = pose_at((2, 3), "E")
    scan = Lidar(LidarSpec(max_range=2.0)).scan(maze.has_wall, pose)
    grid = occupancy_grid(scan.ranges, scan.angles, pose, max_range=2.0, half=3)
    rc = lambda x, y: ((3 + 3) - y, x - (2 - 3))        # x0=2
    assert grid[rc(3, 4)] == OCC_FREE                   # 30°束 blocked，60°束经(2,4)穿入
    print("[free 覆盖] OK：多束证据合并取 free 优先（与 D7 样例语义一致）")


def test_occupancy_measurement_vs_wallquery():
    """验收复核：仅凭量测数据的占据栅格 == 墙查询参考实现栅格（官方迷宫多位姿）。"""
    maze = Maze.default()
    lidar = Lidar(LidarSpec(max_range=2.0))
    for cell, heading in [((2, 11), "N"), ((7, 7), "E"), ((11, 12), "W"), ((0, 0), "S")]:
        pose = pose_at(cell, heading)
        scan = lidar.scan(maze.has_wall, pose)
        got = occupancy_grid(scan.ranges, scan.angles, pose, max_range=2.0, half=3)
        # 参考实现：D7 样例式墙查询逐格穿越（blocked 先写、free 覆盖）
        half = 3
        size = 2 * half + 1
        x0, y0 = cell
        ref = np.full((size, size), OCC_UNKNOWN, dtype=np.int8)

        def to_rc(cx, cy):
            c, r = cx - (x0 - half), (y0 + half) - cy
            return (r, c) if 0 <= c < size and 0 <= r < size else None

        blocked_set, free_set = set(), set()
        for i in range(180):
            free, nb = _ref_walk(maze.has_wall, pose.x, pose.y,
                                 pose.theta_deg + scan_angles()[i], 2.0)
            free_set.update(free)
            if nb is not None:
                blocked_set.add(nb)
        for c in blocked_set:
            rc = to_rc(*c)
            if rc:
                ref[rc] = OCC_BLOCKED
        for c in free_set:
            rc = to_rc(*c)
            if rc:
                ref[rc] = OCC_FREE
        assert np.array_equal(got, ref), f"位姿 {cell}/{heading} 栅格不一致"
    print("[D7 复核] OK：正式实现与样例式墙查询栅格在 4 位姿逐位一致")


def test_ego_rotation():
    """ego 旋转：N 恒等；E=逆时针 90°（右邻格转到正上方）。"""
    maze = empty_maze(7)
    lidar = Lidar(LidarSpec(max_range=2.0))
    for heading, k in (("N", 0), ("E", 1), ("S", 2), ("W", 3)):
        pose = pose_at((3, 3), heading)
        scan = lidar.scan(maze.has_wall, pose)
        world = occupancy_grid(scan.ranges, scan.angles, pose, max_range=2.0, half=3)
        ego = occupancy_grid(scan.ranges, scan.angles, pose, max_range=2.0, half=3, ego=True)
        expect = np.rot90(world, k=k) if k else world
        assert np.array_equal(ego, expect), f"ego 旋转 {heading} 不符"
    # 车头朝上性质：朝东时正前方格（东邻）出现在中心行上一行
    pose = pose_at((3, 3), "E")
    scan = lidar.scan(maze.has_wall, pose)
    ego = occupancy_grid(scan.ranges, scan.angles, pose, max_range=2.0, half=3, ego=True)
    assert ego[2, 3] == OCC_FREE
    print("[ego 旋转] OK：车头朝上约定 N/E/S/W 映射正确")


def test_ghost_alphas():
    alphas = ghost_alphas(10, n_recent=3, alpha0=0.5, decay=0.6)
    assert len(alphas) == 3
    assert alphas[-1] == 0.5 and abs(alphas[0] - 0.18) < 1e-12
    assert all(b > a for a, b in zip(alphas, alphas[1:]))   # 旧→新递增
    assert len(ghost_alphas(2, n_recent=5)) == 2            # 帧数不足时截断
    print("[残影] OK：时间衰减透明度序列正确")


if __name__ == "__main__":
    test_polar_to_cartesian()
    test_to_world_chain()
    test_walk_cells_matches_raycast()
    test_occupancy_grid_semantics()
    test_free_overrides_blocked()
    test_occupancy_measurement_vs_wallquery()
    test_ego_rotation()
    test_ghost_alphas()
    print("\n点云/占据栅格单测全部通过 [PASS]")
