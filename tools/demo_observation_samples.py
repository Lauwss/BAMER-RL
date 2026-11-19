# -*- coding: utf-8 -*-
"""D7 观测表示三选一样例生成（PLAN.md 决策登记 D7：agent 出样例供用户选定）。

三种候选观测表示（与 M2.3 命名一致）：
    A. 原始点云        —— 世界系散点 + 单帧极坐标视图（信息无损，维度高）
    B. 局部占据栅格    —— 鼠周围 W×W 格 free/blocked/unknown 三值图（结构化稀疏）
    C. 降采样极坐标向量 —— 360° 均分 S 扇区取最小距离（维度最低，即插即用）

雷达数据来源（重要）：本脚本内嵌一份**样例专用临时 DDA raycast**（Amanatides & Woo
格边界穿越，未与解析解对拍），仅用于生成 D7 选型样例；正式 LidarSpec + raycast
在阶段二 M2.2 `bamer_mouse/lidar.py` 实现（简单房间解析解对拍 + 开口穿透单测），
届时用正式实现复核本页样例。

参数取 D4 默认项：360° / 180 束 / 量程 2.0 格 / 噪声关闭。
输出：docs/d7_samples/*.png + README.md（选型说明，D7 待用户拍板）。
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

from bamer_env import Maze

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "docs", "d7_samples")

# D4 默认雷达参数（样例用）
N_BEAMS = 180
MAX_RANGE = 2.0

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HEADING_DEG = {"E": 0.0, "N": 90.0, "W": 180.0, "S": 270.0}


# ---------- 样例专用临时 raycast（M2.2 正式版将落在 bamer_mouse/lidar.py） ----------
def _next_boundary(ox, oy, ix, iy, dx, dy):
    """Amanatides & Woo：从点 (ox,oy) 所在格 (ix,iy) 出发，射线到下一格边界的参数距离。"""
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
    return tx, ty


def _raycast_beam(maze, x, y, theta_deg, max_range=MAX_RANGE):
    """单束 raycast：返回 (距离[格], 回波点 xy or None)。无回波返回 (NaN, None)。"""
    a = math.radians(theta_deg)
    dx, dy = math.cos(a), math.sin(a)
    ox, oy = x + 0.5, y + 0.5
    ix, iy = x, y
    while True:
        tx, ty = _next_boundary(ox, oy, ix, iy, dx, dy)
        t = min(tx, ty)
        if t > max_range:
            return float("nan"), None
        if tx <= ty:                       # 先穿过竖直边界（E/W 墙）
            d = "E" if dx > 0 else "W"
            if maze.has_wall(ix, iy, d):
                return t, (ox + t * dx, oy + t * dy)
            ix += 1 if dx > 0 else -1
        else:                              # 先穿过水平边界（N/S 墙）
            d = "N" if dy > 0 else "S"
            if maze.has_wall(ix, iy, d):
                return t, (ox + t * dx, oy + t * dy)
            iy += 1 if dy > 0 else -1


def _scan(maze, x, y, n_beams=N_BEAMS, max_range=MAX_RANGE):
    """全扫描：返回 (ranges[n], points[n,2])，无回波距离为 NaN、点为 NaN 占位。"""
    angles = np.arange(n_beams) * (360.0 / n_beams)
    ranges = np.empty(n_beams)
    points = np.full((n_beams, 2), np.nan)
    for i, ang in enumerate(angles):
        r, p = _raycast_beam(maze, x, y, ang, max_range)
        ranges[i] = r
        if p is not None:
            points[i] = p
    return angles, ranges, points


def _walk_beam_cells(maze, x, y, theta_deg, max_range=MAX_RANGE):
    """单束逐格穿越：返回 (途经格列表, 撞墙侧邻格 or None)——占据栅格分类用。"""
    a = math.radians(theta_deg)
    dx, dy = math.cos(a), math.sin(a)
    ox, oy = x + 0.5, y + 0.5
    ix, iy = x, y
    free = [(ix, iy)]
    while True:
        tx, ty = _next_boundary(ox, oy, ix, iy, dx, dy)
        t = min(tx, ty)
        if t > max_range:
            return free, None
        if tx <= ty:
            d = "E" if dx > 0 else "W"
            if maze.has_wall(ix, iy, d):
                return free, (ix + 1 if dx > 0 else ix - 1, iy)
            ix += 1 if dx > 0 else -1
        else:
            d = "N" if dy > 0 else "S"
            if maze.has_wall(ix, iy, d):
                return free, (ix, iy + 1 if dy > 0 else iy - 1)
            iy += 1 if dy > 0 else -1
        free.append((ix, iy))


# ---------- 样例位姿自动挑选（走廊纵深 / 开口附近路口） ----------
def _open_dirs(maze, x, y):
    return {d for d in "NESW" if not maze.has_wall(x, y, d)}


def pick_poses(maze):
    corridor = junction = None
    for yy in range(maze.n):
        for xx in range(maze.n):
            od = _open_dirs(maze, xx, yy)
            if corridor is None and len(od) == 2 and ({"N", "S"} <= od or {"E", "W"} <= od):
                corridor = (xx, yy)
            if junction is None and len(od) >= 3:
                junction = (xx, yy)
            if corridor and junction:
                return corridor, junction
    return (0, 0), (maze.n - 1, maze.n - 1)


def _heading_of(maze, x, y):
    od = sorted(_open_dirs(maze, x, y))
    return od[0] if od else "N"


# ---------- 绘制 ----------
def _draw_walls(ax, maze, lw=0.8, color="0.45"):
    n = maze.n
    for xx in range(n):
        for yy in range(n):
            if maze.wall_south[xx][yy]:
                ax.plot([xx, xx + 1], [yy, yy], color=color, lw=lw, zorder=2)
            if maze.wall_west[xx][yy]:
                ax.plot([xx, xx], [yy, yy + 1], color=color, lw=lw, zorder=2)
    for xx in range(n):   # 外边框
        ax.plot([xx, xx + 1], [0, 0], color=color, lw=lw, zorder=2)
        ax.plot([xx, xx + 1], [n, n], color=color, lw=lw, zorder=2)
    for yy in range(n):
        ax.plot([0, 0], [yy, yy + 1], color=color, lw=lw, zorder=2)
        ax.plot([n, n], [yy, yy + 1], color=color, lw=lw, zorder=2)


def _draw_mouse(ax, x, y, heading):
    a = math.radians(HEADING_DEG[heading])
    ca, sa = math.cos(a), math.sin(a)
    pts = ((0.26, 0.0), (-0.17, 0.16), (-0.10, 0.0), (-0.17, -0.16))
    tri = [(x + 0.5 + px * ca - py * sa, y + 0.5 + px * sa + py * ca) for px, py in pts]
    ax.add_patch(plt.Polygon(tri, closed=True, color="#d62728", zorder=6))


def fig_a_pointcloud(maze, pose, angles, ranges, points):
    """方案 A：原始点云（世界系叠加主视图 + r-θ 极坐标单帧视图）。"""
    x, y, heading = pose
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 6.2),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    # 世界系叠加
    ax1.set_xlim(-0.5, maze.n - 0.5)
    ax1.set_ylim(-0.5, maze.n - 0.5)
    ax1.set_aspect("equal")
    ax1.grid(True, ls=":", lw=0.5, color="0.75")
    _draw_walls(ax1, maze)
    valid = ~np.isnan(ranges)
    sc = ax1.scatter(points[valid, 0], points[valid, 1], c=ranges[valid],
                     cmap="viridis", s=7, zorder=5)
    fig.colorbar(sc, ax=ax1, fraction=0.04, label="距离（格）")
    _draw_mouse(ax1, x, y, heading)
    ax1.set_title("世界系叠加：点云贴墙 / 开口穿透")
    # r-θ 极坐标单帧
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    theta = np.radians(angles)
    r = np.where(valid, ranges, MAX_RANGE)
    ax2.scatter(theta, r, s=6, c=np.where(valid, ranges, 0), cmap="viridis")
    ax2.scatter(theta[~valid], np.full((~valid).sum(), MAX_RANGE), s=6, c="lightgray",
                label="无回波(=量程上限)")
    ax2.set_title("机载单帧 r-θ 视图")
    ax2.legend(loc="lower left", fontsize=7)
    fig.suptitle(f"方案 A：原始点云（180 束 × 360°，量程 {MAX_RANGE} 格）"
                 f"——位姿 {pose[0]},{pose[1]} 朝 {heading}", fontsize=12)
    return fig


def fig_b_occupancy(maze, pose, half=3):
    """方案 B：局部占据栅格（free / blocked / unknown）。

    数组行 0 = 迷宫 y 最大（y+half）、列 0 = x 最小（x-half），imshow 直绘后
    y 刻度标注真实迷宫 y 值；真实墙线在行空间（y 向下）叠加作对照。
    """
    x, y, heading = pose
    n = 2 * half + 1
    free_set, blocked_set = set(), set()
    for ang in range(0, 360, 2):
        free, blocked = _walk_beam_cells(maze, x, y, ang)
        free_set.update(free)
        if blocked is not None:
            blocked_set.add(blocked)

    grid = np.zeros((n, n))          # 0 unknown 1 free 2 blocked
    for cx, cy in blocked_set:       # 先 blocked
        c, r = cx - (x - half), (y + half) - cy
        if 0 <= c < n and 0 <= r < n:
            grid[r, c] = 2
    for cx, cy in free_set:          # free 覆盖 blocked（该格可经其他路径进入）
        c, r = cx - (x - half), (y + half) - cy
        if 0 <= c < n and 0 <= r < n:
            grid[r, c] = 1

    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    cmap = ListedColormap(["#e9e9e9", "#bfe3bf", "#e58f8f"])
    ax.imshow(grid, cmap=cmap, norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3), zorder=1)
    ax.grid(True, ls=":", lw=0.5, color="0.55", zorder=2)

    # 环境侧参考：真实墙线（样例对照用，非观测输入）
    for cx in range(x - half, x + half + 1):
        for cy in range(y - half, y + half + 1):
            if not (0 <= cx < maze.n and 0 <= cy < maze.n):
                continue
            c, r = cx - (x - half), (y + half) - cy
            if maze.wall_south[cx][cy] and r + 1 < n:    # 南墙 → 行空间下边界
                ax.plot([c - 0.5, c + 0.5], [r + 0.5, r + 0.5], color="k", lw=1.6, zorder=3)
            if maze.wall_west[cx][cy]:                   # 西墙 → 列左边界
                ax.plot([c - 0.5, c - 0.5], [r - 0.5, r + 0.5], color="k", lw=1.6, zorder=3)

    # 鼠标记（行空间 y 向下，旋转角取负保证屏幕方向正确）
    a = math.radians(-HEADING_DEG[heading])
    ca, sa = math.cos(a), math.sin(a)
    pts = ((0.26, 0.0), (-0.17, 0.16), (-0.10, 0.0), (-0.17, -0.16))
    tri = [(half + px * ca - py * sa, half + px * sa + py * ca) for px, py in pts]
    ax.add_patch(plt.Polygon(tri, closed=True, color="#d62728", zorder=6))

    ax.set_xticks(range(n))
    ax.set_xticklabels([str(x - half + c) for c in range(n)], fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels([str(y + half - r) for r in range(n)], fontsize=7)
    ax.set_xlabel("迷宫 x")
    ax.set_ylabel("迷宫 y")
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color="#bfe3bf", label="free 射线可穿"),
               mpatches.Patch(color="#e58f8f", label="blocked 该侧有墙"),
               mpatches.Patch(color="#e9e9e9", label="unknown 量程外")]
    ax.legend(handles=handles, loc="upper left", fontsize=7)
    ax.set_title(f"方案 B：局部占据栅格（{n}×{n} 窗，黑色线=真实墙线仅作对照）"
                 f"——位姿 {x},{y} 朝 {heading}", fontsize=11)
    return fig


def fig_c_polar_vector(maze, pose, n_sectors=24):
    """方案 C：降采样极坐标向量（n_sectors 扇区 × 最小距离，rose 图）。"""
    x, y, heading = pose
    angles, ranges, _ = _scan(maze, x, y)
    sector_vals = np.empty(n_sectors)
    width = 360.0 / n_sectors
    for s in range(n_sectors):
        mask = (angles >= s * width) & (angles < (s + 1) * width)
        seg = ranges[mask]
        seg = seg[~np.isnan(seg)]
        sector_vals[s] = seg.min() if len(seg) else MAX_RANGE   # 无回波 → 量程上限
    fig = plt.figure(figsize=(6.4, 6.4))
    ax = fig.add_subplot(projection="polar")
    theta = np.radians(np.arange(n_sectors) * width + width / 2)
    colors = plt.cm.viridis(sector_vals / MAX_RANGE)
    ax.bar(theta, sector_vals, width=np.radians(width) * 0.92, color=colors,
           edgecolor="0.4", lw=0.4, alpha=0.9)
    ax.set_ylim(0, MAX_RANGE * 1.05)
    ax.set_title(f"方案 C：降采样极坐标向量（{n_sectors} 扇区 × 最小距离，维度 {n_sectors}）"
                 f"——位姿 {x},{y} 朝 {heading}", fontsize=11, pad=16)
    return fig


def main():
    os.makedirs(OUT, exist_ok=True)
    maze = Maze.default()
    corridor, junction = pick_poses(maze)
    poses = {
        "走廊纵深": (*corridor, _heading_of(maze, *corridor)),
        "开口/路口": (*junction, _heading_of(maze, *junction)),
    }
    print(f"样例位姿：{poses}")

    saved = []
    for name, pose in poses.items():
        x, y, heading = pose
        angles, ranges, points = _scan(maze, x, y)
        echo = int((~np.isnan(ranges)).sum())
        tag = {"走廊纵深": "corridor", "开口/路口": "junction"}[name]
        f = fig_a_pointcloud(maze, pose, angles, ranges, points)
        p = os.path.join(OUT, f"obs_a_pointcloud_{tag}.png")
        f.savefig(p, dpi=130, bbox_inches="tight"); plt.close(f); saved.append(p)
        f = fig_b_occupancy(maze, pose)
        p = os.path.join(OUT, f"obs_b_occupancy_{tag}.png")
        f.savefig(p, dpi=130, bbox_inches="tight"); plt.close(f); saved.append(p)
        f = fig_c_polar_vector(maze, pose)
        p = os.path.join(OUT, f"obs_c_polar_{tag}.png")
        f.savefig(p, dpi=130, bbox_inches="tight"); plt.close(f); saved.append(p)
        print(f"[{name}] 位姿 {pose}：{echo}/{N_BEAMS} 束有回波")

    for p in saved:
        print(f"已保存: {p}")
    print(f"\n样例图共 {len(saved)} 张（2 位姿 × 3 方案）→ {OUT}")
    print("同位姿逐方案对照即可横向比较；D7 待用户选定（AGENT.md 3.2：观测表示为用户决策项）")


if __name__ == "__main__":
    main()
