# -*- coding: utf-8 -*-
"""点云生成与呈现（M2.3，PLAN.md 阶段二）。

坐标链：机体系极坐标扫描（lidar.ScanResult）→ 机体系笛卡尔（hit_xy）
       → 世界系点云（齐次变换 to_world：旋转 theta_deg + 平移到位姿）。

观测表示（D7 已定，2026-11-19 用户拍板）：算法侧输入 = 局部占据栅格（方案 B）。
occupancy_grid 只依赖雷达量测（ranges/angles）+ 机器鼠自身位姿做"测距数据
格穿越"——不查询墙体（无 wall_query 参数），与真实机器鼠仅凭自身传感数据
建图的语义一致（R1 信息封闭自洽：任何算法侧拿到的量测都能复算出同一栅格）。

窗口约定（与 docs/d7_samples 方案 B 样例一致，正式窗参随 M2.6 契约冻结定稿）：
- 世界对齐窗口 (2*half+1)×(2*half+1)，列 0 = x-(half)，行 0 = y+(half)
  （imshow 直绘时上=北）；
- 取值 0=unknown 量程外 / 1=free 射线可穿 / 2=blocked 回波边界另一侧；
- 先写 blocked 再写 free（free 覆盖 blocked：该格可经其他路径进入）；
- ego=True 时按朝向旋转为"车头朝上"表示（N 恒等，E 逆时针转 90°，S 180°，W 270°）。

呈现四层（对应成果要求"实时显示激光点云"）：
    1. 极坐标单帧视图    plot_polar（雷达原始 r-θ 视角，报告讲解用）
    2. 世界系叠加主视图  由 bamer_env/viewer 承担（Frame.points 挂点已接通）
    3. 点云残影          ghost_alphas + draw_points（近 N 帧按时间衰减叠加）
    4. 信念图            属算法侧产出，仅当用户要求时由 agent 提供渲染（暂缺）
"""
import math
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .kinematics import HEADING_DEG, Pose, cell_of

# 栅格取值常量（D7-B）
OCC_UNKNOWN, OCC_FREE, OCC_BLOCKED = 0, 1, 2

# 回波边界"另一侧"判定用的前探偏移（远小于格尺寸）
_HIT_EPS = 1e-6


# ---------- 坐标链 ----------
def polar_to_cartesian(ranges: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """机体系极坐标 (r, θ) → 机体系笛卡尔 (n,2)；无回波（NaN r）→ NaN 行。"""
    r = np.asarray(ranges, dtype=float)
    a = np.radians(np.asarray(angles, dtype=float))
    xy = np.stack([r * np.cos(a), r * np.sin(a)], axis=-1)
    xy[~np.isfinite(r)] = np.nan
    return xy


def to_world(points_local: np.ndarray, pose: Pose) -> np.ndarray:
    """机体系点集 → 世界系（齐次变换：先旋转 theta_deg，再平移到位姿）。

    points_local (n,2)；NaN 坐标经旋转平移后仍为 NaN（无回波行自动保持无效）。
    机体系 x 轴 = 机器鼠正前方；束角/朝向均以"逆时针为正"。
    """
    pts = np.asarray(points_local, dtype=float)
    a = math.radians(pose.theta_deg)
    c, s = math.cos(a), math.sin(a)
    return pts @ np.array([[c, -s], [s, c]]).T + np.array([pose.x, pose.y])


def scan_to_world(scan, pose: Pose, drop_no_echo: bool = True) -> np.ndarray:
    """一帧扫描 → 世界系点云（坐标链全链；drop_no_echo 丢弃 NaN 无回波束）。"""
    if drop_no_echo:
        return to_world(scan.hit_xy[scan.echo_mask], pose)
    return to_world(scan.hit_xy, pose)


# ---------- 测距数据格穿越（无墙体查询，R1 自洽） ----------
def walk_cells(ox: float, oy: float, theta_deg: float, dist: float) -> Tuple[List[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """从 (ox,oy) 沿射向走 dist 距离的逐格穿越（纯几何，不查墙）。

    返回 (free_cells, hit_neighbor)：
    - free_cells：射线在 (0, dist) 内经过的格（含起始格）；
    - hit_neighbor：dist 为回波距离时，回波边界另一侧的邻格（无法判定时 None）。

    与 lidar.raycast 同一套边界推进与角点 tie 约定（先竖直后水平），因此
    噪声关闭时由量测数据复算的穿越路径与真实射线路径逐格一致（有单测对拍）；
    噪声开启时按带噪距离退化处理（free 优先的写入顺序自动消解小偏差）。
    """
    a = math.radians(theta_deg)
    dx, dy = math.cos(a), math.sin(a)
    ix, iy = int(math.floor(ox)), int(math.floor(oy))
    free = [(ix, iy)]
    eps = 1e-9
    for _ in range(4 * int(math.ceil(dist)) + 8):
        if dx > eps:
            tx = (ix + 1 - ox) / dx
        elif dx < -eps:
            tx = (ix - ox) / dx
        else:
            tx = float("inf")
        if dy > eps:
            ty = (iy + 1 - oy) / dy
        elif dy < -eps:
            ty = (iy - oy) / dy
        else:
            ty = float("inf")
        t = tx if tx <= ty else ty
        if t > dist + eps:
            return free, None                      # dist 内无更多边界
        if tx <= ty:
            neighbor = (ix + 1 if dx > 0 else ix - 1, iy)
        else:
            neighbor = (ix, iy + 1 if dy > 0 else iy - 1)
        if t >= dist - eps:
            return free, neighbor                  # 回波恰在此边界：不跨入
        # 跨入邻格（量测意义上该段可穿）
        if tx <= ty:
            ix += 1 if dx > 0 else -1
        else:
            iy += 1 if dy > 0 else -1
        free.append((ix, iy))
    return free, None


def occupancy_grid(ranges: np.ndarray, angles: np.ndarray, pose: Pose, max_range: float,
                   half: int = 3, ego: bool = False) -> np.ndarray:
    """局部占据栅格（D7-B）：仅凭雷达量测 + 自身位姿复算。

    参数：ranges/angles 为一帧扫描（机体系）；max_range 与扫描量程一致；
    half 窗口半宽（默认 3 → 7×7）；ego=True 输出"车头朝上"旋转表示。
    返回 (2*half+1, 2*half+1) int8 数组，取值 OCC_UNKNOWN/OCC_FREE/OCC_BLOCKED。
    """
    ranges = np.asarray(ranges, dtype=float)
    angles = np.asarray(angles, dtype=float)
    x0, y0 = cell_of(pose)
    size = 2 * half + 1
    grid = np.full((size, size), OCC_UNKNOWN, dtype=np.int8)

    def to_rc(cx: int, cy: int) -> Optional[Tuple[int, int]]:
        c, r = cx - (x0 - half), (y0 + half) - cy
        if 0 <= c < size and 0 <= r < size:
            return (r, c)
        return None

    blocked_cells: List[Tuple[int, int]] = []
    free_cells: List[Tuple[int, int]] = []
    for i in range(ranges.shape[0]):
        echo = bool(np.isfinite(ranges[i]))
        dist = float(ranges[i]) if echo else float(max_range)
        free, neighbor = walk_cells(pose.x, pose.y, pose.theta_deg + angles[i], dist)
        free_cells.extend(free)
        if echo and neighbor is not None:
            blocked_cells.append(neighbor)

    for cx, cy in blocked_cells:               # 先 blocked
        rc = to_rc(cx, cy)
        if rc is not None:
            grid[rc[0], rc[1]] = OCC_BLOCKED
    for cx, cy in free_cells:                  # 后 free（free 覆盖 blocked）
        rc = to_rc(cx, cy)
        if rc is not None:
            grid[rc[0], rc[1]] = OCC_FREE

    if ego:
        # 行 0 = 北（世界对齐上=北）；ego 要求车头朝上：
        # N 恒等；E 需把"右"转到"上"（图像逆时针 90°）；S 转 180°；W 转 270°。
        k = {"N": 0, "E": 1, "S": 2, "W": 3}[pose.heading]
        if k:
            grid = np.rot90(grid, k=k)
    return np.ascontiguousarray(grid)


# ---------- 呈现辅助（层 1 / 层 3；层 2 由 bamer_env.viewer 承担） ----------
def plot_polar(ranges: np.ndarray, angles: np.ndarray, max_range: float,
               title: str = "机载单帧 r-θ 视图", ax=None):
    """层 1：极坐标单帧视图（雷达原始视角；无回波束以灰色画在量程上限）。"""
    ranges = np.asarray(ranges, dtype=float)
    angles = np.asarray(angles, dtype=float)
    echo = np.isfinite(ranges)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 5.2), subplot_kw={"projection": "polar"})
    r = np.where(echo, ranges, max_range)
    sc = ax.scatter(np.radians(angles[echo]), r[echo], s=6,
                    c=ranges[echo], cmap="viridis")
    if (~echo).any():
        ax.scatter(np.radians(angles[~echo]), np.full((~echo).sum(), max_range),
                   s=6, c="lightgray", label="无回波(=量程上限)")
        ax.legend(loc="lower left", fontsize=7)
    ax.set_ylim(0, max_range * 1.05)
    # 稀疏径向刻度 + 标签错角：避免 0° 轴上密集数字互相重叠（目检问题修正）
    n_rgrid = 4
    step = max_range / n_rgrid
    ax.set_rgrids([step * k for k in range(1, n_rgrid + 1)],
                  angle=157.5, fontsize=7)
    ax.set_title(title, fontsize=10, pad=12)
    return ax


def draw_points(ax, points_world: np.ndarray, color: str = "#1f77b4",
                size: float = 5.0, alpha: float = 0.8, zorder: float = 6.0):
    """层 2/3 基元：把一帧世界系点云画到给定坐标轴（NaN 点由 matplotlib 跳过）。"""
    pts = np.asarray(points_world, dtype=float)
    if pts.size == 0:
        return None
    return ax.scatter(pts[:, 0], pts[:, 1], s=size, c=color, alpha=alpha, zorder=zorder)


def ghost_alphas(n_frames: int, n_recent: int = 5, alpha0: float = 0.5,
                 decay: float = 0.6) -> List[float]:
    """层 3：残影透明度序列（旧→新），近 n_recent 帧按 decay 几何衰减。

    例：n_recent=3, alpha0=0.5, decay=0.6 → 旧 [0.18, 0.30] + 新 0.5。
    当前帧（最新一帧）建议由 draw_points 用实色另画，残影只放历史帧。
    """
    n = min(int(n_recent), int(n_frames))
    return [alpha0 * decay ** (n - 1 - i) for i in range(n)]
