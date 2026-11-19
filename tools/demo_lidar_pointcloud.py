# -*- coding: utf-8 -*-
"""M2.2/M2.3 验收演示：雷达点云可视化 + 带点云探索录像。

产出（docs/，对应 PLAN.md 阶段二验收"点云可视化图通过目检"）：
    lidar_wall_front.png    墙前贴墙：0.5 格直视内墙（世界系 + 极坐标 + 占据栅格）
    lidar_opening.png       开口穿透：PPT 虚线开口处射线穿过缺失墙边界
    lidar_corridor.png      走廊纵深：官方迷宫最长 sightline
    lidar_ghost_trail.png   点云残影：近 N 帧时间衰减叠加（呈现四层之第 3 层）
    _demo_pointcloud.gif    带实时点云的完整探索录像（展示层 Frame.points 挂点）

位姿挑选为环境侧演示逻辑（可用墙信息）；探索策略仅消费 info 中的雷达量测
（R1 合法）。全部渲染基于正式 LidarSpec/raycast/占据栅格实现。
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import bamer_env
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from bamer_env import BamerEnv, Maze
from bamer_env.interaction import EVENT_IMAGE_FEED, EVENT_TASK_COMPLETE
from bamer_env.viewer import EpisodeAnimator, HEADING_DEG, record_episode
from bamer_mouse import (
    Lidar,
    LidarSpec,
    ghost_alphas,
    occupancy_grid,
    plot_polar,
    pose_at,
    raycast,
    scan_to_world,
    try_move,
)

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "docs")

LIDAR = Lidar(LidarSpec())          # D4 默认：360° / 180 束 / 2.0 格 / 无噪声
OCC_CMAP = ListedColormap(["#e9e9e9", "#bfe3bf", "#e58f8f"])
OCC_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)


# ---------- 场景绘制（与 bamer_env/viewer 同风格，演示脚本自含） ----------
def _draw_walls(ax, maze, lw=1.6, color="k"):
    n = maze.n
    for x in range(n):
        for y in range(n):
            if maze.wall_south[x][y]:
                ax.plot([x, x + 1], [y, y], color=color, lw=lw, zorder=3, solid_capstyle="round")
            if maze.wall_west[x][y]:
                ax.plot([x, x], [y, y + 1], color=color, lw=lw, zorder=3, solid_capstyle="round")
    for k in range(n):
        ax.plot([k, k + 1], [0, 0], color=color, lw=lw, zorder=3)
        ax.plot([k, k + 1], [n, n], color=color, lw=lw, zorder=3)
        ax.plot([0, 0], [k, k + 1], color=color, lw=lw, zorder=3)
        ax.plot([n, n], [k, k + 1], color=color, lw=lw, zorder=3)


def _draw_mouse(ax, pose, color="#d62728", zorder=7):
    a = math.radians(pose.theta_deg)
    ca, sa = math.cos(a), math.sin(a)
    pts = ((0.26, 0.0), (-0.17, 0.16), (-0.10, 0.0), (-0.17, -0.16))
    tri = [(pose.x + px * ca - py * sa, pose.y + px * sa + py * ca) for px, py in pts]
    ax.add_patch(plt.Polygon(tri, closed=True, color=color, zorder=zorder))


def _draw_occupancy(ax, grid, pose, half, maze):
    """占据栅格视图（世界对齐，行 0=北）；真实墙线叠加仅为对照（非观测输入）。"""
    size = 2 * half + 1
    ax.imshow(grid, cmap=OCC_CMAP, norm=OCC_NORM, zorder=1)
    ax.grid(True, ls=":", lw=0.5, color="0.55", zorder=2)
    x0, y0 = int(pose.x - 0.5), int(pose.y - 0.5)
    for cx in range(x0 - half, x0 + half + 1):
        for cy in range(y0 - half, y0 + half + 1):
            if not (0 <= cx < maze.n and 0 <= cy < maze.n):
                continue
            r, c = (y0 + half) - cy, cx - (x0 - half)
            if maze.wall_south[cx][cy] and r + 1 < size:
                ax.plot([c - 0.5, c + 0.5], [r + 0.5, r + 0.5], color="k", lw=1.6, zorder=3)
            if maze.wall_west[cx][cy]:
                ax.plot([c - 0.5, c - 0.5], [r - 0.5, r + 0.5], color="k", lw=1.6, zorder=3)
    a = math.radians(-pose.theta_deg)   # 行空间 y 向下，屏幕方向取负
    ca, sa = math.cos(a), math.sin(a)
    pts = ((0.3, 0.0), (-0.2, 0.18), (-0.12, 0.0), (-0.2, -0.18))
    tri = [(half + px * ca - py * sa, half + px * sa + py * ca) for px, py in pts]
    ax.add_patch(plt.Polygon(tri, closed=True, color="#d62728", zorder=6))
    ax.set_xticks(range(size))
    ax.set_xticklabels([str(x0 - half + c) for c in range(size)], fontsize=7)
    ax.set_yticks(range(size))
    ax.set_yticklabels([str(y0 + half - r) for r in range(size)], fontsize=7)
    ax.set_xlabel("迷宫 x")
    ax.set_ylabel("迷宫 y")


def _triple_figure(maze, cell, heading, title, out_name, lidar=LIDAR):
    """三联图：世界系点云叠加（大图）+ 极坐标单帧 + 局部占据栅格（D7-B）。"""
    pose = pose_at(cell, heading)
    scan = lidar.scan(maze.has_wall, pose)
    pts = scan_to_world(scan, pose, drop_no_echo=True)

    fig = plt.figure(figsize=(12.8, 9.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1])
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.set_xlim(-0.5, maze.n - 0.5)
    ax1.set_ylim(-0.5, maze.n - 0.5)
    ax1.set_aspect("equal")
    ax1.grid(True, ls=":", lw=0.5, color="0.75")
    _draw_walls(ax1, maze)
    sc = ax1.scatter(pts[:, 0], pts[:, 1], c=scan.ranges[scan.echo_mask],
                     cmap="viridis", s=9, zorder=5)
    fig.colorbar(sc, ax=ax1, fraction=0.04, label="距离（格）")
    _draw_mouse(ax1, pose)
    ax1.set_title("世界系点云叠加（点贴墙 / 开口穿透）", fontsize=11)

    ax2 = fig.add_subplot(gs[0, 1], projection="polar")
    plot_polar(scan.ranges, scan.angles, lidar.spec.max_range,
               title="机载单帧 r-θ 视图", ax=ax2)

    ax3 = fig.add_subplot(gs[1, 1])
    grid = occupancy_grid(scan.ranges, scan.angles, pose, lidar.spec.max_range, half=3)
    _draw_occupancy(ax3, grid, pose, 3, maze)
    ax3.set_title("局部占据栅格（D7-B）", fontsize=10)

    fig.suptitle(f"{title} —— 位姿 ({cell[0]},{cell[1]}) 朝 {heading}，"
                 f"{int(scan.echo_mask.sum())}/{lidar.spec.n_beams} 束回波"
                 f"（量程 {lidar.spec.max_range:g} 格）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(OUT, out_name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path, pose, scan


# ---------- 位姿自动挑选 ----------
def pick_poses(maze):
    """（环境侧演示逻辑）墙前 / 走廊纵深 两类验收位姿。

    走廊位姿取纵深 2~8 格的最长 sightline（默认量程 2.0 截不到的"远处墙"，
    用扩展量程 8.0 演示面板呈现）。
    """
    wall_pose = corridor_pose = None
    best_r = -1.0
    for x in range(maze.n):
        for y in range(maze.n):
            for h in ("N", "E", "S", "W"):
                if maze.has_wall(x, y, h):
                    interior = ((h == "N" and y + 1 < maze.n) or (h == "S" and y >= 1)
                                or (h == "E" and x + 1 < maze.n) or (h == "W" and x >= 1))
                    if interior and wall_pose is None:
                        wall_pose = ((x, y), h)
                else:
                    r, _, _ = raycast(maze.has_wall, x + 0.5, y + 0.5,
                                      HEADING_DEG[h], max_range=float(maze.n))
                    if not math.isnan(r) and 2.0 < r <= 8.0 and r > best_r:
                        corridor_pose, best_r = ((x, y), h), r
    return wall_pose, corridor_pose, best_r


def find_ppt_opening(ppt_maze, base_maze):
    """PPT 虚线开口 = PPT 图缺失而底图存在的墙边界；取靠图中心的一个。"""
    openings = []
    for x in range(ppt_maze.n):
        for y in range(ppt_maze.n):
            if base_maze.wall_south[x][y] and not ppt_maze.wall_south[x][y] and y >= 1:
                openings.append(((x, y - 1), "N"))       # 从南侧格北望开口
            if base_maze.wall_west[x][y] and not ppt_maze.wall_west[x][y] and x >= 1:
                openings.append(((x - 1, y), "E"))       # 从东侧格西望开口
    openings.sort(key=lambda ch: -min(ch[0][0], ppt_maze.n - 1 - ch[0][0],
                                      ch[0][1], ppt_maze.n - 1 - ch[0][1]))
    return openings[0]


def ghost_figure(maze, out_name, n_recent=5):
    """点云残影：行进中近 N 帧按时间衰减叠加（呈现层 3）。"""
    history = []
    pose = pose_at((0, 0), "N")
    for act in ["F", "TR", "F", "F", "TL", "F", "F", "TR", "F", "F", "TL", "F"]:
        pose = try_move(pose, act, maze.has_wall).pose
        history.append((pose, LIDAR.scan(maze.has_wall, pose)))

    fig, ax = plt.subplots(figsize=(8.6, 8.4))
    ax.set_xlim(-0.5, maze.n - 0.5)
    ax.set_ylim(-0.5, maze.n - 0.5)
    ax.set_aspect("equal")
    ax.grid(True, ls=":", lw=0.5, color="0.75")
    _draw_walls(ax, maze)
    tail = history[-(n_recent + 1):-1]
    for (p, s), alpha in zip(tail, ghost_alphas(len(history), n_recent=n_recent)):
        pts = scan_to_world(s, p, drop_no_echo=True)
        ax.scatter(pts[:, 0], pts[:, 1], s=7, c="#1f77b4", alpha=alpha, zorder=5)
    cur_pose, cur_scan = history[-1]
    pts = scan_to_world(cur_scan, cur_pose, drop_no_echo=True)
    ax.scatter(pts[:, 0], pts[:, 1], s=12, c="#08318a", zorder=6, label="当前帧")
    ax.plot([p.x for p, _ in history], [p.y for p, _ in history], "-",
            color="#d62728", lw=1.4, alpha=0.65, zorder=5, label="轨迹")
    _draw_mouse(ax, cur_pose)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"点云残影（近 {n_recent} 帧时间衰减叠加）——当前位姿 "
                 f"({int(cur_pose.x - 0.5)},{int(cur_pose.y - 0.5)}) 朝 {cur_pose.heading}",
                 fontsize=11)
    path = os.path.join(OUT, out_name)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------- 探索录像（带实时点云） ----------
def sensor_policy_factory(n=16, seed=0, wander=0.15):
    """仅消费雷达量测 + 自身轨迹的探索策略（R1 合法）。

    依据：info["ranges"]（机体系束角 -180+2i → 正前 i90 / 左 i135 / 右 i45 /
    正后 i0，NaN = 量程内无墙）+ obs["cell"]/obs["heading"]（自身状态）。
    规则：优先走向"可达且未访问"的邻格（直 > 左 > 右 > 后）；无新格时按
    左手定则巡航（左 > 直 > 右 > 后）——顺墙绕行天然带 commitment，不振荡。
    可通行判据（垂直束几何）：束角恰为 90° 整倍数时，回波 ∈ {0.5（共享边
    上有墙）或 ≥1.5（边缘开放、更远处有墙）或 NaN（2.0 内无墙）}——故
    "NaN 或回波 >1.0" = 该向共享边无墙 = 可通行；"=0.5" = 贴边墙。
    """
    rng = random.Random(seed)
    visited = set()
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))          # 与 HEADINGS N/E/S/W 对应
    beam_of = {0: 90, 1: 135, 2: 45, 3: 0}             # 动作 → 机体系束角索引
    turn = {0: 0, 1: -1, 2: +1, 3: +2}                 # 动作 → 朝向下标增量

    def policy(obs, info):
        x, y = obs["cell"] % n, obs["cell"] // n
        visited.add((x, y))
        r = info["ranges"]
        h = obs["heading"]
        if rng.random() < wander:
            return rng.randrange(4)
        fresh, open_ = [], []
        for a in (0, 1, 2, 3):
            d = r[beam_of[a]]
            traversable = np.isnan(d) or d > 1.0        # 见判据说明；0.5=贴边墙
            if traversable:
                nh = (h + turn[a]) % 4
                nxt = (x + dirs[nh][0], y + dirs[nh][1])
                open_.append(a)
                if nxt not in visited:
                    fresh.append(a)
        if fresh:                                       # 新格优先：直 > 左 > 右 > 后
            for a in (0, 1, 2, 3):
                if a in fresh:
                    return a
        for a in (1, 0, 2, 3):                          # 左手定则巡航
            if a in open_:
                return a
        return rng.randrange(4)

    return policy


def demo_gif(env, out_name, max_steps=1500, stride=6):
    frames = record_episode(env, sensor_policy_factory(wander=0.25), max_steps=max_steps)
    animator = EpisodeAnimator(env.scene, frames, max_steps=env.max_steps,
                               title="机器鼠迷宫探索（实时激光点云）")
    path = os.path.join(OUT, out_name)
    animator.save_gif(path, fps=14, stride=stride)
    feeds = sum(1 for f in frames for e in f.events if e.type == EVENT_IMAGE_FEED)
    done = any(e.type == EVENT_TASK_COMPLETE for f in frames for e in f.events)
    return path, frames, feeds, done


def main():
    maze = Maze.default()
    base_maze = Maze.load(os.path.join(os.path.dirname(bamer_env.__file__),
                                       "default_maze.json"))
    wall_pose, corridor_pose, corridor_r = pick_poses(maze)
    opening_cell, opening_heading = find_ppt_opening(maze, base_maze)
    print(f"验收位姿：墙前 {wall_pose} · 开口 {opening_cell}/{opening_heading} · "
          f"走廊 {corridor_pose}（纵深 {corridor_r:.2f} 格）")

    saved = []
    p, pose, scan = _triple_figure(maze, *wall_pose, "验收①：墙前点云贴墙",
                                   "lidar_wall_front.png")
    saved.append(p)
    front_r = scan.ranges[90]       # 正前束（机体系 0°）
    print(f"  墙前：正前束 {front_r:.3f} 格（解析解 0.5）")

    p, pose, scan = _triple_figure(maze, opening_cell, opening_heading,
                                   "验收②：开口处穿透", "lidar_opening.png")
    saved.append(p)
    n_no_echo = int(np.isnan(scan.ranges).sum())
    print(f"  开口：{n_no_echo}/{LIDAR.spec.n_beams} 束量程内无回波（穿过开口望向纵深）")

    p, pose, scan = _triple_figure(
        maze, *corridor_pose, "验收③：走廊纵深 sightline", "lidar_corridor.png",
        lidar=Lidar(LidarSpec(max_range=8.0)))      # 演示扩展量程；D4 训练默认仍为 2.0
    saved.append(p)
    print(f"  走廊：正前束 {scan.ranges[90]:.3f} 格（演示扩展量程 8.0；训练默认 2.0）")

    saved.append(ghost_figure(maze, "lidar_ghost_trail.png"))

    env = BamerEnv(lidar=LidarSpec(), max_steps=1500)
    env.reset(seed=0)
    gif, frames, feeds, done = demo_gif(env, "_demo_pointcloud.gif")
    saved.append(gif)
    print(f"  录像：{len(frames)} 帧 / step {frames[-1].step}，"
          f"图像下发 {feeds} 次，任务完成：{'是' if done else '否（截断于 max_steps）'}")

    for p in saved:
        print(f"已保存: {p}")


if __name__ == "__main__":
    main()
