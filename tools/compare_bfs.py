# -*- coding: utf-8 -*-
"""对比开孔前后迷宫的 BFS 距离，并把最短路径叠加到场景图上，核实提取正确性。"""
import os
import sys
from collections import deque

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bamer_env import Maze  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def bfs_path(maze, src, dst):
    q = deque([src])
    prev = {src: None}
    while q:
        cur = q.popleft()
        if cur == dst:
            path = []
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            return path[::-1]
        for nb in maze.neighbors(*cur):
            if nb not in prev:
                prev[nb] = cur
                q.append(nb)
    return None


orig = Maze.load(os.path.join(ROOT, "bamer_env", "default_maze.json"))
ppt = Maze.load(os.path.join(ROOT, "bamer_env", "default_maze_ppt.json"))

for name, m in (("原始官方图（未开孔）", orig), ("PPT 场景（开孔后）", ppt)):
    for goal in [(8, 8), (7, 7)]:
        p = bfs_path(m, (0, 0), goal)
        n = len(p) - 1 if p else None
        # 折数（转弯次数）
        folds = 0
        if p:
            for i in range(1, len(p) - 1):
                v1 = (p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1])
                v2 = (p[i + 1][0] - p[i][0], p[i + 1][1] - p[i][1])
                folds += v1 != v2
        print(f"{name}  起点(0,0)->{goal}: {n} 步 {folds} 折")

# 开孔后到 (8,8) 的路径叠加渲染
p = bfs_path(ppt, (0, 0), (8, 8))
fig, ax = plt.subplots(figsize=(9, 9))
n = 16
for x in range(n):
    for y in range(n):
        if ppt.wall_south[x][y]:
            ax.plot([x, x + 1], [y, y], "k-", lw=2.4, solid_capstyle="round")
        if ppt.wall_west[x][y]:
            ax.plot([x, x], [y, y + 1], "k-", lw=2.4, solid_capstyle="round")
        if y == n - 1:
            ax.plot([x, x + 1], [y + 1, y + 1], "k-", lw=2.4, solid_capstyle="round")
        if x == n - 1:
            ax.plot([x + 1, x + 1], [y, y + 1], "k-", lw=2.4, solid_capstyle="round")
xs, ys = zip(*p)
ax.plot([x + 0.5 for x in xs], [y + 0.5 for y in ys], "r-o", lw=1.8, ms=3.5, zorder=5)
ax.plot(xs[0] + 0.5, ys[0] + 0.5, "g^", ms=11, zorder=6, label="start")
ax.plot(xs[-1] + 0.5, ys[-1] + 0.5, "b*", ms=14, zorder=6, label="goal(8,8)")
ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(-0.5, n - 0.5)
ax.set_aspect("equal"); ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.grid(True, ls=":", lw=0.6)
ax.legend()
ax.set_title(f"PPT场景 最短路 (0,0)->(8,8): {len(p)-1} 步")
out = os.path.join(ROOT, "docs", "_shortest_path_check.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"\n路径图已保存 {out}")
