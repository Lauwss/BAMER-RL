# -*- coding: utf-8 -*-
"""诊断迷宫连通性：找出从起点不可达的格子，并渲染成图。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bamer_env import Maze
from bamer_env.maze import DIRS

m = Maze.default()
reach = m.reachable_from((0, 0))
unreach = [(x, y) for x in range(m.n) for y in range(m.n) if (x, y) not in reach]
print(f"不可达格子 {len(unreach)} 个: {unreach[:30]}{' ...' if len(unreach) > 30 else ''}")

# 渲染：可达区白色，不可达区红色
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 7))
for x in range(m.n):
    for y in range(m.n):
        if (x, y) not in reach:
            ax.add_patch(plt.Rectangle((x, y), 1, 1, color="red", alpha=0.55, zorder=4))
        if m.wall_south[x][y]:
            ax.plot([x, x + 1], [y, y], "k-", lw=2.2, solid_capstyle="round")
        if m.wall_west[x][y]:
            ax.plot([x, x], [y, y + 1], "k-", lw=2.2, solid_capstyle="round")
        if y == m.n - 1:
            ax.plot([x, x + 1], [y + 1, y + 1], "k-", lw=2.2, solid_capstyle="round")
        if x == m.n - 1:
            ax.plot([x + 1, x + 1], [y, y + 1], "k-", lw=2.2, solid_capstyle="round")
ax.set_xlim(-0.5, m.n - 0.5)
ax.set_ylim(-0.5, m.n - 0.5)
ax.set_aspect("equal")
ax.set_xticks(range(m.n)); ax.set_yticks(range(m.n))
ax.grid(True, ls=":", lw=0.6)
out = os.path.join(os.path.dirname(__file__), "..", "docs", "_connectivity_debug.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"已保存 {out}")
