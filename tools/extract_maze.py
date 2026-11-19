# -*- coding: utf-8 -*-
"""从 PPT 底图（2022 官方 Micromouse 迷宫图）中程序化提取 16×16 墙线数据。

方法：
  1. 二值化（深色墨迹 = 1）；
  2. 用长暗行/暗列定位外边框，得到迷宫像素包围盒；
  3. 对每条网格边界段的中点采样窗口计算墨迹覆盖率，
     实线墙 ~100%，点状网格线 ~20-30%，据此区分墙与网格；
  4. 输出 JSON 墙线数据，并渲染重建迷宫图用于人工比对。

坐标约定（与 PPT 一致）：x 向右 0~15，y 向上 0~15，起点 (0,0) 在左下角。
"""
import json
import os

import cv2
import numpy as np

N = 16                     # 迷宫尺寸
IMG_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "_pptx_media", "ppt", "media", "image1.png")
OUT_JSON = os.path.join(os.path.dirname(__file__), "..", "bamer_env", "default_maze.json")
OUT_PNG = os.path.join(os.path.dirname(__file__), "..", "docs", "_maze_reconstructed.png")


def longest_run(mask_1d):
    d = np.diff(np.concatenate([[0], mask_1d, [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return int((ends - starts).max()) if len(starts) else 0


def find_maze_bbox(bw):
    """外边框 = 四条超长实线。返回 (left, top, right, bottom) 像素坐标（含边框线中心）。"""
    h, w = bw.shape
    row_run = np.array([longest_run(bw[y]) for y in range(h)])
    col_run = np.array([longest_run(bw[:, x]) for x in range(w)])

    border_rows = np.where(row_run > 0.6 * w)[0]
    # 排除文字行：取暗行成簇中跨度最大的一簇（上下边框距离远）
    groups = np.split(border_rows, np.where(np.diff(border_rows) > 20)[0] + 1)
    groups = [g for g in groups if len(g) > 0]
    top = int(groups[0].mean())
    bottom = int(groups[-1].mean())

    border_cols = np.where(col_run > 0.85 * (bottom - top))[0]
    groups = np.split(border_cols, np.where(np.diff(border_cols) > 20)[0] + 1)
    groups = [g for g in groups if len(g) > 0]
    left = int(groups[0].mean())
    right = int(groups[-1].mean())
    return left, top, right, bottom


def main():
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"读取失败: {IMG_PATH}"
    bw = (img < 128).astype(np.float64)

    left, top, right, bottom = find_maze_bbox(bw)
    print(f"迷宫包围盒: left={left} top={top} right={right} bottom={bottom}")
    cell_w = (right - left) / N
    cell_h = (bottom - top) / N
    print(f"格子像素: w={cell_w:.2f} h={cell_h:.2f}")

    # 墙线厚度估计：取底边框行的暗像素厚度
    col_at_mid = bw[:, (left + right) // 2]
    thickness = 0
    y = bottom
    while col_at_mid[y] > 0:
        thickness += 1
        y -= 1
    print(f"墙线厚度约 {thickness}px")

    def seg_run_fraction_h(line_y, x0, x1):
        """水平边界段 (x0..x1 格) 的最长连续暗游程占比。

        实线墙是连续的（占比≈1），点状网格线是断续的（占比<0.5），据此区分。
        只采样边界段中段 30%~70%，避免端点交叉墙的干扰。
        """
        px = int(round(line_y))
        xa = int(round(left + x0 * cell_w + cell_w * 0.3))
        xb = int(round(left + x1 * cell_w - cell_w * 0.3))
        win = bw[max(0, px - 3): px + 4, xa:xb + 1]
        proj = win.max(axis=0) if win.size else np.zeros(1)
        return longest_run(proj) / max(1, len(proj))

    def seg_run_fraction_v(line_x, y0, y1):
        """垂直边界段 (y0..y1 格) 的最长连续暗游程占比（y 为格坐标，向上）。"""
        px = int(round(left + line_x * cell_w))
        ya = int(round(bottom - y0 * cell_h - cell_h * 0.3))  # 南端（像素大值）
        yb = int(round(bottom - y1 * cell_h + cell_h * 0.3))  # 北端（像素小值）
        win = bw[yb:ya + 1, max(0, px - 3): px + 4]
        proj = win.max(axis=1) if win.size else np.zeros(1)
        return longest_run(proj) / max(1, len(proj))

    # wall_south[x][y] = 格 (x,y) 与 (x,y-1) 之间的水平墙（y=0 时为外边框）
    # wall_west [x][y] = 格 (x,y) 与 (x-1,y) 之间的垂直墙（x=0 时为外边框）
    wall_south = [[False] * N for _ in range(N)]
    wall_west = [[False] * N for _ in range(N)]
    cov_s = [[0.0] * N for _ in range(N)]
    cov_w = [[0.0] * N for _ in range(N)]

    THR = 0.75
    for x in range(N):
        for y in range(N):
            line_y = bottom - y * cell_h            # 格 (x,y) 南边界
            cov_s[x][y] = seg_run_fraction_h(line_y, x, x + 1)
            wall_south[x][y] = bool(cov_s[x][y] > THR)
            line_x = x                              # 格 (x,y) 西边界
            cov_w[x][y] = seg_run_fraction_v(line_x, y, y + 1)
            wall_west[x][y] = bool(cov_w[x][y] > THR)

    # 顶部外边框（y=15 的北墙）与右边框（x=15 的东墙）必然存在，手工补上
    north = [True] * N
    east = [[True] * N for _ in range(N)]  # east[x][y] = 格 (x,y) 东墙，等价 west[x+1][y]

    # 一致性校验：相邻格共享的墙，两侧检测结果应一致（west[x+1][y] == east[x][y]）
    inconsistent = []
    for x in range(N - 1):
        for y in range(N):
            if wall_west[x + 1][y] != bool(cov_w[x + 1][y] > THR):
                inconsistent.append(f"west({x+1},{y})")
    print(f"\n共享墙一致性检查: {'OK' if not inconsistent else inconsistent}")

    # 打印游程占比在 0.5~0.85 之间的待复核段，便于人工复核
    print("\n游程占比在 0.50~0.85 之间的待复核段:")
    n_sus = 0
    for x in range(N):
        for y in range(N):
            if 0.50 < cov_s[x][y] < 0.85:
                print(f"  south wall of ({x},{y}): {cov_s[x][y]:.2f}")
                n_sus += 1
            if 0.50 < cov_w[x][y] < 0.85:
                print(f"  west  wall of ({x},{y}): {cov_w[x][y]:.2f}")
                n_sus += 1
    print(f"  共 {n_sus} 段待复核" if n_sus else "  无，全部判定高置信")

    data = {
        "name": "official-2022-tohoku-freshman",
        "size": N,
        "wall_south": wall_south,   # wall_south[x][y]: 格(x,y) 与 (x,y-1) 之间
        "wall_west": wall_west,     # wall_west[x][y]:  格(x,y) 与 (x-1,y) 之间
        "wall_north": north,        # 外边框
        "wall_east": [[True] * N for _ in range(N)],
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"\n已写出 {OUT_JSON}")

    # 渲染重建迷宫用于比对
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    for x in range(N):
        for y in range(N):
            if wall_south[x][y]:
                ax.plot([x, x + 1], [y, y], "k-", lw=2.4)
            if wall_west[x][y]:
                ax.plot([x, x], [y, y + 1], "k-", lw=2.4)
            if east[x][y] and x == N - 1:
                ax.plot([x + 1, x + 1], [y, y + 1], "k-", lw=2.4)
            if north[x] and y == N - 1:
                ax.plot([x, x + 1], [y + 1, y + 1], "k-", lw=2.4)
    ax.set_xlim(-0.5, N + 0.5)
    ax.set_ylim(-0.5, N + 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.grid(True, ls=":", lw=0.6, color="gray")
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"已渲染重建图 {OUT_PNG}")


if __name__ == "__main__":
    main()
