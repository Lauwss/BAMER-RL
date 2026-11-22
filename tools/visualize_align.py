# -*- coding: utf-8 -*-
"""方向对齐度通道（CH_ALIGN）可视化参考。

渲染的是真实实现的 belief.alignment_field()（非重画）——构造一个复刻失败模式
的场景：智能体沿走廊向北扫完一段（prev_boundary → cell），眼前同时存在
"前方新区域 / 东侧支路入口 / 背后来路支路"三类候选，展示通道如何把
"候选是否在我来路方向"写成候选格自身位置上 RF-1 可读的量。

左图：末通道 uint8 场 + 几何标注（当前格 ★ / 上边界 ● / 行进向量 v）+ 墙线。
右图：RF 视角放大——值写在候选格自身位置（1×1 head 感受野 1 即可读），
      而锚点在旧 RF/膨胀 RF 外（通道 15 不可用的原因，即本通道存在的理由）。

用法：python tools/visualize_align.py [输出路径]（默认 output/align_channel_reference.png）
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

from algo.belief import BeliefState

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

N = 16
PREV = (8, 4)      # 上个决策边界位置（走廊南段）
CURR = (8, 8)      # 当前格（走廊北段，刚走完 v=(0,+4) 的一程）
CANDS = {          # 注释用候选格 → (语义标签, 文本锚点)
    (8, 11): ("正前（延续扫掠）", (10.0, 13.0)),
    (12, 8): ("垂直", (8.6, 10.4)),
    (12, 6): ("斜后（东支路）", (11.6, 4.3)),
    (8, 1):  ("回头（来路支路）", (9.4, 0.7)),
}


def build_walls():
    """复刻失败模式的合成场景：走廊 + 东支路 + 背后来路支路 + 北部开阔区。"""
    south = np.zeros((N, N), dtype=np.uint8)
    west = np.zeros((N, N), dtype=np.uint8)
    south[:, 0] = 1; south[:, N - 1] = 1          # 外边框
    west[0, :] = 1; west[N - 1, :] = 1
    for y in range(2, 9):                          # 走廊（x 7..9，y 2..8）左墙
        west[7, y] = 1
    for y in range(2, 9):                          # 走廊右墙（y=6 开口 → 东支路）
        if y != 6:
            west[10, y] = 1
    south[7, 2] = 1; south[9, 2] = 1               # 走廊南封（x=8 开口 → 来路支路）
    west[7, 0] = 1; west[7, 1] = 1                 # 来路支路（x 7..9，y 0..1）
    west[10, 0] = 1; west[10, 1] = 1
    for x in range(10, 14):                        # 东支路（x 10..13，y=6）
        south[x, 6] = 1; south[x, 7] = 1
    west[14, 6] = 1
    west[3, 10] = 1; west[3, 11] = 1; west[3, 12] = 1   # 北区点缀结构
    south[5, 12] = 1; south[6, 12] = 1
    return south, west


def draw_walls(ax, south, west, lw=2.2):
    segs = []
    for x in range(N):
        for y in range(N):
            if south[x, y]:
                segs.append(((x, y), (x + 1, y)))
            if west[x, y]:
                segs.append(((x, y), (x, y + 1)))
    ax.add_collection(LineCollection(segs, colors="black", linewidths=lw, zorder=4))


def cos_from_uint8(val):
    """反推 cos：val = round((cos+1)·127)+1 → cos = (val−1)/127 − 1。"""
    return (val - 1) / 127.0 - 1.0


def main(out_path):
    south, west = build_walls()

    # 用真实实现算场（非重画）
    b = BeliefState(n=N, use_dir=True)
    b.cell = CURR
    b.prev_boundary_cell = PREV
    f = b.alignment_field()

    # 场值 sanity（几何必须与解析值一致，防脚本与实现漂移）
    assert f[8, 11] == 255 and f[8, 1] == 1 and f[12, 8] == 128 and f[8, 8] == 0
    v = (CURR[0] - PREV[0], CURR[1] - PREV[1])
    print(f"[sanity] 东支路 (12,6): uint8={f[12, 6]} "
          f"(cos={cos_from_uint8(int(f[12, 6])):+.3f})")
    print(f"[sanity] 正前 (8,11)={f[8, 11]}  回头 (8,1)={f[8, 1]}  垂直 (12,8)={f[12, 8]}")

    fig = plt.figure(figsize=(15.0, 8.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.22, 1.0], wspace=0.22,
                          left=0.045, right=0.90, top=0.85, bottom=0.14)

    box = dict(boxstyle="round,pad=0.28", fc="white", ec="0.4", alpha=0.94)

    # ---------- 左图：场 + 几何标注 ----------
    ax = fig.add_subplot(gs[0, 0])
    masked = np.ma.masked_equal(f, 0)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9d9d9")
    im = ax.imshow(masked.T, origin="lower", extent=(0, N, 0, N),
                   cmap=cmap, vmin=0, vmax=255, zorder=1)
    draw_walls(ax, south, west)
    ax.set_xticks(range(0, N + 1, 2)); ax.set_yticks(range(0, N + 1, 2))
    ax.grid(color="gray", alpha=0.25, lw=0.4, zorder=2)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("CH_ALIGN：对齐度场 a(c) = cos∠(c − p, v)（网络输入的末通道）", fontsize=11.5)

    # 行进向量与两个锚点
    ax.annotate("", xy=(CURR[0] + 0.5, CURR[1] + 0.5), xytext=(PREV[0] + 0.5, PREV[1] + 0.5),
                arrowprops=dict(arrowstyle="-|>", color="crimson", lw=2.4, shrinkA=8, shrinkB=8), zorder=6)
    ax.plot(PREV[0] + 0.5, PREV[1] + 0.5, "o", ms=9, mfc="white", mec="black", mew=1.6, zorder=7)
    ax.plot(CURR[0] + 0.5, CURR[1] + 0.5, "*", ms=19, mfc="gold", mec="black", mew=1.2, zorder=7)
    ax.annotate("v = p − prev_boundary = (0, +4)\n上个 option 净位移（朝北）",
                xy=(7.9, 6.2), xytext=(0.7, 6.6), fontsize=9.5, color="crimson", bbox=box,
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1.0), zorder=8)
    ax.annotate("上边界 ●（通道 15 单像素）\n当前格 ★（a=0 未定义）",
                xy=(8.5, 7.0), xytext=(0.7, 1.6), fontsize=9, bbox=box,
                arrowprops=dict(arrowstyle="->", lw=0.9), zorder=8)

    # 候选格标注
    for (cx, cy), (label, (tx, ty)) in CANDS.items():
        val = int(f[cx, cy])
        cosv = cos_from_uint8(val)
        ax.plot(cx + 0.5, cy + 0.5, "X", ms=11, mfc="white", mec="black", mew=1.6, zorder=7)
        ax.annotate(f"{label}\n({cx},{cy}) → {val} (cos={cosv:+.2f})",
                    xy=(cx + 0.5, cy + 0.5), xytext=(tx, ty), fontsize=9.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.28", fc="lemonchiffon", ec="0.4", alpha=0.95),
                    arrowprops=dict(arrowstyle="->", lw=1.0), zorder=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    ticks = [1, 64, 128, 192, 255]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(["−1.0", "−0.5", "0.0", "+0.5", "+1.0"])
    cbar.set_label("cos 对齐度（uint8 = round((cos+1)·127)+1）", fontsize=9.5)

    # ---------- 右图：RF 视角放大 ----------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(masked.T, origin="lower", extent=(0, N, 0, N), cmap=cmap, vmin=0, vmax=255, zorder=1)
    draw_walls(ax2, south, west, lw=2.6)
    ax2.set_xticks(range(0, N + 1, 2)); ax2.set_yticks(range(0, N + 1, 2))
    ax2.grid(color="gray", alpha=0.25, lw=0.4, zorder=2)
    ax2.set_xlabel("x"); ax2.set_ylabel("y")
    ax2.set_title("为什么 RF 看得见：值写在候选格自身位置上", fontsize=11.5)
    ax2.set_xlim(2.0, 16.0); ax2.set_ylim(0.0, 14.0)

    cand = (12, 6)
    ccx, ccy = cand[0] + 0.5, cand[1] + 0.5
    ax2.add_patch(Circle((ccx, ccy), 3, fill=False, ls="--", ec="navy", lw=1.6, zorder=5))
    ax2.add_patch(Circle((ccx, ccy), 7, fill=False, ls=":", ec="steelblue", lw=1.6, zorder=5))
    ax2.annotate("旧 RF=7（半径 3）", xy=(ccx - 3 * 0.7071, ccy + 3 * 0.7071),
                 xytext=(2.3, 11.5), fontsize=9, color="navy",
                 arrowprops=dict(arrowstyle="->", color="navy", lw=0.9), zorder=6)
    ax2.annotate("膨胀后 RF=15（半径 7）", xy=(ccx, ccy + 7),
                 xytext=(2.3, 13.2), fontsize=9, color="steelblue",
                 arrowprops=dict(arrowstyle="->", color="steelblue", lw=0.9), zorder=6)
    ax2.plot(ccx, ccy, "X", ms=13, mfc="white", mec="black", mew=1.8, zorder=7)
    ax2.annotate(f"候选格 ({cand[0]},{cand[1]}) 自身 a={int(f[cand])}\n"
                 "1×1 head 感受野 1 即可读——比较已由特征层预计算",
                 xy=(ccx, ccy), xytext=(2.4, 7.8), fontsize=9.5, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", fc="lemonchiffon", ec="0.4", alpha=0.95),
                 arrowprops=dict(arrowstyle="->", lw=1.0), zorder=8)
    ax2.plot(PREV[0] + 0.5, PREV[1] + 0.5, "o", ms=9, mfc="white", mec="black", mew=1.6, zorder=7)
    ax2.annotate("锚点 ●：距该候选 4.5 格 > 半径 3\n→ 旧 RF 外不可见（通道 15 的病；\n对齐场即其可用替代）",
                 xy=(PREV[0] + 0.9, PREV[1] + 0.5), xytext=(5.6, 0.6), fontsize=9.5, color="firebrick",
                 bbox=dict(boxstyle="round,pad=0.28", fc="mistyrose", ec="firebrick", alpha=0.92),
                 arrowprops=dict(arrowstyle="->", color="firebrick", lw=1.0, shrinkA=2, shrinkB=6), zorder=8)

    fig.suptitle("方向连续性通道（dir_channel / CH_ALIGN）可视化参考——场景复刻 repeat 主成分：扫完走廊后「前方 vs 来路」的距离打平候选",
                 fontsize=12.5, y=0.965)
    fig.text(0.5, 0.035,
             "语义：+1=正前（延续扫掠）→ −1=回头（来路方位）。灰 = 0（未定义：episode 首边界 / 零位移 option / 当前格自身）——全零=未知约定。",
             ha="center", fontsize=9.5, color="0.25")
    fig.text(0.5, 0.012,
             "排序在距离打平时由本通道裁决「回头成本不对称」；门控 cfg.dir_channel；布局 dist 在前、align 恒居末位（chan_scale /255）。",
             ha="center", fontsize=9.5, color="0.25")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join("output", "align_channel_reference.png")
    main(out)
