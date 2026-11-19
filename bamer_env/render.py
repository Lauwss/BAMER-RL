# -*- coding: utf-8 -*-
"""场景渲染：按 PPT 图示风格绘制迷宫、状态机、终点与起点。

图标直接复用 PPT 中提取的原图（docs/_pptx_media/ppt/media/），缺失时退化为
彩色方块 + 中文标注。用法：

    python -m bamer_env.render                # 渲染默认场景存到 docs/_scene_default.png
"""
import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

from .scene import Scene, build_scene

ROOT = os.path.join(os.path.dirname(__file__), "..")
MEDIA = os.path.join(ROOT, "docs", "_pptx_media", "ppt", "media")

# 图标素材（提取自 PPT）
ICONS = {
    "quyuan": os.path.join(MEDIA, "image2.jpeg"),
    "zhaojun": os.path.join(MEDIA, "image4.jpeg"),
    "chuguo": os.path.join(MEDIA, "image3.jpeg"),
    "xihan": os.path.join(MEDIA, "image5.jpeg"),
}
LABELS = {"quyuan": "屈原", "zhaojun": "昭君", "chuguo": "楚国", "xihan": "西汉"}
FALLBACK_COLORS = {"quyuan": "#2e7dd1", "zhaojun": "#c2489a", "chuguo": "#e68a00", "xihan": "#7a5230"}

# 中文字体（Windows: 微软雅黑）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def _draw_icon(ax, name: str, cell, cell_px: float = 75.0, dpi: int = 150):
    """在格子 cell 处画图标，图标约占格子的 80%。

    OffsetImage 显示尺寸 = 原图像素 × zoom × dpi/72，故
    zoom = 目标显示像素 × 72 / (dpi × 原图最长边)。
    """
    x, y = cell
    cx, cy = x + 0.5, y + 0.5
    path = ICONS.get(name)
    if path and os.path.exists(path):
        img = Image.open(path).convert("RGB")
        w, h = img.size
        target_px = 0.8 * cell_px
        zoom = target_px * 72.0 / (dpi * max(w, h))
        oi = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(oi, (cx, cy), frameon=False, zorder=5)
        ax.add_artist(ab)
    else:
        ax.add_patch(plt.Rectangle((x + 0.15, y + 0.15), 0.7, 0.7,
                                   color=FALLBACK_COLORS.get(name, "gray"), zorder=5))
    ax.text(cx, y + 0.06, LABELS.get(name, name), ha="center", va="bottom",
            fontsize=7.5, color="0.25", zorder=6)


def render_scene(scene: Scene, title: str = "机器鼠迷宫场景", save_path: Optional[str] = None):
    """渲染场景，返回 (fig, ax)。save_path 给定时同时保存 PNG。"""
    n = scene.maze.n
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.grid(True, ls=":", lw=0.7, color="0.55", zorder=0)
    ax.tick_params(labelsize=9)

    # 墙线（画在格边界上）
    for x in range(n):
        for y in range(n):
            if scene.maze.wall_south[x][y]:
                ax.plot([x, x + 1], [y, y], "k-", lw=2.6, zorder=3, solid_capstyle="round")
            if scene.maze.wall_west[x][y]:
                ax.plot([x, x], [y, y + 1], "k-", lw=2.6, zorder=3, solid_capstyle="round")
            # 推导的北墙/东墙（避免重复画共享墙，只补最外圈）
            if y == n - 1:
                ax.plot([x, x + 1], [y + 1, y + 1], "k-", lw=2.6, zorder=3, solid_capstyle="round")
            if x == n - 1:
                ax.plot([x + 1, x + 1], [y, y + 1], "k-", lw=2.6, zorder=3, solid_capstyle="round")

    # 起点：向北箭头
    sx, sy = scene.start
    ax.annotate("", xy=(sx + 0.5, sy + 0.72), xytext=(sx + 0.5, sy + 0.18),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.6), zorder=4)

    # 状态机与终点（figsize 9in × dpi150 → 每格约 9*150/17.5 ≈ 77px，取 75）
    cell_px = 75.0
    dpi = fig.dpi
    for name, cell in scene.destinations.items():
        _draw_icon(ax, name, cell, cell_px, dpi)
    for name, cell in scene.state_machines.items():
        _draw_icon(ax, name, cell, cell_px, dpi)

    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"场景图已保存: {save_path}")
    return fig, ax


if __name__ == "__main__":
    scene = build_scene()
    render_scene(scene, save_path=os.path.join(ROOT, "docs", "_scene_default.png"))
