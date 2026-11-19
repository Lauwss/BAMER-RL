# -*- coding: utf-8 -*-
"""展示层（M1.3，PLAN.md 阶段一）——成果呈现要求的实现载体。

| 成果要求           | 本模块实现                                              |
| ------------------ | ------------------------------------------------------- |
| 动态呈现探索路径   | matplotlib 逐帧动画（墙线 + 鼠位姿 + 轨迹尾迹）         |
| 实时步数           | HUD（主视图标题栏：step / 朝向 / 识别 / 派送）          |
| 实时激光点云       | `Frame.points` 挂点已接通（M2.2/M2.3）：接雷达的 Env 经 info["points"] 自动随帧叠加 |
| 状态机识别图像展示 | ImageFeed 事件触发的右侧图像面板                        |
| 最终最优路线/录屏  | 阶段四（依赖 M2.4 EpisodeRecord 回放数据）              |

用法示例：
    from bamer_env.env import BamerEnv
    from bamer_env.viewer import EpisodeAnimator, record_episode

    env = BamerEnv(max_steps=800)
    frames = record_episode(env, my_policy)          # my_policy(obs, info) -> action
    animator = EpisodeAnimator(env.scene, frames, max_steps=env.max_steps)
    animator.save_gif("docs/_demo_exploration.gif", fps=12, stride=4)

说明：状态机标记在"识别后"才显示（呈现探索发现过程）；终点图标为题目公开
信息始终显示。展示层供环境侧/汇报使用，不进入算法侧数据流（R1 不受影响）。
"""
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Polygon
from PIL import Image

from .interaction import EVENT_ARRIVED, EVENT_IMAGE_FEED
from .render import ICONS, LABELS
from .scene import Scene

ROOT = os.path.join(os.path.dirname(__file__), "..")

# 中文字体（Windows: 微软雅黑；render.py 已设，此处幂等兜底）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HEADING_DEG = {"E": 0.0, "N": 90.0, "W": 180.0, "S": 270.0}
MACHINE_COLORS = {"quyuan": "#2e7dd1", "zhaojun": "#c2489a"}
DELIVERED_COLOR = "#2e9e5b"


@dataclass
class Frame:
    """逐帧记录：位姿 + 该步事件（+ 可选点云挂点）。轨迹由帧序列推导，无需冗余存储。"""
    step: int
    x: float                     # 世界系格中心坐标 (x+0.5, y+0.5)
    y: float
    heading: str
    heading_deg: float
    events: tuple = ()
    points: Optional[np.ndarray] = None   # 世界系点云 (N,2)——阶段二 M2.2/M2.3 接通


def _frame_from_env(env, step: int, info: dict) -> Frame:
    x, y = env.pose_cell
    return Frame(
        step=step, x=x + 0.5, y=y + 0.5,
        heading=env.heading, heading_deg=HEADING_DEG[env.heading],
        events=tuple(info.get("events", ())),
        points=info.get("points"),   # M2.2/M2.3 接通：info 中的世界系点云（未接雷达为 None）
    )


def record_episode(env, policy: Callable, max_steps: Optional[int] = None,
                   max_frames: int = 5000) -> List[Frame]:
    """运行一个完整 episode（reset → terminated/truncated）并逐帧记录。

    policy(obs, info) -> action：仅用于展示层演示（如占位随机漫步），
    不是算法基线——正式策略由阶段三 RL 产出。
    """
    obs, info = env.reset()
    frames = [_frame_from_env(env, info.get("steps", 0), info)]
    limit = env.max_steps if max_steps is None else int(max_steps)
    while len(frames) < max_frames:
        if frames[-1].step >= limit:
            break
        action = policy(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(_frame_from_env(env, info["steps"], info))
        if terminated or truncated:
            break
    return frames


# ---------- 静态绘制（与 render.py 风格一致，供动画/快照共用） ----------
def _setup_maze_axes(ax, n: int):
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.grid(True, ls=":", lw=0.7, color="0.55", zorder=0)
    ax.tick_params(labelsize=8)


def _draw_walls(ax, maze):
    n = maze.n
    for x in range(n):
        for y in range(n):
            if maze.wall_south[x][y]:
                ax.plot([x, x + 1], [y, y], "k-", lw=2.2, zorder=3, solid_capstyle="round")
            if maze.wall_west[x][y]:
                ax.plot([x, x], [y, y + 1], "k-", lw=2.2, zorder=3, solid_capstyle="round")
            if y == n - 1:
                ax.plot([x, x + 1], [y + 1, y + 1], "k-", lw=2.2, zorder=3, solid_capstyle="round")
            if x == n - 1:
                ax.plot([x + 1, x + 1], [y, y + 1], "k-", lw=2.2, zorder=3, solid_capstyle="round")


def _draw_icon(ax, name: str, cell, fraction: float = 0.8):
    """终点/状态机图标（PPT 素材，缺失时退化为彩色方块）。

    用数据坐标 extent 铺在格子中央（与 dpi 无关、尺寸精确）；
    注意 imshow 会改坐标范围，调用方画完图标后需恢复 xlim/ylim。
    """
    x, y = cell
    path = ICONS.get(name)
    if path and os.path.exists(path):
        img = np.asarray(Image.open(path).convert("RGB"))
        m = (1.0 - fraction) / 2.0
        ax.imshow(img, extent=(x + m, x + 1 - m, y + m, y + 1 - m),
                  origin="upper", interpolation="bilinear", zorder=4)
    else:
        ax.add_patch(plt.Rectangle((x + 0.15, y + 0.15), 0.7, 0.7, color="0.6", zorder=4))
    ax.text(x + 0.5, y + 0.06, LABELS.get(name, name), ha="center", va="bottom",
            fontsize=7.5, color="0.25", zorder=6)


def _draw_machine_markers(ax, scene: Scene, visible_names: Sequence[str] = ()) -> Dict[str, tuple]:
    """状态机标记（识别后显示；呈现"探索发现"过程）。返回 {名称: (圆圈, 标签)} 供动画切换。"""
    artists = {}
    for name, (mx, my) in scene.state_machines.items():
        circle = Circle((mx + 0.5, my + 0.5), 0.36, fill=False,
                        color=MACHINE_COLORS.get(name, "0.3"), lw=2.4,
                        zorder=6, visible=name in visible_names)
        label = ax.text(mx + 0.5, my + 0.56, LABELS.get(name, name), ha="center", va="bottom",
                        fontsize=8, color=MACHINE_COLORS.get(name, "0.3"), fontweight="bold",
                        zorder=7, visible=name in visible_names)
        ax.add_patch(circle)
        artists[name] = (circle, label)
    return artists


def _draw_delivery_rings(ax, scene: Scene, delivered_names: Sequence[str] = ()) -> Dict[str, Circle]:
    rings = {}
    for dname, (dx, dy) in scene.destinations.items():
        ring = Circle((dx + 0.5, dy + 0.5), 0.44, fill=False, color=DELIVERED_COLOR,
                      lw=2.6, ls="--", zorder=6, visible=dname in delivered_names)
        ax.add_patch(ring)
        rings[dname] = ring
    return rings


def _mouse_triangle(cx: float, cy: float, deg: float) -> List[Tuple[float, float]]:
    """机器鼠三角标记（指向 heading_deg）。"""
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    pts = ((0.26, 0.0), (-0.17, 0.16), (-0.10, 0.0), (-0.17, -0.16))
    return [(cx + px * ca - py * sa, cy + px * sa + py * ca) for px, py in pts]


def _hud(ax, scene: Scene, step: int, heading: str, identified, delivered,
         max_steps: Optional[int], title: str = "机器鼠迷宫探索"):
    ids = "/".join(LABELS.get(m, m) for m in sorted(identified)) or "—"
    dels = "/".join(LABELS.get(m, m) for m in sorted(delivered)) or "—"
    limit = f"/{max_steps}" if max_steps is not None else ""
    ax.set_title(f"{title} | 实时步数 step {step}{limit} | 朝向 {heading} "
                 f"| 已识别: {ids} | 已派送: {dels}", fontsize=11)


# ---------- 单帧快照（BamerEnv.render 使用） ----------
def draw_snapshot(scene: Scene, pose_cell, heading: str, trail_cells, steps: int,
                  max_steps: Optional[int] = None, identified=(), delivered=(),
                  points: Optional[np.ndarray] = None, title: str = "BamerEnv 当前帧"):
    fig, ax = plt.subplots(figsize=(8.6, 8.2))
    _setup_maze_axes(ax, scene.maze.n)
    _draw_walls(ax, scene.maze)
    for name, cell in scene.destinations.items():
        _draw_icon(ax, name, cell)
    ax.set_xlim(-0.5, scene.maze.n - 0.5)   # 恢复 imshow 改动的坐标范围
    ax.set_ylim(-0.5, scene.maze.n - 0.5)
    _draw_machine_markers(ax, scene, visible_names=identified)
    # 派送环按终点名标注（delivered 集合存的是状态机名，经 delivery_map 映射）
    delivered_dests = {scene.delivery_map[m] for m in delivered if m in scene.delivery_map}
    _draw_delivery_rings(ax, scene, delivered_names=delivered_dests)
    if trail_cells:
        tx = [x + 0.5 for x, _ in trail_cells]
        ty = [y + 0.5 for _, y in trail_cells]
        ax.plot(tx, ty, "-", color="#d62728", lw=1.4, alpha=0.65, zorder=5)
    if points is not None and len(points):
        ax.scatter(points[:, 0], points[:, 1], s=5, c="#1f77b4", alpha=0.8, zorder=6)
    px, py = pose_cell
    ax.add_patch(Polygon(_mouse_triangle(px + 0.5, py + 0.5, HEADING_DEG[heading]),
                         closed=True, color="#d62728", zorder=7))
    _hud(ax, scene, steps, heading, identified, delivered, max_steps, title=title)
    fig.tight_layout()
    return fig


# ---------- 逐帧动画 ----------
class EpisodeAnimator:
    """探索过程动画：主视图（墙线/鼠/轨迹/步数 HUD/点云挂点）+ 识别图像面板。"""

    def __init__(self, scene: Scene, frames: Sequence[Frame],
                 max_steps: Optional[int] = None, title: str = "机器鼠迷宫探索回放"):
        self.scene = scene
        self.frames = list(frames)
        self.max_steps = max_steps
        self.title = title

        self.fig = plt.figure(figsize=(11.5, 8.6))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[3.4, 1], wspace=0.05)
        self.ax = self.fig.add_subplot(gs[0, 0])
        self.ax_img = self.fig.add_subplot(gs[0, 1])

        # 主视图静态层
        _setup_maze_axes(self.ax, scene.maze.n)
        _draw_walls(self.ax, scene.maze)
        for name, cell in scene.destinations.items():
            _draw_icon(self.ax, name, cell)
        self.ax.set_xlim(-0.5, scene.maze.n - 0.5)   # 恢复 imshow 改动的坐标范围
        self.ax.set_ylim(-0.5, scene.maze.n - 0.5)
        self._machines = _draw_machine_markers(self.ax, scene)       # 识别后显示
        self._rings = _draw_delivery_rings(self.ax, scene)           # 派送后显示

        # 动态层：轨迹尾迹 / 点云挂点 / 鼠标记
        self._trail, = self.ax.plot([], [], "-", color="#d62728", lw=1.4, alpha=0.65, zorder=5)
        self._points = self.ax.scatter([], [], s=5, c="#1f77b4", alpha=0.8, zorder=6, visible=False)
        self._mouse = Polygon(_mouse_triangle(self.frames[0].x, self.frames[0].y,
                                              self.frames[0].heading_deg),
                              closed=True, color="#d62728", zorder=7)
        self.ax.add_patch(self._mouse)

        # 识别图像面板（ImageFeed 事件触发）
        self.ax_img.set_xlim(0, 1)
        self.ax_img.set_ylim(0, 1)
        self.ax_img.set_xticks([])
        self.ax_img.set_yticks([])
        self._img_artist = None
        self._img_placeholder = self.ax_img.text(0.5, 0.55, "等待识别…\n（进入状态机格后\n此处显示下发图像）",
                                                 ha="center", va="center", fontsize=10, color="0.35")
        self.ax_img.set_title("摄像头 / 识别图像", fontsize=11)

        self._update(0)

    # ---------- 动画 ----------
    def _update(self, i: int):
        frame = self.frames[i]
        # 累积事件状态（识别 / 派送 / 最新图像）
        identified, delivered, latest_feed = set(), set(), None
        for f in self.frames[:i + 1]:
            for ev in f.events:
                if ev.type == EVENT_IMAGE_FEED:
                    identified.add(ev.payload["machine"])
                    latest_feed = ev
                elif ev.type == EVENT_ARRIVED:
                    delivered.add(ev.payload["destination"])
        for name, (circle, label) in self._machines.items():
            circle.set_visible(name in identified)
            label.set_visible(name in identified)
        for dname, ring in self._rings.items():
            ring.set_visible(dname in delivered)

        # 轨迹尾迹（由帧序列推导）
        self._trail.set_data([f.x for f in self.frames[:i + 1]],
                             [f.y for f in self.frames[:i + 1]])
        # 点云挂点（阶段二接通后生效）
        if frame.points is not None and len(frame.points):
            self._points.set_offsets(frame.points)
            self._points.set_visible(True)
        else:
            self._points.set_visible(False)
        # 鼠位姿
        self._mouse.set_xy(_mouse_triangle(frame.x, frame.y, frame.heading_deg))
        # HUD 实时步数
        _hud(self.ax, self.scene, frame.step, frame.heading, identified, delivered,
             self.max_steps, title=self.title)
        # 图像面板：显示最近一次 ImageFeed（保持展示"当前识别的图像"）
        if latest_feed is not None:
            path = latest_feed.payload.get("image") or ""
            if path and os.path.exists(path):
                img = np.asarray(Image.open(path).convert("RGB"))
                if self._img_artist is None:
                    self._img_artist = self.ax_img.imshow(img)   # origin=upper，原图方向显示
                else:
                    self._img_artist.set_data(img)
                # 显式设定像素坐标范围（__init__ 的 set_xlim 已关闭 autoscale）
                self.ax_img.set_xlim(0, img.shape[1])
                self.ax_img.set_ylim(img.shape[0], 0)
            if self._img_placeholder is not None:
                self._img_placeholder.set_visible(False)
            self.ax_img.set_title(
                f"摄像头 / 识别图像\nstep {latest_feed.step} 识别：{latest_feed.payload.get('label', '')}",
                fontsize=11)
        return []

    def _animation(self, stride: int, interval: float) -> FuncAnimation:
        idxs = list(range(0, len(self.frames), stride))
        if idxs[-1] != len(self.frames) - 1:
            idxs.append(len(self.frames) - 1)
        return FuncAnimation(self.fig, self._update, frames=idxs,
                             init_func=lambda: (self._update(0), [])[1],
                             interval=interval, blit=False)

    def save_gif(self, path: str, fps: int = 12, stride: int = 1, dpi: int = 110) -> str:
        """导出探索录像 GIF（验收要求：输出 1 段完整探索录像）。"""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        anim = self._animation(stride=stride, interval=1000.0 / fps)
        anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi,
                  savefig_kwargs={"facecolor": "white"})
        return path
