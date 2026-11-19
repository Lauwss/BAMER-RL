# -*- coding: utf-8 -*-
"""机器鼠运动学模型（M2.1，PLAN.md 阶段二）。

几何约定（D3，用户确认：迷宫为纯离散网格，无内在物理尺度）：
- 全部几何以"格"为单位，格边长 CELL = 1.0；墙体为零厚度线段，位于格边界上；
- 机器鼠为质点，静止时位于所在格中心 (x+0.5, y+0.5)；
- 世界系与 bamer_env 一致：x 向右 0~n-1，y 向上 0~n-1，起点 (0,0) 在左下角；
- 朝向 theta_deg 为世界系角度：东=0°、北=90°、西=180°、南=270°（与 viewer 一致）。

低层动作原语（D5 默认项）：{F 前进一格, TL 左转90°, TR 右转90°, TB 后转180°}，
语义与 BamerEnv 占位动作空间逐位一致（TL/TR/TB 作用在 N,E,S,W 上）。
匀速插值仅供动画呈现（R4：每个动作恰计 1 step，含旋转）；动作完成后逻辑格
= floor(位姿)，格中心规范位姿下即所在格。

碰撞语义（占位，随 M2.6 契约冻结确认）：仅 F 可能碰撞——动作指向有墙边界
= 位置不变 + moved=False + 撞击方向，无物理滑移；惩罚/终止语义由用户奖励设计定。

解耦约定：本模块不 import bamer_env。墙体查询以 wall_query(x, y, direction)->bool
回调注入（环境侧传 Maze.has_wall 即可），保持 bamer_mouse 与场景层单向依赖、
便于阶段四 MATLAB 逐行移植。
"""
import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

# 格边长（D3：无量纲格单位；展示层可选标注 1 格 ≡ 180 mm，不进入仿真核心）
CELL = 1.0

# 低层动作原语（D5）：0=F 1=TL 2=TR 3=TB（与 BamerEnv PRIMITIVES 同序）
PRIMITIVES = ("F", "TL", "TR", "TB")
# 世界系角度（与 viewer.HEADING_DEG 一致）
HEADING_DEG = {"E": 0.0, "N": 90.0, "W": 180.0, "S": 270.0}
_TURN_DELTA_DEG = {"TL": +90.0, "TR": -90.0, "TB": +180.0}
_HEADINGS_ORDER = ("E", "N", "W", "S")   # 角度 0/90/180/270 顺查表

Cell = Tuple[int, int]
WallQuery = Callable[[int, int, str], bool]   # (x, y, direction) -> bool


def normalize_deg(theta_deg: float) -> float:
    """角度归一化到 [0, 360)。"""
    return theta_deg % 360.0


def deg_to_heading(theta_deg: float) -> str:
    """世界系角度 → 最近基本朝向（规范位姿下 theta 恒为 90° 整倍数，精确映射）。"""
    idx = int(round(normalize_deg(theta_deg) / 90.0)) % 4
    return _HEADINGS_ORDER[idx]


def heading_to_deg(heading: str) -> float:
    try:
        return HEADING_DEG[heading]
    except KeyError:
        raise ValueError(f"未知朝向: {heading!r}，可选 {tuple(HEADING_DEG)}") from None


@dataclass(frozen=True)
class Pose:
    """机器鼠位姿：位置（格，世界系）+ 朝向（度，世界系）。"""
    x: float
    y: float
    theta_deg: float

    @property
    def heading(self) -> str:
        return deg_to_heading(self.theta_deg)

    @property
    def cell(self) -> Cell:
        return cell_of(self)

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.theta_deg)


def cell_center(cell: Cell) -> Tuple[float, float]:
    """格 → 中心点坐标 (x+0.5, y+0.5)。"""
    x, y = cell
    return (x + 0.5, y + 0.5)


def pose_at(cell: Cell, heading: str) -> Pose:
    """格中心静止位姿（规范位姿：每个 step 末尾机器鼠均处于该形态）。"""
    cx, cy = cell_center(cell)
    return Pose(cx, cy, heading_to_deg(heading))


def cell_of(pose: Pose) -> Cell:
    """连续位姿 → 逻辑格（格中心规范位姿下即所在格）。"""
    return (int(math.floor(pose.x)), int(math.floor(pose.y)))


_CARDINAL_DIR = {0.0: (1.0, 0.0), 90.0: (0.0, 1.0), 180.0: (-1.0, 0.0), 270.0: (0.0, -1.0)}


def forward_dir(theta_deg: float) -> Tuple[float, float]:
    """朝向的单位方向向量 (dx, dy)。

    规范位姿（theta 为 90° 整倍数）使用精确基向量快照：避免 cos(90°) 的
    IEEE 残差（6e-17）在长序列中累积浮点漂移，并保证与解析解/逐行移植的
    MATLAB 版本逐位一致。
    """
    t = normalize_deg(theta_deg)
    if t in _CARDINAL_DIR:
        return _CARDINAL_DIR[t]
    a = math.radians(t)
    return (math.cos(a), math.sin(a))


def apply_primitive(pose: Pose, primitive: str) -> Pose:
    """纯几何动作演化（不含碰撞判定；F 恒前进一格，带墙判定用 try_move）。"""
    if primitive == "F":
        dx, dy = forward_dir(pose.theta_deg)
        return Pose(pose.x + dx, pose.y + dy, pose.theta_deg)
    if primitive in _TURN_DELTA_DEG:
        return Pose(pose.x, pose.y, normalize_deg(pose.theta_deg + _TURN_DELTA_DEG[primitive]))
    raise ValueError(f"未知动作原语: {primitive!r}，可选 {PRIMITIVES}")


@dataclass(frozen=True)
class MoveOutcome:
    """一次带碰撞判定的动作结果。

    moved=False 表示 F 被墙挡下（位姿不变）；direction 为 F 尝试的世界系朝向
    （旋转动作 None）。上层据此产生 Collision 事件/奖励，本模块不做任务逻辑。
    """
    pose: Pose
    moved: bool
    direction: Optional[str] = None


def try_move(pose: Pose, primitive: str, wall_query: WallQuery) -> MoveOutcome:
    """带碰撞判定的动作演化（wall_query 由环境侧注入，如 Maze.has_wall）。

    碰撞语义（占位，随 M2.6 确认）：动作指向有墙边界 = 位置不变 + Collision，
    计步由 Env 负责（本模块不计数，保证单一职责）。
    """
    if primitive == "F":
        direction = pose.heading
        x, y = cell_of(pose)
        if wall_query(x, y, direction):
            return MoveOutcome(pose=pose, moved=False, direction=direction)
        return MoveOutcome(pose=apply_primitive(pose, "F"), moved=True, direction=direction)
    return MoveOutcome(pose=apply_primitive(pose, primitive), moved=True, direction=None)


def interpolate(pose0: Pose, primitive: str, t: float) -> Pose:
    """动作过程匀速插值（仅动画呈现用，R4 下每个动作仍恰计 1 step）。

    t ∈ [0,1] 为该动作的时间比例：F 位置线性推进，TL/TR/TB 原地匀速旋转。
    t=0 返回 pose0；t=1 返回 apply_primitive(pose0, primitive)。
    注意：插值中间位姿可能跨格边界（非规范位姿），不用于传感器扫描与逻辑判定。
    """
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"插值比例 t 必须在 [0,1]，得到 {t}")
    if primitive == "F":
        dx, dy = forward_dir(pose0.theta_deg)
        return Pose(pose0.x + t * dx, pose0.y + t * dy, pose0.theta_deg)
    if primitive in _TURN_DELTA_DEG:
        return Pose(pose0.x, pose0.y, normalize_deg(pose0.theta_deg + t * _TURN_DELTA_DEG[primitive]))
    raise ValueError(f"未知动作原语: {primitive!r}，可选 {PRIMITIVES}")
