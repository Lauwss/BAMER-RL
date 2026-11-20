# -*- coding: utf-8 -*-
"""激光雷达仿真（M2.2，PLAN.md 阶段二）。

LidarSpec 参数表（D4 默认项，用户可随时推翻）：
    FOV 360°、180 束（2°/束）、量程 2.0 格、噪声默认关闭
    （开启时 σ_range=0.02 格，仅作用在回波距离上）、安装于鼠体中心、
    独立随机流（np.random.default_rng，不污染环境种子流）。

raycast = DDA 格边界穿越（Amanatides & Woo）：按 M2.1 几何约定，墙只存在于
格边界上（零厚度线段），射线求交退化为"逐边界查 has_wall"——比线段求交更快、
无数值病态、可逐行移植 MATLAB。开口 = 缺失的墙边界，射线自然穿过；
角点同时穿越（tx == ty）约定先竖直后水平（确定性对拍前提）。

扫描时机（环境侧职责）：每个 step 末尾机器鼠静止于格中心（规范位姿）采一帧
全扫描；动画呈现用插值位姿 + 上帧点云残影，不在插值位姿上重新扫描。

输出（机体系）：ScanResult.ranges[n]（NaN=无回波）+ angles[n]（机体系束角，
0°=正前方、逆时针为正）+ hit_xy[n,2]（机体系笛卡尔，无回波行为 NaN）；
世界系点云由 pointcloud.to_world 齐次变换（M2.3 坐标链）。
测距噪声作用于回波距离后，hit_xy 由带噪距离重算（量测一致性：hit 与 range
始终自洽，而不是"真值点 + 噪声读数"）。

解耦约定：本模块不 import bamer_env，墙体查询以 wall_query(x, y, direction)->bool
回调注入（环境侧传 Maze.has_wall）；单测以解析解对拍。
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .kinematics import Pose, WallQuery

# 射线起点必须严格位于格内部（规范位姿=格中心满足）；插值中间位姿可能落在
# 格边界上，此时几何退化，直接报错而不是静默给出歧义结果。
_ON_BOUNDARY_EPS = 1e-9


@dataclass(frozen=True)
class LidarSpec:
    """雷达参数表（不可变；训练配置里按字段记录即可复现）。"""
    n_beams: int = 180          # 束数（360°/束距 2°）
    fov_deg: float = 360.0      # 视场角
    max_range: float = 2.0      # 量程（格）
    noise_sigma: float = 0.0    # 测距噪声 σ（格）；0 = 关闭（D4 默认）
    beam0_deg: float = -180.0   # 束 0 的机体系角；束 i 角 = beam0 + i * fov/n_beams

    def __post_init__(self):
        if self.n_beams <= 0:
            raise ValueError(f"n_beams 必须为正，得到 {self.n_beams}")
        if self.fov_deg <= 0:
            raise ValueError(f"fov_deg 必须为正，得到 {self.fov_deg}")
        if self.max_range <= 0:
            raise ValueError(f"max_range 必须为正，得到 {self.max_range}")
        if self.noise_sigma < 0:
            raise ValueError(f"noise_sigma 不能为负，得到 {self.noise_sigma}")

    def angles(self) -> np.ndarray:
        """机体系束角数组（0°=正前方，逆时针为正），覆盖 fov（半开区间）。"""
        return self.beam0_deg + np.arange(self.n_beams) * (self.fov_deg / self.n_beams)

    def to_dict(self) -> dict:
        return {
            "n_beams": self.n_beams, "fov_deg": self.fov_deg,
            "max_range": self.max_range, "noise_sigma": self.noise_sigma,
            "beam0_deg": self.beam0_deg,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LidarSpec":
        return cls(**{k: d[k] for k in
                      ("n_beams", "fov_deg", "max_range", "noise_sigma", "beam0_deg")
                      if k in d})


@dataclass(frozen=True)
class ScanResult:
    """一帧全扫描（机体系）。无回波束：range=NaN、hit_xy 行 NaN。

    R1 说明：ranges/angles/hit_xy 均为雷达量测数据，是算法侧唯一合法的
    环境信息通道之一；不携带任何场景内部结构。
    """
    ranges: np.ndarray    # (n,) 回波距离（格）
    angles: np.ndarray    # (n,) 机体系束角（度）
    hit_xy: np.ndarray    # (n,2) 机体系笛卡尔回波点

    @property
    def echo_mask(self) -> np.ndarray:
        return ~np.isnan(self.ranges)


def raycast(wall_query: WallQuery, ox: float, oy: float, theta_deg: float,
            max_range: float) -> Tuple[float, float, float]:
    """单束 DDA raycast（世界系）。

    参数：wall_query(x, y, direction)->bool 墙体查询；(ox, oy) 射线起点
    （必须严格位于格内部）；theta_deg 世界系射向；max_range 量程。
    返回：(range, hit_x, hit_y)；无回波时 range=NaN、hit 坐标=NaN。

    实现：逐格边界推进——每次取到下一条竖直/水平格边界的参数距离 t，
    先查当前格该侧 has_wall：有墙则在 t 处命中，无墙则跨入邻格继续。
    外边框恒有墙（Maze 构造保证），故循环必然在量程内或边界处终止；
    迭代上限仅作防御（病态 wall_query 时按无回波返回，不死循环）。
    """
    if abs(ox - round(ox)) < _ON_BOUNDARY_EPS or abs(oy - round(oy)) < _ON_BOUNDARY_EPS:
        raise ValueError(f"射线起点必须严格位于格内部，得到 ({ox}, {oy})")
    a = math.radians(theta_deg)
    dx, dy = math.cos(a), math.sin(a)
    ix, iy = int(math.floor(ox)), int(math.floor(oy))
    nan = float("nan")

    for _ in range(4 * int(math.ceil(max_range)) + 8):
        # 下一条竖直/水平格边界的参数距离
        if dx > _ON_BOUNDARY_EPS:
            tx = (ix + 1 - ox) / dx
        elif dx < -_ON_BOUNDARY_EPS:
            tx = (ix - ox) / dx
        else:
            tx = float("inf")
        if dy > _ON_BOUNDARY_EPS:
            ty = (iy + 1 - oy) / dy
        elif dy < -_ON_BOUNDARY_EPS:
            ty = (iy - oy) / dy
        else:
            ty = float("inf")

        t = tx if tx <= ty else ty
        if t > max_range:
            return nan, nan, nan            # 量程内无回波
        if tx <= ty:                        # 角点 tie：先竖直（确定性约定）
            d = "E" if dx > 0 else "W"
            if wall_query(ix, iy, d):
                return t, ox + t * dx, oy + t * dy
            ix += 1 if dx > 0 else -1
        else:                               # 先水平
            d = "N" if dy > 0 else "S"
            if wall_query(ix, iy, d):
                return t, ox + t * dx, oy + t * dy
            iy += 1 if dy > 0 else -1
    return nan, nan, nan                    # 防御分支（良性 wall_query 不会到达）


def _raycast_parallel(wall_query: WallQuery, ox: float, oy: float,
                      theta_degs: np.ndarray, max_range: float,
                      wall_query_batch=None) -> np.ndarray:
    """raycast 的束间并行版本（2026-11-20 向量化提速）。

    所有束从同一起点锁步推进：每轮每束各跨一条格边界，数学全向量化；
    墙查询优先走 wall_query_batch(xs, ys, dirs)->bool 数组（None 时退化为
    逐束标量回调，数学仍向量化）。与标量 raycast 逐位一致的三个保证：
      1. tie 约定同式（tx <= ty 先竖直，同一 IEEE 比较）；
      2. 每束的查询序列不变（每轮恰查一次墙，命中/越界即退出）；
      3. t > max_range 严格大于（恰在量程上的边界仍参与命中判定）。
    """
    n = theta_degs.size
    a = np.radians(np.asarray(theta_degs, dtype=float))
    dx = np.cos(a)
    dy = np.sin(a)
    ix = np.full(n, int(math.floor(ox)), dtype=np.int64)
    iy = np.full(n, int(math.floor(oy)), dtype=np.int64)
    ranges = np.full(n, np.nan)
    finished = np.zeros(n, dtype=bool)
    eps = _ON_BOUNDARY_EPS
    pos_dx, neg_dx = dx > eps, dx < -eps
    pos_dy, neg_dy = dy > eps, dy < -eps

    for _ in range(4 * int(math.ceil(max_range)) + 8):
        act = ~finished
        if not act.any():
            break
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(pos_dx, (ix + 1 - ox) / dx,
                          np.where(neg_dx, (ix - ox) / dx, np.inf))
            ty = np.where(pos_dy, (iy + 1 - oy) / dy,
                          np.where(neg_dy, (iy - oy) / dy, np.inf))
        vert = tx <= ty                     # 角点 tie：先竖直（确定性约定）
        t = np.minimum(tx, ty)
        finished |= act & (t > max_range)   # 量程外无回波（ranges 保持 NaN）
        q = act & ~finished
        if not q.any():
            continue
        dirs = np.where(vert, np.where(pos_dx, "E", "W"),
                        np.where(pos_dy, "N", "S"))[q]
        qx, qy = ix[q], iy[q]
        if wall_query_batch is not None:
            hit = np.asarray(wall_query_batch(qx, qy, dirs), dtype=bool)
        else:
            hit = np.fromiter((wall_query(int(x), int(y), str(d))
                               for x, y, d in zip(qx, qy, dirs)),
                              dtype=bool, count=qx.size)
        hitfull = np.zeros(n, dtype=bool)
        hitfull[np.flatnonzero(q)[hit]] = True
        ranges[hitfull] = t[hitfull]
        finished |= hitfull
        adv = q & ~hitfull
        if adv.any():
            stepx = np.where(vert, np.where(pos_dx, 1, -1), 0)
            stepy = np.where(~vert, np.where(pos_dy, 1, -1), 0)
            ix += stepx * adv
            iy += stepy * adv
    return ranges


def _polar_to_xy(ranges: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """机体系极坐标 → 机体系笛卡尔（NaN 距离 → NaN 行）。"""
    r = np.asarray(ranges, dtype=float)
    a = np.radians(np.asarray(angles, dtype=float))
    xy = np.stack([r * np.cos(a), r * np.sin(a)], axis=-1)
    xy[~np.isfinite(r)] = np.nan
    return xy


class Lidar:
    """雷达实例：持有独立随机流（np.random.default_rng），与环境种子流隔离。

    可复现性：seed 固定 + 同位姿 + 同迷宫 → 逐位一致；Env 在每次 reset 时用
    自身 np_random 流派生重置雷达流（见 bamer_env/env.py），保证同 reset 种子
    的整个 episode 含噪声也逐位可复现。
    """

    def __init__(self, spec: Optional[LidarSpec] = None, seed: Optional[int] = None):
        self.spec = spec if spec is not None else LidarSpec()
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def reseed(self, seed: Optional[int]) -> None:
        """重置独立随机流（Env 每次 reset 调用，从环境种子派生）。"""
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def scan(self, wall_query: WallQuery, pose: Pose,
             wall_query_batch=None) -> ScanResult:
        """在位姿 pose 采一帧全扫描（机体系输出；世界系变换见 pointcloud 模块）。

        世界系射向 = pose.theta_deg + 机体系束角；束 0 对准 -180°（正后方），
        束 i 逆时针均布，其中恰有一束角为 0°（正前方）。
        wall_query_batch(xs, ys, dirs)->bool 数组：可选的批量查墙入口
        （Maze.has_wall_batch），提供时束间 DDA 完全向量化（提速 ~5×）；
        缺省退化为逐束标量回调，结果逐位一致。
        """
        angles = self.spec.angles()
        ranges = _raycast_parallel(wall_query, pose.x, pose.y,
                                   pose.theta_deg + angles,
                                   self.spec.max_range, wall_query_batch)
        if self.spec.noise_sigma > 0:
            # 噪声只作用于回波距离；无论回波多少，噪声流消耗固定 n 个样本，
            # 保证同 seed 噪声序列与场景无关（可复现、可对拍）。
            noise = self._rng.normal(0.0, self.spec.noise_sigma, size=angles.shape[0])
            echo = np.isfinite(ranges)
            ranges[echo] = np.clip(ranges[echo] + noise[echo], 1e-6, self.spec.max_range)
        return ScanResult(ranges=ranges, angles=angles, hit_xy=_polar_to_xy(ranges, angles))
