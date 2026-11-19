# -*- coding: utf-8 -*-
"""bamer_mouse — 机器鼠层（PLAN.md 阶段二）。

组成（自底向上）：
    kinematics.M2.1   位姿/动作原语/碰撞/匀速插值（格单位，D3/D5）
    lidar.M2.2        LidarSpec 参数表 + DDA raycast + 全扫描（D4 默认项）
    pointcloud.M2.3   坐标链（极坐标→机体系→世界系）+ 局部占据栅格（D7-B）+ 呈现
    dataio.M2.4       EpisodeRecord 统一格式与 npz+json 存取
    camera.M2.5       识别挂接：Classifier 协议 + GT 直通占位（D9 待定）

铁律自检：本包不 import bamer_env——墙体查询以 wall_query(x, y, direction)->bool
回调注入，传感器输出（量测/点云/栅格）即算法侧唯一合法信息通道（AGENT.md R1）。
"""
from .camera import CameraUnit, Classifier, GTPassthroughClassifier
from .dataio import EpisodeRecord, StepRecord, maze_hash
from .kinematics import (
    PRIMITIVES,
    Pose,
    MoveOutcome,
    apply_primitive,
    cell_center,
    cell_of,
    deg_to_heading,
    forward_dir,
    heading_to_deg,
    interpolate,
    normalize_deg,
    pose_at,
    try_move,
)
from .lidar import Lidar, LidarSpec, ScanResult, raycast
from .pointcloud import (
    OCC_BLOCKED,
    OCC_FREE,
    OCC_UNKNOWN,
    draw_points,
    ghost_alphas,
    occupancy_grid,
    plot_polar,
    polar_to_cartesian,
    scan_to_world,
    to_world,
    walk_cells,
)

__all__ = [
    # M2.1
    "PRIMITIVES", "Pose", "MoveOutcome", "apply_primitive", "cell_center", "cell_of",
    "deg_to_heading", "forward_dir", "heading_to_deg", "interpolate", "normalize_deg",
    "pose_at", "try_move",
    # M2.2
    "Lidar", "LidarSpec", "ScanResult", "raycast",
    # M2.3
    "OCC_BLOCKED", "OCC_FREE", "OCC_UNKNOWN", "draw_points", "ghost_alphas",
    "occupancy_grid", "plot_polar", "polar_to_cartesian", "scan_to_world",
    "to_world", "walk_cells",
    # M2.4
    "EpisodeRecord", "StepRecord", "maze_hash",
    # M2.5
    "CameraUnit", "Classifier", "GTPassthroughClassifier",
]
