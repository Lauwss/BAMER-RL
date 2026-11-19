# -*- coding: utf-8 -*-
"""Episode 数据 IO（M2.4，PLAN.md 阶段二）。

EpisodeRecord 统一格式：
    meta：迷宫哈希、种子、场景配置（起点/布点/终点/派送映射——环境侧事实记录，
          仅作存档元数据；回放/评估读取无妨，训练算法侧不得以其替代传感输入）、
          雷达参数、创建时间；
    steps：逐步记录（步号、位姿、动作原语、ranges/angles 量测、世界系点云、事件）。

存储：npz（数值数组）+ json（meta 与事件），共用基名；目录规范 data/episodes/。
数据格式保持中立（npz/JSON 均为 MATLAB 可直接读取的格式），支撑阶段四
Python↔MATLAB 对拍与回放展示。

用途：回放/展示（阶段四）、RL 经验池导入、诊断。
单测：写→读往返字段一致（tests/test_dataio.py）。
"""
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPISODE_DIR = os.path.join(ROOT, "data", "episodes")


def maze_hash(maze) -> str:
    """迷宫结构哈希（sha256，按墙线数组规范化序列化；同结构必同哈希）。

    参数为鸭型对象：需具有 n / wall_south / wall_west 属性（bamer_env.Maze 满足）。
    """
    payload = json.dumps(
        {"n": maze.n, "wall_south": maze.wall_south, "wall_west": maze.wall_west},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _event_to_dict(ev: Any) -> dict:
    """事件 → JSON 兼容 dict（接受 bamer_env.interaction.Event 或普通 dict）。"""
    if isinstance(ev, dict):
        return {"type": str(ev.get("type")), "step": int(ev.get("step", 0)),
                "payload": ev.get("payload", {})}
    return {"type": str(ev.type), "step": int(ev.step), "payload": dict(ev.payload)}


@dataclass
class StepRecord:
    """单步记录。ranges/angles/points 为 None 表示该步未接雷达。"""
    step: int
    x: float
    y: float
    theta_deg: float
    action: str                       # D5 原语名；reset 帧记 "RESET"
    ranges: Optional[np.ndarray] = None    # (n,) NaN=无回波
    angles: Optional[np.ndarray] = None    # (n,) 机体系束角
    points: Optional[np.ndarray] = None    # (n,2) 世界系点云（NaN 行=无回波）
    events: List[dict] = field(default_factory=list)   # JSON 兼容事件 dict 列表


@dataclass
class EpisodeRecord:
    """一局完整记录（meta + 逐步数据）。"""
    meta: Dict[str, Any]
    steps: List[StepRecord] = field(default_factory=list)

    def add_step(self, rec: StepRecord) -> None:
        self.steps.append(rec)

    # ---------- 存储 ----------
    def save(self, base_path: str) -> Dict[str, str]:
        """写入 <base>.npz（数值数组）+ <base>.json（meta/事件/动作序列）。

        传感器数组按"该 episode 全部步等长"处理：无雷达时维度记 0；
        个别步缺数据的位置以 NaN 填充并记录于 has_sensor 掩码。
        """
        base_path = os.path.abspath(base_path)
        parent = os.path.dirname(base_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        n_steps = len(self.steps)
        ranges_n = max((len(s.ranges) for s in self.steps if s.ranges is not None), default=0)
        points_shape = max((s.points.shape for s in self.steps if s.points is not None),
                           default=(0, 2))
        has_sensor = np.zeros(n_steps, dtype=bool) if n_steps else np.zeros(0, dtype=bool)
        ranges = np.full((n_steps, ranges_n), np.nan) if ranges_n else np.zeros((n_steps, 0))
        angles = np.full((n_steps, ranges_n), np.nan) if ranges_n else np.zeros((n_steps, 0))
        points = (np.full((n_steps, points_shape[0], points_shape[1]), np.nan)
                  if points_shape[0] else np.zeros((n_steps, 0, 2)))

        actions: List[str] = []
        events_json: List[List[dict]] = []
        for i, s in enumerate(self.steps):
            actions.append(s.action)
            events_json.append([_event_to_dict(e) for e in s.events])
            if s.ranges is not None:
                has_sensor[i] = True
                if ranges_n:
                    ranges[i, :len(s.ranges)] = s.ranges
                    angles[i, :len(s.angles)] = s.angles
            if s.points is not None and points_shape[0]:
                points[i, :s.points.shape[0], :] = s.points

        npz_path = base_path + ".npz"
        np.savez(
            npz_path,
            step=np.array([s.step for s in self.steps], dtype=np.int64),
            x=np.array([s.x for s in self.steps], dtype=float),
            y=np.array([s.y for s in self.steps], dtype=float),
            theta_deg=np.array([s.theta_deg for s in self.steps], dtype=float),
            action=np.array(actions, dtype="<U8"),
            has_sensor=has_sensor,
            ranges=ranges, angles=angles, points=points,
        )
        json_path = base_path + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"meta": self.meta, "events": events_json}, f,
                      ensure_ascii=False, indent=1)
        return {"npz": npz_path, "json": json_path}

    @classmethod
    def load(cls, base_path: str) -> "EpisodeRecord":
        """读回 save 写出的记录（npz + json 基名一致）。"""
        base_path = os.path.abspath(base_path)
        with open(base_path + ".json", encoding="utf-8") as f:
            doc = json.load(f)
        data = np.load(base_path + ".npz", allow_pickle=False)
        steps: List[StepRecord] = []
        n = data["step"].shape[0]
        for i in range(n):
            step = StepRecord(
                step=int(data["step"][i]),
                x=float(data["x"][i]), y=float(data["y"][i]),
                theta_deg=float(data["theta_deg"][i]),
                action=str(data["action"][i]),
                events=list(doc["events"][i]),
            )
            if bool(data["has_sensor"][i]) and data["ranges"].shape[1]:
                step.ranges = data["ranges"][i].copy()
                step.angles = data["angles"][i].copy()
            if data["points"].shape[1]:
                step.points = data["points"][i].copy()
            steps.append(step)
        return cls(meta=doc["meta"], steps=steps)
