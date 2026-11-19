# -*- coding: utf-8 -*-
"""Option 级经验回放缓冲（rl_algorithm_design.md §4.5/§6.1）。

转移格式（SMDP）：(s_before, option, Σr, s_after, τ) + 终止/截断标志 + 双端
合法掩码 + 辅助任务 GT 墙线（训练监督专用）。观测按 uint8 存储（单条
(16,16,16) uint8 = 4 KB；默认容量 10 万条 ≈ 0.8 GB + GT/掩码开销）。

掩码说明：合法性依赖账本簿记（visited_ever / machine_known / carrying /
dest_known），不能从 16 通道张量唯一复算——按 DQN 存动作掩码的标准做法，
s_before / s_after 两端掩码随转移一并存储（各 259 B，开销可忽略）。

GT 墙线说明：辅助头目标 = 转移发生时 episode 迷宫的规范形式 GT 墙线
（south/west 0/1 uint8，各 256 B）。GT 仅训练期监督（rl_algorithm_design.md
§4.1"GT 仅训练监督"），由 trainer（环境侧特权角色，与 reward 同通道）提供，
绝不进入策略输入。

时间语义：terminated（真正任务终点）不自举；truncated（超时截断）自举
（time-limit aware 标准做法，避免截断状态 Q 值被系统性压低）。
"""
from typing import Optional

import numpy as np
import torch

from .actions import N_OPTIONS


class ReplayBuffer:
    """环形 uint8 经验池。"""

    def __init__(self, capacity: int, n: int = 16):
        self.capacity = int(capacity)
        self.n = int(n)
        c = self.capacity
        self.s = np.zeros((c, 16, n, n), dtype=np.uint8)
        self.a = np.zeros(c, dtype=np.int16)
        self.r = np.zeros(c, dtype=np.float32)
        self.s2 = np.zeros((c, 16, n, n), dtype=np.uint8)
        self.tau = np.zeros(c, dtype=np.int16)
        self.terminal = np.zeros(c, dtype=bool)
        self.mask_b = np.zeros((c, N_OPTIONS), dtype=bool)     # s_before 合法掩码
        self.mask_a2 = np.zeros((c, N_OPTIONS), dtype=bool)    # s_after 合法掩码
        self.gt_south = np.zeros((c, n, n), dtype=np.uint8)    # 辅助 GT（0/1 规范墙线）
        self.gt_west = np.zeros((c, n, n), dtype=np.uint8)
        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, tau: int,
             terminal: bool, mask_b: np.ndarray, mask_a2: np.ndarray,
             gt_south: np.ndarray, gt_west: np.ndarray) -> None:
        i = self.pos
        self.s[i] = s
        self.a[i] = int(a)
        self.r[i] = float(r)
        self.s2[i] = s2
        self.tau[i] = int(tau)
        self.terminal[i] = bool(terminal)
        self.mask_b[i] = mask_b
        self.mask_a2[i] = mask_a2
        self.gt_south[i] = gt_south
        self.gt_west[i] = gt_west
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, rng: np.random.Generator) -> Optional[dict]:
        """均匀采样 batch 条 → numpy dict（设备搬移归 agent）。不足 batch 返回 None。"""
        if self.size < batch:
            return None
        idx = rng.integers(0, self.size, size=batch)
        return {
            "s": self.s[idx], "a": self.a[idx].astype(np.int64),
            "r": self.r[idx], "s2": self.s2[idx],
            "tau": self.tau[idx].astype(np.int64),
            "terminal": self.terminal[idx],
            "mask_b": self.mask_b[idx], "mask_a2": self.mask_a2[idx],
            "gt_south": self.gt_south[idx], "gt_west": self.gt_west[idx],
        }

    def clear(self) -> None:
        """阶段切换时清空（课程切换机制：阶段 0 的 transition 奖励语义失效，
        混用毒化 target——rl_algorithm_design.md §6.2）。"""
        self.pos = 0
        self.size = 0
