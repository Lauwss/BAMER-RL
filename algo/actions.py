# -*- coding: utf-8 -*-
"""高层动作空间（D6 定稿，docs/rl_algorithm_design.md §3 / contract_v1.md §5）。

259 个离散选项 = 256 Explore(点) + 2 Pick(标签) + 1 Deliver(无参)：

    索引 0~255   Explore 目标格（idx = y*n+x，与 env obs cell 同构）
    索引 256     Pick("quyuan")
    索引 257     Pick("zhaojun")
    索引 258     Deliver

合法条件（动态掩码，网络侧 Q 置 −∞ 且不进 dueling 均值）：
    Explore(x,y)：目标格已知空闲（信念图开放边界/到访格）且从当前格在信念图上
                  BFS 可达，且 ≠ 当前格（τ=0 的空转 option 会被掩码排除——
                  否则可能构造出步数永不推进的死循环）；
    Pick(标签)：  该机已识别 且 手上为空（单机负载已确认）；
    Deliver：    手上有机器 且 对应终点已定位（delivery_map 硬编码执行器——
                 题目规则"识别为屈原→楚国 / 昭君→西汉"，非场景泄漏）。

Deliver 合法性引入时序约束：终点未定位前 Deliver 非法，策略被迫先完成足够
探索；无需终止动作——选择 Pick/Deliver 而非 Explore 本身就是"停止探索"。
"""
from typing import Optional, Set, Tuple

import numpy as np

from .belief import (BeliefState, DEST_NAMES, HEADINGS, MACHINE_NAMES)  # noqa: F401 (re-export)

Cell = Tuple[int, int]

N_EXPLORE = 256
IDX_PICK = {"quyuan": 256, "zhaojun": 257}
IDX_DELIVER = 258
N_OPTIONS = 259

# 识别结果 → 派送目的地（题目规则；硬编码执行器，R1 允许：非场景内部字段）
DELIVERY_MAP = {"quyuan": "chuguo", "zhaojun": "xihan"}
DEST_NAMES = ("chuguo", "xihan")


def explore_index(cell: Cell, n: int = 16) -> int:
    """格 (x,y) → Explore 选项索引（= env obs cell 序号 y*n+x）。"""
    x, y = cell
    return y * n + x


def index_cell(idx: int, n: int = 16) -> Cell:
    """Explore 选项索引 → 格 (x,y)。"""
    return (int(idx) % n, int(idx) // n)


def option_label(idx: int, n: int = 16) -> str:
    """选项的人类可读标签（日志/诊断用）。"""
    if idx < N_EXPLORE:
        return f"Explore{index_cell(idx, n)}"
    for name, code in IDX_PICK.items():
        if idx == code:
            return f"Pick({name})"
    if idx == IDX_DELIVER:
        return "Deliver"
    raise ValueError(f"选项索引越界: {idx}")


def reachable_cells(belief: BeliefState, start: Cell) -> Set[Cell]:
    """信念图上的格级可达集（开放边界 = 已探明无墙；unknown = 不可通行的悲观约定）。

    起点恒在集合中；经开放边界扩展——开放边界意味着两侧格子均已知空闲
    （射线穿越即证），故可达集 ⊆ 已知空闲集。
    """
    seen = {start}
    stack = [start]
    n = belief.n
    while stack:
        x, y = stack.pop()
        for d, (dx, dy) in zip(HEADINGS, ((0, 1), (1, 0), (0, -1), (-1, 0))):
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and (nx, ny) not in seen \
                    and belief.open_boundary((x, y), d):
                seen.add((nx, ny))
                stack.append((nx, ny))
    return seen


def legal_mask(belief: BeliefState) -> np.ndarray:
    """当前信念下的合法选项掩码 (259,) bool。

    调用时机 = 决策边界（belief 已消费到当前步的 obs/info）。
    合法集保证非空：机器鼠来路边界必然已探明无墙（每个到访格的入射开放
    边界由扫描揭示），Explore 回溯目标恒存在。
    """
    n = belief.n
    mask = np.zeros(N_OPTIONS, dtype=bool)
    reach = reachable_cells(belief, belief.cell)
    reach.discard(belief.cell)                      # 排除 τ=0 空转目标
    for cell in reach:
        mask[explore_index(cell, n)] = True
    if belief.carrying is None:
        for name in MACHINE_NAMES:
            if name in belief.machine_known and name not in belief.machine_delivered:
                mask[IDX_PICK[name]] = True    # 已识别未送达才可 Pick（送达后回取无意义）
    else:
        dest = DELIVERY_MAP[belief.carrying]
        if dest in belief.dest_known:
            mask[IDX_DELIVER] = True
    if not mask.any():
        raise RuntimeError("合法选项集为空（信念图异常：来路边界缺失？）")
    return mask


def option_target(idx: int, belief: BeliefState) -> Cell:
    """选项目标格：Explore → 目标点；Pick → 该机已识别位置；Deliver → 终点定位格。"""
    if idx < N_EXPLORE:
        return index_cell(idx, belief.n)
    for name, code in IDX_PICK.items():
        if idx == code:
            if name not in belief.machine_known:
                raise ValueError(f"Pick({name}) 非法：该机未识别")
            return belief.machine_known[name]
    if idx == IDX_DELIVER:
        if belief.carrying is None:
            raise ValueError("Deliver 非法：手上无机器")
        dest = DELIVERY_MAP[belief.carrying]
        if dest not in belief.dest_known:
            raise ValueError(f"Deliver 非法：终点 {dest} 未定位")
        return belief.dest_known[dest]
    raise ValueError(f"选项索引越界: {idx}")
