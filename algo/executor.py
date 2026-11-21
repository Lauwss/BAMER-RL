# -*- coding: utf-8 -*-
"""确定性执行器：(格,朝向) 1024 态 BFS + 每步重规划 + 不可达中止（rl_algorithm_design.md §3.2）。

⚠️ 转向计步（铁律 R4）后最短路不是格子 BFS——纯格子图搜出的路多转弯，直接
吃掉步数指标。状态空间 (cell, heading) = 16×16×4 = 1024 态，每条边恰对应一个
D5 原语（F/TL/TR/TB），BFS 最短路 = 最少原语数 = 最少步数。

核心机制：
    - 每步重规划：option 执行途中信念在更新（新墙揭示），已规划路径可能被
      新发现的墙堵死——每步在最新信念图上重规划（1024 态 BFS，微秒级），
      只执行规划的第一个原语；
    - 不可达中止：目标在信念图上彻底不可达 → option 以失败终止，控制权交还
      高层（失败信号本身是探索信息）；
    - 碰撞一致性：执行器只在"已探明无墙"边界上规划 F，无噪声下不会碰撞；
      若发生（信念/环境不一致，如 Collision 事件揭示新墙），由每步重规划自然
      纠正（belief 已从事件更新账本）。

执行器展开的低层动作全部落在 D5 原语 {F, TL, TR, TB} 上（§3.2 约束）；option
终止条件 = 到达目标格 / 不可达中止 / 环境终止（terminated/truncated）。
"""
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from bamer_env.maze import DIRS

from .actions import option_target
from .belief import HEADINGS, BeliefState

# 朝向转移：TL/TR/TB 作用在 HEADINGS 下标上（模 4），每个旋转原语计 1 步（R4）
_TURN = {"TL": -1, "TR": +1, "TB": +2}

Cell = Tuple[int, int]


def bfs_path(n: int, start_cell: Cell, start_heading: str, target: Cell,
             open_fn) -> Optional[List[str]]:
    """(格,朝向) BFS 最短路（最少原语数）。

    open_fn(cell, direction) -> bool：该边界"已探明无墙"判据。执行器传入
    belief.open_boundary（信念图），全知参照（eval.py）传入 GT 墙查询的取反。
    返回原语列表；起点即目标返回空表；不可达返回 None。
    """
    h0 = HEADINGS.index(start_heading)
    if start_cell == target:
        return []
    start = (start_cell[0], start_cell[1], h0)
    prev = {start: None}                    # state → (prev_state, primitive)
    dq = deque([start])
    while dq:
        x, y, h = dq.popleft()
        d = HEADINGS[h]
        dx, dy = DIRS[d]
        nx, ny = x + dx, y + dy
        if 0 <= nx < n and 0 <= ny < n and open_fn((x, y), d):
            ns = (nx, ny, h)
            if ns not in prev:
                prev[ns] = ((x, y, h), "F")
                if (nx, ny) == target:
                    return _reconstruct(prev, ns)
                dq.append(ns)
        for prim, dh in _TURN.items():
            ns = (x, y, (h + dh) % 4)
            if ns not in prev:
                prev[ns] = ((x, y, h), prim)
                dq.append(ns)
    return None


def _reconstruct(prev: dict, state: Tuple[int, int, int]) -> List[str]:
    prims = []
    cur = state
    while prev[cur] is not None:
        cur, prim = prev[cur][0], prev[cur][1]
        prims.append(prim)
    prims.reverse()
    return prims


@dataclass
class OptionResult:
    """一次 option 执行的完整记录（SMDP 转移的原始材料）。"""
    option: int                     # 选项索引
    steps: int                      # τ：实际消耗的低层步数
    events: List[object] = field(default_factory=list)   # 执行期间全部交互事件
    path: List[Cell] = field(default_factory=list)       # 实际走过的格序列（含起点）
    success: bool = False           # 到达目标格
    aborted: bool = False           # 不可达中止（失败信号交还高层）
    terminated: bool = False        # 环境终止（全任务完成 / 课程终止）
    truncated: bool = False         # 环境截断（超时）

    @property
    def start_cell(self) -> Optional[Cell]:
        return self.path[0] if self.path else None


class Executor:
    """option 执行器：持有一个 BeliefState，驱动 BamerEnv 直到 option 终止。"""

    def __init__(self, belief: BeliefState, max_option_steps: int = 512,
                 lazy_replan: bool = True):
        self.belief = belief
        self.max_option_steps = int(max_option_steps)
        # 按需重规划（2026-11-21 性能等价变换）：账本自上次规划后无变化时，确定性
        # BFS 从当前 (cell,heading) 必然给出缓存路径的剩余后缀 → 直接沿缓存执行，
        # 跳过逐步 BFS。等价性：BFS/运动学均确定性，路径后缀 = 现状重规划结果。
        # 账本逐帧比对（脏检测）保证任何新墙揭示即刻触发重规划，语义不变。
        self.lazy_replan = bool(lazy_replan)

    # ---------- 规划 ----------
    def plan(self, target: Cell) -> Optional[List[str]]:
        """在最新信念图上规划到 target 的原语序列（1024 态 BFS）。

        实测注记（2026-11-20）：曾试做 numpy 前沿扩展版，微基准 0.85ms vs 标量
        0.56ms（0.7×，更慢）——标量版提前终止只访问路径半径内 ~150 态，而逐层
        numpy 的固定开销（fancy indexing/unique/分配）在 1024 态规模摊不平，
        短路占优时标量更快。负结果详见 CHANGELOG，勿再重复此优化方向。
        """
        b = self.belief
        return bfs_path(b.n, b.cell, b.heading, target, b.open_boundary)

    # ---------- 执行 ----------
    def run(self, env, option_idx: int) -> OptionResult:
        """执行一个高层选项：展开为原语序列，每步重规划，直至 option 终止。

        前置条件：option_idx 已过合法性掩码（trainer 负责；此处仅防御校验）。
        信念逐步更新（belief.observe），option 结束后由 trainer 登记轨迹账本。
        """
        b = self.belief
        target = option_target(option_idx, b)
        result = OptionResult(option=option_idx, steps=0)
        result.path = [b.cell]

        if b.cell == target:
            # 目标即当前格：合法掩码已排除 Explore 自身格；Pick/Deliver 到位即完成
            # 的状态在掩码语义下不可能持久（拾取/派送随到达即时生效），防御性立即成功
            result.success = True
            return result

        plan: Optional[List[str]] = None    # 缓存路径（lazy 模式沿后缀执行）
        plan_i = 0
        plan_ledger: Optional[Tuple[np.ndarray, np.ndarray]] = None
        while result.steps < self.max_option_steps:
            if self.lazy_replan:
                dirty = (plan is None or plan_i >= len(plan)
                         or not (np.array_equal(plan_ledger[0], b.ledger.sensed_south)
                                 and np.array_equal(plan_ledger[1], b.ledger.sensed_west)))
                if dirty:
                    plan = self.plan(target)
                    plan_i = 0
                    if plan is None:
                        result.aborted = True   # 不可达中止（失败信号交还高层）
                        break
                    plan_ledger = (b.ledger.sensed_south.copy(), b.ledger.sensed_west.copy())
                prim = plan[plan_i]             # 无新信息：缓存后缀 = 现状重规划结果
                plan_i += 1
            else:
                plan = self.plan(target)
                if plan is None:
                    result.aborted = True       # 不可达中止（失败信号交还高层）
                    break
                prim = plan[0]                  # 每步重规划：只执行第一个原语
            obs, _, terminated, truncated, info = env.step(prim)
            result.steps += 1
            result.events.extend(info["events"])
            result.terminated = bool(terminated)
            result.truncated = bool(truncated)
            b.observe(obs, info)
            if b.cell != result.path[-1]:
                result.path.append(b.cell)
            if b.cell == target:
                result.success = True       # 到达即成功（terminated 标志已先行记录）
                break
            if terminated or truncated:
                break
        else:
            result.aborted = True           # 步数保险丝（理论上不可达：BFS 有路必进展）
        return result
