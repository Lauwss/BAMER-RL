# -*- coding: utf-8 -*-
"""评估协议：固定 seed 套件 + 全知 BFS 参照 + 指标（rl_algorithm_design.md §7）。

指标（§7.1）：
    主指标   成功率 / 搜寻步数 / 派送步数 / 总步数（含转向，铁律 R4）
    参照上限 全知 BFS 参照：已知全图下最优任务序的 (格,朝向) 最短路径总步数
             ——策略步数与参照的差距是核心报告量
    诊断量   探索区域率、墙知识率、option 数、搜寻阶段占比

参照实现 = (格,朝向,携带,送达) 状态空间 BFS（8192 态）：与 env 语义逐条一致的
强制事件自动机——空手到达机器格强制拾取、已携带到达映射终点格强制送达、
已送达不再拾取、单机负载。规则 A（2026-11-20 用户确认）的拓扑性代价（机器
挡在唯一路径上时的强制拾取序）由此如实计入参照，策略 vs 参照差距严格公平。

R1：本模块为评估侧（特权角色），GT 迷宫/布点仅用于参照计算与指标统计，
绝不进入策略输入。
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from bamer_env.scene import Scene

from .actions import DELIVERY_MAP
from .belief import HEADINGS, BeliefState

Cell = Tuple[int, int]


@dataclass
class EpisodeMetrics:
    """单条评估 episode 的指标。"""
    seed: int
    success: bool                       # terminated（全任务完成）
    total_steps: int                    # 总步数（含转向；truncated 时 = max_steps）
    search_steps: Optional[int]         # 搜寻步数：4 POI 全部定位时的累计步数（未达成 None）
    delivery_steps: Optional[int]       # 派送步数 = total − search（未完成 None）
    options: int                        # option 数
    explored_ratio: float               # 探索区域率 = 到访格 / 总格
    wall_knowledge_ratio: float         # 墙知识率 = 已观测槽位 / 全部槽位
    terminated: bool
    truncated: bool

    def gap_to_reference(self, reference: Optional[int]) -> Optional[int]:
        if reference is None or not self.success:
            return None
        return self.total_steps - reference


@dataclass
class EvalSummary:
    """套件汇总（两套分开报告：held-out 随机 / 官方图）。"""
    suite: str
    n_episodes: int
    success_rate: float
    mean_total_steps: Optional[float]
    mean_search_steps: Optional[float]
    mean_delivery_steps: Optional[float]
    mean_options: Optional[float]
    mean_explored_ratio: float
    mean_wall_knowledge_ratio: float
    mean_gap_to_reference: Optional[float]
    per_episode: List[EpisodeMetrics] = field(default_factory=list)


# ---------- 全知 BFS 参照 ----------
def omniscient_reference(scene: Scene) -> Optional[int]:
    """全知参照：已知全图 + 布点下，任务规则下的最短 (格,朝向) 路径总步数。

    与 env 语义**逐条一致**的强制事件自动机（2026-11-20 规则 A 确认后升级）：
    状态 = (格, 朝向, carrying, 送达集)；到达机器格且空手且未送达 → 强制拾取
    （0 额外步）；已携带且映射终点格到达 → 强制送达（0 额外步）；已送达机器
    不再拾取；单机负载。两条移动边 = 1 个 D5 原语 = 1 步（R4 转向计步）。
    有效 (carrying, delivered) 组合 8 种 × 1024 = 8192 状态，BFS 精确最优。
    不可达返回 None（场景校验已保证任务点互达，理论不会发生）。
    """
    maze = scene.maze
    machines = dict(scene.state_machines)
    dests = dict(scene.destinations)
    n = maze.n

    def open_gt(cell: Cell, direction: str) -> bool:
        return not maze.has_wall(cell[0], cell[1], direction)

    def settle(cell: Cell, c: int, dq: int, dz: int):
        """到达事件的强制结算（与 interaction.enter_events 同序：先拾取后派送）。"""
        for mname, mcell in machines.items():
            if cell == mcell:
                if c == 0 and (mname == "quyuan" and not dq or mname == "zhaojun" and not dz):
                    c = 1 if mname == "quyuan" else 2          # 强制拾取（单机负载：空手才触发）
        for mname, dname in DELIVERY_MAP.items():
            if c == (1 if mname == "quyuan" else 2) and dests.get(dname) == cell:
                if mname == "quyuan":
                    dq, c = 1, 0                               # 强制送达，双手解放
                else:
                    dz, c = 1, 0
        return c, dq, dz

    from collections import deque
    h0 = HEADINGS.index(scene.start_heading)
    c0, dq0, dz0 = settle(scene.start, 0, 0, 0)               # 出生格即机器/终点格的防御
    start = (scene.start[0], scene.start[1], h0, c0, dq0, dz0)
    if dq0 and dz0:
        return 0
    seen = {start}
    dq_queue = deque([(start, 0)])
    while dq_queue:
        (x, y, h, c, dqv, dzv), cost = dq_queue.popleft()
        if dqv and dzv:
            return cost
        # F 边：仅当前朝向 d = HEADINGS[h]（转向由独立的旋转边承担，R4 计步）
        d = HEADINGS[h]
        dx, dy = ((0, 1), (1, 0), (0, -1), (-1, 0))[h]
        nx, ny = x + dx, y + dy
        if 0 <= nx < n and 0 <= ny < n and open_gt((x, y), d):
            c2, dq2, dz2 = settle((nx, ny), c, dqv, dzv)
            ns = (nx, ny, h, c2, dq2, dz2)          # 前进不改朝向
            if ns not in seen:
                seen.add(ns)
                dq_queue.append((ns, cost + 1))
        for dh in (-1, +1, +2):                     # TL/TR/TB 原地旋转，各计 1 步
            ns = (x, y, (h + dh) % 4, c, dqv, dzv)
            if ns not in seen:
                seen.add(ns)
                dq_queue.append((ns, cost + 1))
    return None


# ---------- 套件评估 ----------
def run_eval_episode(agent, env, belief: BeliefState, executor, seed: int,
                     max_steps: int, frontier_only: bool = False) -> EpisodeMetrics:
    """贪心（ε=0）跑一条完整 episode，只统计不学习。"""
    obs, info = env.reset(seed=seed)
    belief.reset(env.scene.maze.n)
    belief.observe(obs, info)
    steps = 0
    options = 0
    search_steps: Optional[int] = None
    terminated = truncated = False
    abort_guard = 0
    while True:
        s = belief.tensor()
        mask = _legal_mask_safe(belief, frontier_only)
        action = agent.act(s, mask, greedy=True)
        result = executor.run(env, action)
        options += 1
        steps += result.steps
        belief.commit_option(result.path, result.start_cell)
        if search_steps is None and belief.poi_complete():
            search_steps = steps
        terminated, truncated = result.terminated, result.truncated
        if result.steps == 0:
            abort_guard += 1     # 不可达中止：失败信号交还高层；连续中止 8 次判死锁
            if abort_guard >= 8:
                break
        else:
            abort_guard = 0
        if terminated or truncated:
            break
    n = belief.n
    return EpisodeMetrics(
        seed=seed, success=bool(terminated), total_steps=steps,
        search_steps=search_steps,
        delivery_steps=(steps - search_steps) if (search_steps is not None and terminated) else None,
        options=options,
        explored_ratio=len(belief.visited) / (n * n),
        wall_knowledge_ratio=1.0 - belief.ledger.unknown_count() / (2 * n * n),
        terminated=bool(terminated), truncated=bool(truncated),
    )


def _legal_mask_safe(belief: BeliefState, frontier_only: bool = False):
    from .actions import legal_mask
    return legal_mask(belief, frontier_only=frontier_only)


def evaluate(agent, env_factory: Callable[[], object], seeds: Sequence[int],
             belief: BeliefState, executor, suite: str = "held-out",
             frontier_only: bool = False) -> EvalSummary:
    """固定 seed 套件评估（§7.2：held-out 随机套件 / 官方图套件分开报告）。"""
    metrics: List[EpisodeMetrics] = []
    for seed in seeds:
        env = env_factory()
        metrics.append(run_eval_episode(agent, env, belief, executor, int(seed),
                                        env.max_steps, frontier_only=frontier_only))
        try:
            env.close()
        except Exception:
            pass
    ok = [m for m in metrics if m.success]
    mean = lambda vals: (sum(vals) / len(vals)) if vals else None   # noqa: E731
    return EvalSummary(
        suite=suite, n_episodes=len(metrics),
        success_rate=len(ok) / len(metrics) if metrics else 0.0,
        mean_total_steps=mean([m.total_steps for m in ok]),
        mean_search_steps=mean([m.search_steps for m in ok if m.search_steps is not None]),
        mean_delivery_steps=mean([m.delivery_steps for m in ok if m.delivery_steps is not None]),
        mean_options=mean([m.options for m in ok]),
        mean_explored_ratio=mean([m.explored_ratio for m in metrics]) or 0.0,
        mean_wall_knowledge_ratio=mean([m.wall_knowledge_ratio for m in metrics]) or 0.0,
        mean_gap_to_reference=None,     # 与参照的差距在调用方逐场景核算（参照依赖布点）
        per_episode=metrics,
    )


def summarize_with_reference(summary: EvalSummary, scenes: Dict[int, Scene]) -> EvalSummary:
    """逐 episode 核算与全知参照的差距（同一 seed 需配对同一场景实例）。"""
    gaps = []
    for m in summary.per_episode:
        scene = scenes.get(m.seed)
        ref = omniscient_reference(scene) if scene is not None else None
        gap = m.gap_to_reference(ref)
        if gap is not None:
            gaps.append(gap)
    if gaps:
        summary.mean_gap_to_reference = sum(gaps) / len(gaps)
    return summary
