# -*- coding: utf-8 -*-
"""评估协议：固定 seed 套件 + 全知 BFS 参照 + 指标（rl_algorithm_design.md §7）。

指标（§7.1）：
    主指标   成功率 / 搜寻步数 / 派送步数 / 总步数（含转向，铁律 R4）
    参照上限 全知 BFS 参照：已知全图下最优任务序的 (格,朝向) 最短路径总步数
             ——策略步数与参照的差距是核心报告量
    诊断量   探索区域率、墙知识率、option 数、搜寻阶段占比

参照实现 = (格,朝向,任务进度) 状态空间 BFS：进度 = 已按序到访的任务点数
（单机负载约束下合法任务序仅 2 条：Pq→Dq→Pz→Dz / Pz→Dz→Pq→Dq），到达进度
对应任务点自动推进（拾取/派送随到达生效）。语义注意：env 侧"空手经过机器格
即拾取"未建模——若最优路径恰好空手穿越他机格，实际执行会强制改变任务序；
此极端拓扑差异在报告时如实注明（参照定义见 rl_algorithm_design.md §7.1）。

R1：本模块为评估侧（特权角色），GT 迷宫/布点仅用于参照计算与指标统计，
绝不进入策略输入。
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from bamer_env.scene import Scene

from .belief import HEADINGS, BeliefState

Cell = Tuple[int, int]

_TURN = {"TL": -1, "TR": +1, "TB": +2}


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
def _task_orders() -> List[List[str]]:
    """单机负载下的合法任务点序（拾取 P 在其派送 D 前，配对固定）：仅 2 条。"""
    return [["quyuan", "chuguo", "zhaojun", "xihan"],
            ["zhaojun", "xihan", "quyuan", "chuguo"]]


def omniscient_reference(scene: Scene) -> Optional[int]:
    """全知参照：已知全图 + 布点下，最优任务序的 (格,朝向) 最短路径总步数。

    状态 = (格, 朝向, 进度)；到达进度对应任务点 → 进度自动 +1（拾取/派送随
    到达自动生效，无需额外动作）；进度走满 = 任务完成。两条合法序取最小；
    不可达返回 None（场景校验已保证任务点互达，理论不会发生）。
    """
    maze = scene.maze

    def open_gt(cell: Cell, direction: str) -> bool:
        return not maze.has_wall(cell[0], cell[1], direction)

    waypoints_by_order = []
    for order in _task_orders():
        pts = []
        for name in order:
            pts.append(scene.state_machines[name] if name in scene.state_machines
                       else scene.destinations[name])
        waypoints_by_order.append(pts)

    best = None
    for waypoints in waypoints_by_order:
        cost = _staged_bfs_cost(maze.n, scene.start, scene.start_heading, waypoints, open_gt)
        if cost is not None and (best is None or cost < best):
            best = cost
    return best


def _staged_bfs_cost(n: int, start: Cell, start_heading: str,
                     waypoints: List[Cell], open_fn) -> Optional[int]:
    """(格,朝向,进度) BFS：进度 k 表示 waypoints[:k] 已按序到访；到达 waypoints[k]
    自动推进。每条边 = 1 个 D5 原语 = 1 步（R4 转向计步）。"""
    h0 = HEADINGS.index(start_heading)
    start = (start[0], start[1], h0, 0)
    seen = {start}
    dq = deque([(start, 0)])
    while dq:
        (x, y, h, k), cost = dq.popleft()
        if k < len(waypoints) and (x, y) == waypoints[k]:
            k += 1                              # 到达任务点：拾取/派送自动生效（0 额外步）
            if k == len(waypoints):
                return cost
        for d, (dx, dy) in zip(HEADINGS, ((0, 1), (1, 0), (0, -1), (-1, 0))):
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and open_fn((x, y), d) \
                    and (nx, ny, h, k) not in seen:
                seen.add((nx, ny, h, k))
                dq.append(((nx, ny, h, k), cost + 1))
        for prim, dh in _TURN.items():
            if (x, y, (h + dh) % 4, k) not in seen:
                seen.add((x, y, (h + dh) % 4, k))
                dq.append(((x, y, (h + dh) % 4, k), cost + 1))
    return None


# ---------- 套件评估 ----------
def run_eval_episode(agent, env, belief: BeliefState, executor, seed: int,
                     max_steps: int) -> EpisodeMetrics:
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
        mask = _legal_mask_safe(belief)
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


def _legal_mask_safe(belief: BeliefState):
    from .actions import legal_mask
    return legal_mask(belief)


def evaluate(agent, env_factory: Callable[[], object], seeds: Sequence[int],
             belief: BeliefState, executor, suite: str = "held-out") -> EvalSummary:
    """固定 seed 套件评估（§7.2：held-out 随机套件 / 官方图套件分开报告）。"""
    metrics: List[EpisodeMetrics] = []
    for seed in seeds:
        env = env_factory()
        metrics.append(run_eval_episode(agent, env, belief, executor, int(seed),
                                        env.max_steps))
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
