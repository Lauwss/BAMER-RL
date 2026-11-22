# -*- coding: utf-8 -*-
"""执行器单测（§8 冒烟 2）：已知图最短路解析解对拍（含转向计步）；重规划/中止用例；Pick/Deliver option 全流程。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from algo.actions import IDX_DELIVER, IDX_PICK, N_EXPLORE, explore_index, legal_mask
from algo.belief import W_FREE, W_WALL, BeliefState
from algo.executor import Executor, bfs_path, bfs_path_weighted
from bamer_env import BamerEnv, Maze
from bamer_mouse.lidar import LidarSpec


def empty_maze(n: int) -> Maze:
    return Maze([[False] * n for _ in range(n)], [[False] * n for _ in range(n)])


def wall_grid(n, walls_between):
    """按 ((x,y), 方向) 列表开/关墙的迷宫：walls_between 中的边界无墙，其余有墙。"""
    south = [[True] * n for _ in range(n)]
    west = [[True] * n for _ in range(n)]
    # 题目场景：外边框封闭，全部内部边界默认有墙，按需开通
    for (x, y), d in walls_between:
        if d == "S":
            south[x][y] = False
        elif d == "N":
            south[x][y + 1] = False
        elif d == "W":
            west[x][y] = False
        elif d == "E":
            west[x + 1][y] = False
    return Maze(south, west)


def test_bfs_turn_counting():
    """解析解对拍（转向计步 R4）：TB=1 步 180°、原路折返优于绕转。"""
    open_corridor = [((x, 0), "E") for x in range(0, 3)]        # 一条 1×4 走廊 (0..3,0)
    maze = wall_grid(4, open_corridor)
    open_fn = lambda cell, d: not maze.has_wall(cell[0], cell[1], d)
    # (0,0) 朝北 → (3,0)：必须转身；TB F F F = TR F F F = 4 步（TB/TR 均一步转向，并列最优）
    prims = bfs_path(4, (0, 0), "N", (3, 0), open_fn)
    assert prims in (["TB", "F", "F", "F"], ["TR", "F", "F", "F"]), prims
    # (0,0) 朝北 → (1,0)：TR F = 2（或 TB F = 2），BFS 取其一（并列最优）
    prims = bfs_path(4, (0, 0), "N", (1, 0), open_fn)
    assert len(prims) == 2 and prims[-1] == "F"
    # 起点 = 目标：空序列
    assert bfs_path(4, (2, 0), "N", (2, 0), open_fn) == []
    # 不可达：None
    assert bfs_path(4, (0, 0), "N", (3, 3), open_fn) is None
    print("[BFS 对拍] OK：TB 计 1 步 / 折返最短 / 空序列 / 不可达 None")


def test_bfs_detour():
    """有墙迷宫的最短路对拍：墙迫使绕行，步数与手算一致。"""
    # 3×3：开通 (0,0)E、(1,0)E、(0,1)N 形成路径 (0,0)→(2,0)→(2,1)；(0,0)N 被墙挡
    walls = [((0, 0), "E"), ((1, 0), "E"), ((2, 0), "N")]
    maze = wall_grid(3, walls)
    open_fn = lambda cell, d: not maze.has_wall(cell[0], cell[1], d)
    # (0,0) 朝北 → (2,1)：唯一通路 = 东走廊 + (2,0) 北开；手算 TR(E) F F TL(N) F = 5 步
    prims = bfs_path(3, (0, 0), "N", (2, 1), open_fn)
    assert prims == ["TR", "F", "F", "TL", "F"], prims
    print("[BFS 绕行] OK：墙迫使折返，最少原语数与手算一致")


def _scan_env(maze, machines, dests, n):
    return BamerEnv(maze=maze, state_machines=machines, destinations=dests,
                    max_steps=300, lidar=LidarSpec())


def test_executor_explore_options():
    """Executor 驱动 Explore option：到达即成功、步数=信念图 BFS 最短、无碰撞、信念同步。"""
    maze = empty_maze(5)
    env = _scan_env(maze, {"quyuan": (0, 2), "zhaojun": (3, 1)},
                    {"chuguo": (0, 4), "xihan": (1, 3)}, 5)
    obs, info = env.reset(seed=0)
    belief = BeliefState(n=5)
    belief.observe(obs, info)
    ex = Executor(belief)

    # option 1：Explore (0,1)（reset 扫描已揭示：N 边界开放）
    idx = explore_index((0, 1), 5)
    assert legal_mask(belief)[idx]
    r1 = ex.run(env, idx)
    assert r1.success and r1.steps == 1 and r1.path == [(0, 0), (0, 1)]
    assert not r1.aborted and r1.terminated is False
    assert all(e.type in ("Reported",) for e in r1.events)   # 途经无 POI 格 → 仅上报
    belief.commit_option(r1.path, r1.start_cell)

    # option 2：继续北上 (0,2) = 屈原格 → 首达识别+拾取事件入 option 记录
    idx = explore_index((0, 2), 5)
    assert legal_mask(belief)[idx]
    r2 = ex.run(env, idx)
    assert r2.success and r2.steps == 1
    types = [e.type for e in r2.events]
    assert "ImageFeed" in types and "Pickup" in types
    assert belief.carrying == "quyuan" and belief.machine_known["quyuan"] == (0, 2)
    belief.commit_option(r2.path, r2.start_cell)

    # option 3：跨图 Explore (2,0)（折返东行），验证多步 option 的每步重规划
    idx = explore_index((2, 0), 5)
    mask = legal_mask(belief)
    assert mask[idx]
    expected = len(ex.plan((2, 0)))          # 边界处的 BFS 最短原语数
    r3 = ex.run(env, idx)
    assert r3.success and r3.path[0] == (0, 2) and r3.path[-1] == (2, 0)
    assert belief.cell == (2, 0)
    assert r3.steps == expected              # 无噪声下执行步数 = 边界规划步数
    assert not any(e.type == "Collision" for e in r3.events)
    print("[Explore option] OK：到达/步数=BFS 最短/拾取事件/多步重规划/无碰撞")


def test_executor_abort_unreachable():
    """不可达中止：目标不在信念图可达集 → option 以失败交还高层。"""
    maze = empty_maze(7)
    env = _scan_env(maze, {"quyuan": (6, 6), "zhaojun": (5, 5)},
                    {"chuguo": (0, 6), "xihan": (6, 0)}, 7)
    obs, info = env.reset(seed=0)
    belief = BeliefState(n=7)
    belief.observe(obs, info)
    ex = Executor(belief)
    # 远端格从未被观测 → 不在可达集 → run 直接中止（掩码本应排除，此处白盒验证中止路径）
    r = ex.run(env, explore_index((6, 6), 7))
    assert r.aborted and not r.success and r.steps == 0
    print("[不可达中止] OK：信念图不可达 → 失败信号交还高层")


def test_executor_abort_mid_option():
    """option 执行途中新墙揭示切断唯一路径 → 中途中止（每步重规划语义）。"""
    # 5×5：只开通一条走廊 (0,0)→(4,0)，目标 (4,4) 沿唯一竖直通道 (4,0)→(4,4)
    walls = [((x, 0), "E") for x in range(0, 4)] + [((4, y), "N") for y in range(0, 3)]
    maze = wall_grid(5, walls)
    env = _scan_env(maze, {"quyuan": (1, 0), "zhaojun": (4, 2)},
                    {"chuguo": (3, 0), "xihan": (4, 3)}, 5)
    obs, info = env.reset(seed=0)
    belief = BeliefState(n=5)
    belief.observe(obs, info)
    ex = Executor(belief)
    # 分段探索（悲观规划只走已知区）：先到 (2,0)（reset 扫描揭示范围），再到走廊尽头 (4,0)
    idx = explore_index((2, 0), 5)
    assert legal_mask(belief)[idx]
    r = ex.run(env, idx)
    assert r.success
    belief.commit_option(r.path, r.start_cell)
    idx = explore_index((4, 0), 5)
    assert legal_mask(belief)[idx]
    r = ex.run(env, idx)
    assert r.success
    belief.commit_option(r.path, r.start_cell)
    # 白盒注入：在 (4,0) 正北边界揭示一堵墙（切断通往 (4,4) 的唯一通道），
    # 模拟"新扫描揭示墙"——belief 的悲观更新（测试扮演环境侧揭示）
    belief.ledger.mark_wall(4, 1, "S")
    r2 = ex.run(env, explore_index((4, 4), 5))
    assert r2.aborted and not r2.success       # 唯一路径被切断 → 中止
    print("[中途中止] OK：新墙切断唯一路径 → 每步重规划触发不可达中止")


def test_executor_pick_and_deliver_flow():
    """Pick/Deliver option 全流程（2026-11-20 携带语义）：
    识别（手持时不拾取）→ Deliver → Pick 回取 → Deliver → TaskComplete。"""
    maze = empty_maze(5)
    env = _scan_env(maze, {"quyuan": (0, 2), "zhaojun": (2, 2)},
                    {"chuguo": (4, 2), "xihan": (4, 3)}, 5)
    obs, info = env.reset(seed=0)
    belief = BeliefState(n=5)
    belief.observe(obs, info)
    ex = Executor(belief)

    # 1) 分段 Explore 到昭君格 (2,2)（悲观规划只走已知区）：识别+拾取昭君
    for target in [(2, 0), (2, 2)]:
        idx = explore_index(target, 5)
        assert legal_mask(belief)[idx]
        r = ex.run(env, idx)
        assert r.success and belief.cell == target
        belief.commit_option(r.path, r.start_cell)
    assert belief.carrying == "zhaojun"

    # 2) 手持昭君途经屈原格 (0,2) → 仅识别不拾取（单机负载）
    idx = explore_index((0, 2), 5)
    assert legal_mask(belief)[idx]
    r = ex.run(env, idx)
    assert r.success and belief.carrying == "zhaojun"
    assert "quyuan" in belief.machine_known    # 识别完成
    belief.commit_option(r.path, r.start_cell)

    # 3) Deliver（合法：终点 xihan 已定位？——尚未！先验证掩码再探索）
    mask = legal_mask(belief)
    assert not mask[IDX_DELIVER]               # 西汉未定位 → Deliver 非法（时序约束）
    # Explore 到西汉 (4,3)：BFS 最短路走 (0,3) 行（不经楚国）→ 到达即 Arrived
    idx = explore_index((4, 3), 5)
    assert legal_mask(belief)[idx]
    r = ex.run(env, idx)
    assert r.success
    assert belief.carrying is None and "zhaojun" in belief.machine_delivered
    assert "xihan" in belief.dest_known        # 终点已自主发现并定位
    belief.commit_option(r.path, r.start_cell)

    # 4) Pick(quyuan)（合法：已识别 + 空手）：到达屈原格 → Pickup 生效
    mask = legal_mask(belief)
    assert mask[IDX_PICK["quyuan"]] and not mask[IDX_PICK["zhaojun"]]   # 已送达不再 Pick
    r = ex.run(env, IDX_PICK["quyuan"])
    assert r.success and belief.carrying == "quyuan"
    assert any(e.type == "Pickup" and e.payload["machine"] == "quyuan" for e in r.events)
    belief.commit_option(r.path, r.start_cell)

    # 5) Deliver → 楚国 (4,2)：到达即发现楚国 + 派送 → 双派送达成 TaskComplete
    mask = legal_mask(belief)
    assert mask[IDX_DELIVER]
    r = ex.run(env, IDX_DELIVER)
    assert r.success and r.terminated
    assert belief.carrying is None
    assert belief.machine_delivered == {"quyuan", "zhaojun"}
    assert "chuguo" in belief.dest_known       # 目标格即终点格：发现 + 派送同入
    assert any(e.type == "TaskComplete" for e in r.events)
    # 全程无碰撞（悲观规划下 F 只走已探明开放边界）
    print("[Pick/Deliver 全流程] OK：识别→拾取→派送→回取→派送→TaskComplete，掩码时序约束正确")


def test_executor_step_accounting():
    """步数记账（R3/R4）：option 的 τ = 低层原语数 = env 步数增量，每步恰 +1（含转向）。"""
    maze = empty_maze(5)
    env = _scan_env(maze, {"quyuan": (4, 4), "zhaojun": (3, 1)},
                    {"chuguo": (0, 4), "xihan": (1, 3)}, 5)
    obs, info = env.reset(seed=0)
    belief = BeliefState(n=5)
    belief.observe(obs, info)
    ex = Executor(belief)
    steps_log = []
    orig_step = env.step

    def spy(action):
        out = orig_step(action)
        steps_log.append(out[4]["steps"])
        return out

    env.step = spy
    r = ex.run(env, explore_index((1, 1), 5))   # 斜向目标，必含转向原语
    assert r.success
    assert len(steps_log) == r.steps            # 每个 D5 原语 = 1 个 env step
    assert steps_log[-1] - steps_log[0] == r.steps - 1   # 每步恰计 1
    assert any("Collision" not in e.type for e in r.events)
    print("[步数记账] OK：option 步数与 env 计步一致（含转向原语）")


def _belief_two_routes(n=7):
    """手工信念（槽位共享已仔细核算）：起点 (0,0)E → 目标 (6,0)。

    直走廊 y=0：6 F 全扫过（gain 全 0）→ λ=0.8 代价 6.0；
    绕行走廊 y=1：TL F TR F×6 TR F = 11 原语（3 转向 + 8 移动），
      (0..6,1) 各 gain 1（N 槽未知）→ λ=0.8 代价 3 + 8×0.2 = 4.6 < 6.0
      → 绕行收割获胜。
    λ=0：直走廊 6 原语最短。两走廊间 y=1 行以南边界全标 WALL（槽位共享
    不给直走廊送 gain）；竖井 (0,1)/(6,1) 通道共享槽如实计入双方。
    """
    b = BeliefState(n=n)
    b.cell = (0, 0)
    b.heading = "E"
    led = b.ledger
    for x in range(1, 7):
        led._mark(led.sensed_west, x, 0, W_FREE)      # 直走廊通道
        led._mark(led.sensed_west, x, 1, W_FREE)      # 绕行走廊通道
    for x in range(0, 7):
        led._mark(led.sensed_south, x, 0, W_WALL)     # 底行南界（已扫）
        if 1 <= x <= 5:
            led._mark(led.sensed_south, x, 1, W_WALL)  # 走廊间隔墙（已扫）
    led._mark(led.sensed_south, 0, 1, W_FREE)          # 入口竖井 (0,0)-(0,1)
    led._mark(led.sensed_south, 6, 1, W_FREE)          # 出口竖井 (6,1)-(6,0)
    led._mark(led.sensed_west, 0, 0, W_WALL)           # 西边框（已扫）
    led._mark(led.sensed_west, 0, 1, W_WALL)
    return b


def _prims_to_cells(start, start_heading, prims):
    """原语序列 → 到访格序列（测试断言用；不含碰撞语义——全部边界已探明）。"""
    from algo.belief import HEADINGS
    from bamer_env.maze import DIRS
    cells = [start]
    h = HEADINGS.index(start_heading)
    x, y = start
    for p in prims:
        if p == "F":
            dx, dy = DIRS[HEADINGS[h]]
            x, y = x + dx, y + dy
        else:
            h = (h + {"TL": -1, "TR": 1, "TB": 2}[p]) % 4
        cells.append((x, y))
    return cells


def test_weighted_path_handcheck():
    """信息加权寻路（2026-11-22 P1）：λ=0 取最短；λ=0.8 绕行收割未扫槽获胜。"""
    b = _belief_two_routes()
    g = Executor(b, path_gain_lambda=0.8)._gain_array()
    # gain 手算对拍：绕行走廊 (0..6,1) 各 gain 1（N 槽未知）；直走廊 (0..6,0) 全 0
    # （(6,0) 的 N 槽 = 出口竖井 FREE 已知，不计 gain）
    for x in range(7):
        assert g[x, 1] == 1, f"({x},1) gain={g[x, 1]}"
        assert g[x, 0] == 0, f"({x},0) gain={g[x, 0]}"

    ex0 = Executor(b, path_gain_lambda=0.0)
    p0 = ex0.plan((6, 0))
    assert p0 == ["F"] * 6, p0                              # λ=0：直走廊最短

    ex7 = Executor(b, path_gain_lambda=0.8)
    p1 = ex7.plan((6, 0))
    assert len(p1) == 11, p1                                # 多 5 原语但信息折现划算
    cells = _prims_to_cells((0, 0), "E", p1)
    assert any(y == 1 for (_, y) in cells)                  # 走了绕行走廊
    assert cells[-1] == (6, 0)
    # 手算代价：直 6×1.0=6.0；绕 3 转向 + 8×max(0.1,1−0.8)=3+8×0.2=4.6 < 6.0
    # 确定性：同快照重规划逐位一致
    assert ex7.plan((6, 0)) == p1
    # λ=0 加权函数退化为步数最短（长度等价）
    p2 = bfs_path_weighted(7, (0, 0), "E", (6, 0), b.open_boundary,
                           g, 0.0)
    assert len(p2) == 6
    # 步数单调：加权路径步数 ≥ 无权最短（边权 ≤ 1 的推论）
    assert len(p1) >= len(p0)
    print("[加权寻路] OK：λ=0 直走廊 / λ=0.8 绕行收割（4.6<6.0 手算）/ 确定性 / 步数单调")


def test_weighted_executor_episode_smoke():
    """真环境冒烟：λ=0.5 执行器 Explore 正常（可达目标成功 / 远目标不可达中止）。"""
    maze = empty_maze(5)
    env = _scan_env(maze, {"quyuan": (0, 2), "zhaojun": (3, 1)},
                    {"chuguo": (0, 4), "xihan": (1, 3)}, 5)
    obs, info = env.reset(seed=0)
    belief = BeliefState(n=5)
    belief.observe(obs, info)
    ex = Executor(belief, max_option_steps=200, path_gain_lambda=0.5)
    # reset 后可达集内目标（range-2 扫描揭示的范围）：贪心收割路径正常执行
    for target in [(1, 0), (2, 0), (1, 1), (0, 1)]:
        idx = explore_index(target, 5)
        r = ex.run(env, idx)
        assert r.success and not r.aborted, f"target={target}: {r}"
        assert r.steps >= 1
        belief.commit_option(r.path, r.start_cell)
        print(f"  target={target} steps={r.steps} success={r.success}")
    # 可达集外目标：悲观约定 → 不可达中止（失败信号交还高层，与 λ 无关）
    r_far = ex.run(env, explore_index((4, 4), 5))
    assert r_far.aborted and r_far.steps == 0
    print("[加权执行器冒烟] OK：λ=0.5 可达目标正常 / 远目标不可达中止（设计语义）")


if __name__ == "__main__":
    test_bfs_turn_counting()
    test_bfs_detour()
    test_executor_explore_options()
    test_executor_abort_unreachable()
    test_executor_abort_mid_option()
    test_executor_pick_and_deliver_flow()
    test_executor_step_accounting()
    test_weighted_path_handcheck()
    test_weighted_executor_episode_smoke()
    print("\n执行器单测全部通过 [PASS]")
