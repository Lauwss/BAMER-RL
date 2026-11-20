# -*- coding: utf-8 -*-
"""全知参照单测：强制语义自动机（规则 A）与自由序对照——验证参照严格匹配 env 语义。"""
import os
import sys
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from algo.actions import DELIVERY_MAP
from algo.belief import HEADINGS
from algo.eval import omniscient_reference
from bamer_env.scene import build_scene
from tests.test_executor import wall_grid


def _free_order_reference(maze, machines, dests, start, start_heading):
    """测试内对照实现：自由任务序参照（两条合法序的 staged BFS 取最小，
    不建模强制拾取——用于对照强制语义参照，验证差异捕捉）。"""
    n = maze.n
    open_gt = lambda cell, d: not maze.has_wall(cell[0], cell[1], d)

    def staged(waypoints):
        h0 = HEADINGS.index(start_heading)
        s0 = (start[0], start[1], h0, 0)
        seen = {s0}
        dq = deque([(s0, 0)])
        while dq:
            (x, y, h, k), cost = dq.popleft()
            if k < len(waypoints) and (x, y) == waypoints[k]:
                k += 1
                if k == len(waypoints):
                    return cost
            for d, (dx, dy) in zip(HEADINGS, ((0, 1), (1, 0), (0, -1), (-1, 0))):
                if d != HEADINGS[h]:
                    continue                        # F 仅沿当前朝向（R4：转向另计步）
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n and open_gt((x, y), d) \
                        and (nx, ny, h, k) not in seen:
                    seen.add((nx, ny, h, k))
                    dq.append(((nx, ny, h, k), cost + 1))
            for dh in (-1, +1, +2):
                if (x, y, (h + dh) % 4, k) not in seen:
                    seen.add((x, y, (h + dh) % 4, k))
                    dq.append(((x, y, (h + dh) % 4, k), cost + 1))
        return None

    orders = [["quyuan", "chuguo", "zhaojun", "xihan"],
              ["zhaojun", "xihan", "quyuan", "chuguo"]]
    best = None
    for order in orders:
        pts = [machines[m] if m in machines else dests[m] for m in order]
        cost = staged(pts)
        if cost is not None and (best is None or cost < best):
            best = cost
    return best


def _scene_of(maze, machines, dests):
    return build_scene(maze=maze, state_machines=machines, destinations=dests)


def test_reference_matches_env_semantics_no_conflict():
    """无冲突拓扑（机器不挡路）：强制语义参照 == 自由序参照。"""
    maze = __import__("bamer_env", fromlist=["Maze"]).Maze(
        [[False] * 5 for _ in range(5)], [[False] * 5 for _ in range(5)])
    machines = {"quyuan": (0, 2), "zhaojun": (2, 2)}
    dests = {"chuguo": (4, 2), "xihan": (4, 3)}
    scene = _scene_of(maze, machines, dests)
    ref_forced = omniscient_reference(scene)
    ref_free = _free_order_reference(maze, machines, dests, scene.start, scene.start_heading)
    assert ref_forced == ref_free, (ref_forced, ref_free)
    assert ref_forced > 0
    print(f"[无冲突] OK：强制语义参照 == 自由序参照 = {ref_forced} 步")


def test_reference_captures_forced_pickup_cost():
    """挡路拓扑（程序化搜索锁定的实例）：屈原挡在通往昭君的唯一走廊上 →
    自由序最优为"先送昭君"，但强制拾取序被迫"先送屈原" → 强制语义参照
    严格大于自由序参照（规则 A 的拓扑代价被如实计入参照）。"""
    # 走廊 (0..6,0) + 支路 (1,0)→(1,3) + 西汉支路 (0,1)；屈原(2,0) 挡路，昭君(6,0) 在尽头
    walls = [((x, 0), "E") for x in range(0, 6)] \
        + [((1, y), "N") for y in range(0, 3)] + [((0, 0), "N")]
    maze = wall_grid(7, walls)
    machines = {"quyuan": (2, 0), "zhaojun": (6, 0)}
    dests = {"chuguo": (1, 3), "xihan": (0, 1)}
    scene = _scene_of(maze, machines, dests)
    ref_forced = omniscient_reference(scene)
    ref_free = _free_order_reference(maze, machines, dests, scene.start, scene.start_heading)
    assert (ref_forced, ref_free) == (28, 27), (ref_forced, ref_free)
    assert ref_forced > ref_free
    print(f"[挡路拓扑] OK：强制语义参照 {ref_forced} > 自由序参照 {ref_free} 步"
          f"（规则 A 拓扑代价 = {ref_forced - ref_free} 步，程序化搜索锁定实例）")


def test_reference_official_maze():
    """官方 2022 赛图：参照 = 72 步（(格,朝向,强制事件) 全语义最优；
    旧实现漏转向代价的 50 为错误值，已修复）。"""
    scene = build_scene()
    ref = omniscient_reference(scene)
    assert ref == 72, ref
    print(f"[官方图] OK：全知参照 = {ref} 步（转向计步 + 强制事件语义）")


if __name__ == "__main__":
    test_reference_matches_env_semantics_no_conflict()
    test_reference_captures_forced_pickup_cost()
    test_reference_official_maze()
    print("\n全知参照单测全部通过 [PASS]")
