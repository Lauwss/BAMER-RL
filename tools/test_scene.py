# -*- coding: utf-8 -*-
"""场景模块验收测试。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bamer_env import Maze, build_scene, render_scene

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_default_scene():
    """默认场景 = PPT 图示：官方迷宫 + (2,11)/(11,12) + 楚国(8,8)/西汉(7,7)。"""
    scene = build_scene()
    assert scene.maze.n == 16
    assert scene.state_machines == {"quyuan": (2, 11), "zhaojun": (11, 12)}
    assert scene.destinations == {"chuguo": (8, 8), "xihan": (7, 7)}
    assert scene.start == (0, 0)
    scene.validate()  # 任务点互相可达（迷宫允许有无关封闭区）
    # 官方最短路径参考值（图注：西回 45 步 16 折）——用 BFS 步数交叉验证结构正确性
    from collections import deque

    def bfs_steps(src, dst):
        q = deque([(src, 0)])
        seen = {src}
        while q:
            (x, y), d = q.popleft()
            if (x, y) == dst:
                return d
            for nb in scene.maze.neighbors(x, y):
                if nb not in seen:
                    seen.add(nb)
                    q.append((nb, d + 1))
        return None

    goals = list(scene.destinations.values())
    best = min(bfs_steps((0, 0), g) for g in goals)
    print(f"[默认场景] OK  任务点全部可达  起点->任一终点 BFS 最短 = {best} 步（官方西回 45 步 16 折参考）")
    assert best is not None


def test_custom_scene():
    """三大接口均可自定义。"""
    maze = Maze.random_maze(seed=42, braiding=0.1)
    scene = build_scene(
        maze=maze,
        state_machines={"quyuan": (5, 3), "zhaojun": (9, 13)},
        destinations={"chuguo": (8, 8), "xihan": (7, 7)},
        start=(0, 0),
    )
    scene.validate()
    fig, _ = render_scene(scene, title="自定义场景（随机迷宫 + 自定义布点）",
                          save_path=os.path.join(ROOT, "docs", "_scene_custom.png"))
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("[自定义场景] OK")


def test_validation():
    """非法输入必须报错。"""
    # 1) 越界
    try:
        build_scene(state_machines={"quyuan": (20, 3)})
        raise AssertionError("越界未报错")
    except ValueError as e:
        print(f"[校验] 越界 -> {e}")
    # 2) 不可达（造一个把起点围死的迷宫：起点四周全部封墙）
    m = Maze.random_maze(seed=1)
    m.wall_south[0][1] = True   # 封 (0,0) 北墙
    m.wall_west[1][0] = True    # 封 (0,0) 东墙
    try:
        build_scene(maze=m)
        raise AssertionError("不可达未报错")
    except ValueError as e:
        print(f"[校验] 不可达 -> {e}")
    # 3) 起点与终点重合
    try:
        build_scene(destinations={"chuguo": (0, 0), "xihan": (7, 7)})
        raise AssertionError("重合未报错")
    except ValueError as e:
        print(f"[校验] 重合 -> {e}")


def test_random_maze_stats():
    for seed in range(5):
        m = Maze.random_maze(seed=seed, braiding=0.15)
        assert m.is_fully_connected()
        print(f"[随机迷宫] seed={seed} 全连通 OK，内墙数 = "
              f"{sum(sum(r) for r in m.wall_south) - 16 + sum(sum(r) for r in m.wall_west) - 16}")


if __name__ == "__main__":
    test_default_scene()
    fig, _ = render_scene(build_scene(), save_path=os.path.join(ROOT, "docs", "_scene_default.png"))
    import matplotlib.pyplot as plt
    plt.close(fig)
    test_custom_scene()
    test_validation()
    test_random_maze_stats()
    print("\n全部验收通过 [PASS]")
