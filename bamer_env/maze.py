# -*- coding: utf-8 -*-
"""迷宫结构数据模型。

坐标约定（与 PPT 一致）：16×16 格子，x 向右 0~15，y 向上 0~15，
起点 (0,0) 位于左下角。

墙线采用"南墙 + 西墙"双数组存储：
    wall_south[x][y]  格 (x,y) 与 (x,y-1) 之间的墙；y=0 时为南外边框
    wall_west [x][y]  格 (x,y) 与 (x-1,y) 之间的墙；x=0 时为西外边框
北墙/东墙由相邻格的南墙/西墙推导，最外圈恒为 True（封闭外边框）。
"""
import json
import os
import random
from typing import Dict, Iterable, List, Optional, Set, Tuple

# 方向：dy/dx 与字符串方向互相对应
DIRS = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}

Cell = Tuple[int, int]


class Maze:
    """16×16（或自定义 N×N）迷宫。不可直接修改墙线，探索用感知接口见 has_wall。"""

    def __init__(self, wall_south: List[List[bool]], wall_west: List[List[bool]], name: str = ""):
        n = len(wall_south)
        if len(wall_west) != n or any(len(r) != n for r in wall_south) or any(len(r) != n for r in wall_west):
            raise ValueError("墙线数组必须都是 N×N 方阵")
        if n < 2:
            raise ValueError("迷宫尺寸至少为 2")
        self.n = n
        self.name = name
        self.wall_south = [[bool(v) for v in row] for row in wall_south]
        self.wall_west = [[bool(v) for v in row] for row in wall_west]
        # 外边框强制封闭
        for x in range(n):
            self.wall_south[x][0] = True
        for y in range(n):
            self.wall_west[0][y] = True

    # ---------- 墙线查询（机器人感知视角的唯一合法入口） ----------
    def has_wall(self, x: int, y: int, direction: str) -> bool:
        """格 (x,y) 在 direction 方向上是否有墙。"""
        if direction == "S":
            return self.wall_south[x][y]
        if direction == "W":
            return self.wall_west[x][y]
        if direction == "N":
            return self.wall_south[x][y + 1] if y + 1 < self.n else True
        if direction == "E":
            return self.wall_west[x + 1][y] if x + 1 < self.n else True
        raise ValueError(f"未知方向: {direction}")

    def neighbors(self, x: int, y: int) -> List[Cell]:
        """从 (x,y) 出发无墙可直达的相邻格列表。"""
        out = []
        for d, (dx, dy) in DIRS.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.n and 0 <= ny < self.n and not self.has_wall(x, y, d):
                out.append((nx, ny))
        return out

    # ---------- 拓扑分析 ----------
    def reachable_from(self, start: Cell) -> Set[Cell]:
        seen = {start}
        stack = [start]
        while stack:
            cx, cy = stack.pop()
            for nb in self.neighbors(cx, cy):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return seen

    def is_fully_connected(self) -> bool:
        return len(self.reachable_from((0, 0))) == self.n * self.n

    # ---------- 序列化 ----------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "size": self.n,
            "wall_south": self.wall_south,
            "wall_west": self.wall_west,
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: str) -> "Maze":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["wall_south"], d["wall_west"], name=d.get("name", ""))

    @classmethod
    def default(cls) -> "Maze":
        """PPT 场景迷宫 = 官方 2022 东北大会 Fresher 赛图 + PPT 虚线标记的 13 处开口。

        底图原始提取数据见 default_maze.json；本文件 default_maze_ppt.json
        是按 PPT 第一页虚线开口开通后的最终场景迷宫（全连通）。
        """
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default_maze_ppt.json")
        if not os.path.exists(path):  # 兼容：缺省时退回未开孔底图
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default_maze.json")
        return cls.load(path)

    # ---------- 随机迷宫（用于训练多样化，递归回溯法） ----------
    @staticmethod
    def _wall_between(south: List[List[bool]], west: List[List[bool]], n: int,
                      x: int, y: int, d: str) -> bool:
        if d == "S":
            return south[x][y] or y == 0
        if d == "N":
            return south[x][y + 1] if y + 1 < n else True
        if d == "W":
            return west[x][y] or x == 0
        if d == "E":
            return west[x + 1][y] if x + 1 < n else True
        raise ValueError(d)

    @classmethod
    def random_maze(cls, n: int = 16, seed: Optional[int] = None, braiding: float = 0.0) -> "Maze":
        """生成随机迷宫：初始四面皆墙，递归回溯挖出通道。

        braiding: 0~1，对死胡同额外打通一个开口的概率，形成多路径迷宫。
        """
        rng = random.Random(seed)
        south = [[True] * n for _ in range(n)]
        west = [[True] * n for _ in range(n)]

        def open_between(x: int, y: int, d: str) -> bool:
            dx, dy = DIRS[d]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < n and 0 <= ny < n):
                return False
            if d == "S":
                south[x][y] = False          # (x,y) 与 (x,y-1) 之间
            elif d == "N":
                south[nx][ny] = False        # 即 south[x][y+1]
            elif d == "W":
                west[x][y] = False           # (x,y) 与 (x-1,y) 之间
            else:  # E
                west[nx][ny] = False         # 即 west[x+1][y]
            return True

        visited = [[False] * n for _ in range(n)]
        visited[0][0] = True
        stack: List[Cell] = [(0, 0)]
        while stack:
            x, y = stack[-1]
            options = [
                (d, x + dx, y + dy)
                for d, (dx, dy) in DIRS.items()
                if 0 <= x + dx < n and 0 <= y + dy < n and not visited[x + dx][y + dy]
            ]
            if not options:
                stack.pop()
                continue
            d, nx, ny = rng.choice(options)
            open_between(x, y, d)
            visited[nx][ny] = True
            stack.append((nx, ny))

        if braiding > 0:
            for x in range(n):
                for y in range(n):
                    open_exits = sum(
                        1 for d in DIRS if not cls._wall_between(south, west, n, x, y, d)
                    )
                    if open_exits == 1 and rng.random() < braiding:
                        closed = [d for d in DIRS if cls._wall_between(south, west, n, x, y, d)]
                        rng.shuffle(closed)
                        for d in closed:
                            if open_between(x, y, d):
                                break

        return cls(south, west, name=f"random-{seed}")
