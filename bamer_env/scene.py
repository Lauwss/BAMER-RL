# -*- coding: utf-8 -*-
"""场景构建：迷宫结构 / 状态机位置 / 目的地终点 三大接口。

用法示例：
    from bamer_env.scene import build_scene

    # 1) 默认场景（PPT 图示：官方迷宫 + 屈原(2,11) + 昭君(11,12) + 楚国(8,8) + 西汉(7,7)）
    scene = build_scene()

    # 2) 自定义三个接口
    scene = build_scene(
        maze=Maze.random_maze(seed=7),                      # 迷宫结构接口
        state_machines={"quyuan": (5, 3), "zhaojun": (9, 13)},  # 状态机位置接口
        destinations={"chuguo": (8, 8), "xihan": (7, 7)},   # 目的地终点接口
        start=(0, 0),
    )

状态机位置与地图结构对机器鼠封闭（规则要求），Scene 中的这些字段仅供
环境侧/评估侧使用；机器鼠算法侧只能通过传感器接口逐步获取信息。
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from .maze import Maze

Cell = Tuple[int, int]

# PPT 图示的默认布点
DEFAULT_START: Cell = (0, 0)
DEFAULT_STATE_MACHINES: Dict[str, Cell] = {"quyuan": (2, 11), "zhaojun": (11, 12)}
DEFAULT_DESTINATIONS: Dict[str, Cell] = {"chuguo": (8, 8), "xihan": (7, 7)}
# 识别结果 -> 派送目的地（题目规则）
DEFAULT_DELIVERY_MAP: Dict[str, str] = {"quyuan": "chuguo", "zhaojun": "xihan"}

STATE_MACHINE_NAMES = ("quyuan", "zhaojun")


@dataclass
class Scene:
    """静态场景配置（环境侧持有，机器鼠不可直接读取）。"""

    maze: Maze
    start: Cell = DEFAULT_START
    start_heading: str = "N"  # PPT 图示起点箭头朝北（上）
    state_machines: Dict[str, Cell] = field(default_factory=lambda: dict(DEFAULT_STATE_MACHINES))
    destinations: Dict[str, Cell] = field(default_factory=lambda: dict(DEFAULT_DESTINATIONS))
    delivery_map: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_DELIVERY_MAP))

    def validate(self) -> None:
        n = self.maze.n
        if self.start_heading not in ("N", "S", "E", "W"):
            raise ValueError(f"start_heading 非法: {self.start_heading}")

        def check_cell(cell, what):
            if not (isinstance(cell, tuple) and len(cell) == 2):
                raise ValueError(f"{what} 必须是 (x, y) 二元组: {cell!r}")
            x, y = cell
            if not (0 <= x < n and 0 <= y < n):
                raise ValueError(f"{what} {cell} 超出 {n}×{n} 迷宫范围")

        check_cell(self.start, "start")
        for name, cell in self.state_machines.items():
            check_cell(cell, f"状态机 {name}")
        for name, cell in self.destinations.items():
            check_cell(cell, f"终点 {name}")

        if self.start in self.destinations.values():
            raise ValueError("起点不能与终点重合")

        unknown = set(self.delivery_map) - set(self.state_machines)
        if unknown:
            raise ValueError(f"delivery_map 引用了不存在的状态机: {unknown}")
        missing_dest = set(self.delivery_map.values()) - set(self.destinations)
        if missing_dest:
            raise ValueError(f"delivery_map 引用了不存在的终点: {missing_dest}")

        # 连通性：起点必须能到达全部状态机与终点（否则任务无解）。
        # 注意：迷宫允许存在与任务无关的封闭区域（官方 Micromouse 赛图
        # 含装饰性封闭块），不要求全图连通，只要求任务点互相可达。
        reach = self.maze.reachable_from(self.start)
        for label, cell in (
            [("start", self.start)]
            + [(f"状态机 {k}", v) for k, v in self.state_machines.items()]
            + [(f"终点 {k}", v) for k, v in self.destinations.items()]
        ):
            if cell not in reach:
                raise ValueError(f"{label} {cell} 从起点不可达，任务无解")


def build_scene(
    maze: Optional[Maze] = None,
    state_machines: Optional[Dict[str, Cell]] = None,
    destinations: Optional[Dict[str, Cell]] = None,
    start: Cell = DEFAULT_START,
    start_heading: str = "N",
    delivery_map: Optional[Dict[str, str]] = None,
    validate: bool = True,
) -> Scene:
    """构建场景。

    三大接口：
        maze            迷宫结构：Maze 实例（默认加载 PPT 官方迷宫）
        state_machines  状态机位置：{名称: (x, y)}，默认按 PPT 图示 (2,11)/(11,12)
        destinations    目的地终点：{名称: (x, y)}，默认 楚国(8,8)/西汉(7,7)

    validate=True 时校验坐标合法性与起点连通性（状态机/终点必须可达）。
    """
    if maze is None:
        maze = Maze.default()
    scene = Scene(
        maze=maze,
        start=tuple(start),
        start_heading=start_heading,
        state_machines=dict(state_machines) if state_machines else dict(DEFAULT_STATE_MACHINES),
        destinations=dict(destinations) if destinations else dict(DEFAULT_DESTINATIONS),
        delivery_map=dict(delivery_map) if delivery_map else dict(DEFAULT_DELIVERY_MAP),
    )
    if validate:
        scene.validate()
    return scene
