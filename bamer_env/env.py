# -*- coding: utf-8 -*-
"""Gymnasium 环境封装（M1.2，PLAN.md 阶段一）。

用法示例：
    from bamer_env.env import BamerEnv

    env = BamerEnv()                       # 默认 PPT 场景（官方迷宫 + 默认布点）
    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(0)   # 0=F 1=TL 2=TR 3=TB

    env = BamerEnv(randomize=True)         # 每次 reset 随机迷宫 + 随机布点（种子走 reset(seed=)）
    env = BamerEnv(max_steps=500)

    # R2 代码冻结接口：测试开始后不改代码，经 reset(options=...) 注入新布点
    env.reset(options={"state_machines": {"quyuan": (5, 3), "zhaojun": (9, 13)}})

**占位契约声明（M2.6 契约冻结前不代表最终设计）**：
- 动作空间：Discrete(4) = D5 低层原语 {F 前进一格, TL 左转, TR 右转, TB 后转}，
  每个动作恰计 1 step（含旋转与碰撞，符合题目运动模型铁律 R4）；
  决策级动作空间（D6）待用户拍板后再包装。
- 观测空间：格子级离散 Dict —— cell（格序号 y*n+x）、heading（0=N 1=E 2=S 3=W）、
  identified/delivered（MultiBinary，按状态机名字典序）。这是机器鼠自身状态与
  交互反馈，不含迷宫结构/状态机位置（R1 信息封闭）；激光点云接通后（M2.2/M2.3）
  观测将重构为 D7 已选定的局部占据栅格表示（2026-11-19 用户拍板）。
- 奖励：reward = step_reward（默认 -1.0/步，纯时间惩罚占位）——奖励函数 D8
  归用户设计，本数值仅为阶段一自测服务。
- 终止语义：terminated = 两项派送均达成（TaskComplete）；
  truncated = 达到 max_steps（Timeout）。碰撞只产生事件与 step 计数，不终止。

**传感器套件（M2.2/M2.3 接通，可选，默认关闭）**：
- 构造参数 `lidar=LidarSpec(...)`（默认 None = 不接雷达，行为与阶段一验收版
  完全一致）；开启后每个 step 末尾（机器鼠静止于新格中心，规范位姿）采一帧
  全扫描，reset 时在出生格采第 0 帧；
- info 附加键：`ranges`(n,) / `angles`(n,)（机体系）/ `points`(n,2)（世界系
  点云，与束一一对应，NaN=无回波）——均为雷达量测数据，算法侧合法信息
  通道（R1 自检通过）；
- 雷达随机流独立于环境流，但由 reset 种子派生重置：同 reset(seed) 的整个
  episode 含噪声逐位可复现；
- 观测空间重构（D7 局部占据栅格进入 obs）属 M2.6 契约冻结事项，本阶段不变。

info 字典（算法侧合法信息，R1 自检通过）：steps / events / identified / delivered
（+ 接雷达时 ranges / angles / points），
无任何场景内部字段（迷宫墙线、状态机位置、delivery_map 均不出现在 obs/info 中）。
`env.scene` 为环境侧/评估侧专用（渲染、验收），算法侧不得读取。
"""
from typing import Dict, List, Optional, Tuple

import gymnasium
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces

from .interaction import Event, InteractionManager
from .maze import DIRS, Maze
from .scene import (
    DEFAULT_DESTINATIONS,
    DEFAULT_STATE_MACHINES,
    DEFAULT_START,
    build_scene,
)
# 传感器套件（M2.2/M2.3，可选接通）：bamer_mouse 不反向依赖 bamer_env，无循环导入
from bamer_mouse.kinematics import pose_at
from bamer_mouse.lidar import Lidar, LidarSpec, ScanResult
from bamer_mouse.pointcloud import scan_to_world

# 朝向离散化与 D5 原语（物理层，决策层可再包装）
HEADINGS = ("N", "E", "S", "W")          # 0=N 1=E 2=S 3=W（顺时针）
PRIMITIVES = ("F", "TL", "TR", "TB")     # 0=F 1=TL 2=TR 3=TB
_TURN_DELTA = {"TL": -1, "TR": +1, "TB": +2}   # 作用在 HEADINGS 下标上（模 4）

Cell = Tuple[int, int]


class BamerEnv(gymnasium.Env):
    """机器鼠迷宫探索与派送环境（阶段一占位契约）。"""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        maze: Optional[Maze] = None,
        state_machines: Optional[Dict[str, Cell]] = None,
        destinations: Optional[Dict[str, Cell]] = None,
        start: Cell = DEFAULT_START,
        start_heading: str = "N",
        delivery_map: Optional[Dict[str, str]] = None,
        randomize: bool = False,
        max_steps: int = 1000,
        step_reward: float = -1.0,
        render_mode: Optional[str] = None,
        lidar: Optional[LidarSpec] = None,
        lidar_seed: Optional[int] = None,
    ):
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"render_mode 仅支持 None/'rgb_array'，得到 {render_mode!r}")
        self._maze_fixed = maze
        self._machines_fixed = dict(state_machines) if state_machines else None
        self._destinations_fixed = dict(destinations) if destinations else None
        self._delivery_map_fixed = dict(delivery_map) if delivery_map else None
        self._start = tuple(start)
        self._start_heading = start_heading
        self.randomize = bool(randomize)
        self.max_steps = int(max_steps)
        self.step_reward = float(step_reward)
        self.render_mode = render_mode
        # 传感器套件（M2.2，可选）：默认 None = 不接雷达，与阶段一验收行为一致
        self._lidar = Lidar(lidar, seed=lidar_seed) if lidar is not None else None

        n = maze.n if maze is not None else 16
        self._n = int(n)
        self._n_machines = len(state_machines) if state_machines else len(DEFAULT_STATE_MACHINES)

        # 占位空间（见模块 docstring 契约声明）
        self.action_space = spaces.Discrete(len(PRIMITIVES))
        self.observation_space = spaces.Dict({
            "cell": spaces.Discrete(self._n * self._n),
            "heading": spaces.Discrete(len(HEADINGS)),
            "identified": spaces.MultiBinary(self._n_machines),
            "delivered": spaces.MultiBinary(self._n_machines),
        })

        # 运行期状态（reset 前为 None）
        self.scene: Optional[object] = None   # 环境侧专用（R1：算法侧不得读取）
        self._interaction: Optional[InteractionManager] = None
        self._cell: Optional[Cell] = None
        self._heading: int = 0
        self._steps: int = 0
        self._visited: List[Cell] = []        # 渲染用轨迹（环境侧）

    # ---------- Gymnasium 接口 ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if self._lidar is not None:
            # 雷达独立流由环境种子派生重置：同 reset(seed) → 全 episode 含噪声逐位可复现
            self._lidar.reseed(int(self.np_random.integers(0, 2**31 - 1)))
        options = dict(options or {})

        # 场景来源优先级：reset options 注入（R2 代码冻结接口） > randomize > 构造参数/默认
        scene_kwargs: Dict[str, object] = {}
        if self.randomize:
            maze_seed = int(self.np_random.integers(0, 2**31 - 1))
            maze = Maze.random_maze(seed=maze_seed)
            dests = dict(self._destinations_fixed) if self._destinations_fixed else dict(DEFAULT_DESTINATIONS)
            names = sorted(DEFAULT_STATE_MACHINES)
            candidates = sorted(set(maze.reachable_from(self._start)) - {self._start} - set(dests.values()))
            if len(candidates) < len(names):
                raise ValueError("随机布点可用格不足（迷宫太小或终点占用过多）")
            picks = self.np_random.choice(len(candidates), size=len(names), replace=False)
            scene_kwargs.update(
                maze=maze,
                destinations=dests,
                state_machines={nm: candidates[int(i)] for nm, i in zip(names, picks)},
            )
        else:
            if self._maze_fixed is not None:
                scene_kwargs["maze"] = self._maze_fixed
            if self._machines_fixed is not None:
                scene_kwargs["state_machines"] = dict(self._machines_fixed)
            if self._destinations_fixed is not None:
                scene_kwargs["destinations"] = dict(self._destinations_fixed)
        for key in ("maze", "state_machines", "destinations"):
            if options.get(key) is not None:
                scene_kwargs[key] = options[key]

        self.scene = build_scene(
            start=self._start,
            start_heading=self._start_heading,
            delivery_map=self._delivery_map_fixed,
            **scene_kwargs,
        )
        if self.scene.maze.n != self._n:
            raise ValueError(
                f"reset 迷宫尺寸 {self.scene.maze.n} 与 observation_space 尺寸 {self._n} 不一致")

        self._interaction = InteractionManager(self.scene)
        self._cell = tuple(self.scene.start)
        self._heading = HEADINGS.index(self.scene.start_heading)
        self._steps = 0
        self._visited = [self._cell]
        # 出生格即状态机格时，状态机立即下发图像（step 0）
        events = list(self._interaction.enter_events(0, self._cell, None))
        return self._get_obs(), self._info(events, scan=self._scan_now())

    def step(self, action):
        if self._interaction is None:
            raise RuntimeError("请先调用 reset() 再 step()")
        primitive = self._resolve_action(action)
        step_no = self._steps + 1     # 本动作对应的步号（reset 后从 1 起）

        # 每步实时上报自身位置（题目规则；旋转/碰撞步同样上报）
        events: List[Event] = [self._interaction.report(step_no, self._cell)]
        x, y = self._cell
        if primitive == "F":
            heading = HEADINGS[self._heading]
            if self.scene.maze.has_wall(x, y, heading):
                # 碰撞：位置不变，计 1 步，只产生事件（终止/惩罚语义随用户奖励设计定）
                self._steps = step_no
                events.append(self._interaction.collision_event(step_no, self._cell, heading))
            else:
                dx, dy = DIRS[heading]
                prev, self._cell = self._cell, (x + dx, y + dy)
                self._steps = step_no
                events.extend(self._interaction.enter_events(step_no, self._cell, prev))
        else:
            self._heading = (self._heading + _TURN_DELTA[primitive]) % 4
            self._steps = step_no

        if self._cell != self._visited[-1]:
            self._visited.append(self._cell)

        terminated = self._interaction.task_complete
        truncated = (not terminated) and self._steps >= self.max_steps
        if truncated:
            events.append(self._interaction.timeout_event(self._steps, self.max_steps))

        reward = self.step_reward   # 占位奖励（D8 归用户设计）
        return self._get_obs(), reward, terminated, truncated, self._info(events, scan=self._scan_now())

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        from .viewer import draw_snapshot  # 展示层复用（避免不必要的高层依赖倒置）

        fig = draw_snapshot(
            self.scene,
            pose_cell=self._cell,
            heading=HEADINGS[self._heading],
            trail_cells=self._visited,
            steps=self._steps,
            max_steps=self.max_steps,
            identified=self._interaction.identified,
            delivered=self._interaction.delivered,
        )
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        plt.close(fig)
        return buf[..., :3].copy()

    def close(self):
        plt.close("all")

    # ---------- 只读状态（机器鼠自身信息，算法侧/展示层均可合法使用） ----------
    @property
    def pose_cell(self) -> Cell:
        return self._cell

    @property
    def heading(self) -> str:
        return HEADINGS[self._heading]

    @property
    def visited_cells(self) -> List[Cell]:
        return list(self._visited)

    # ---------- 内部 ----------
    def _resolve_action(self, action) -> str:
        if isinstance(action, str):
            if action not in PRIMITIVES:
                raise ValueError(f"未知动作原语: {action!r}，可选 {PRIMITIVES}")
            return action
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"动作越界: {action}，合法范围 [0, {len(PRIMITIVES)})")
        return PRIMITIVES[action]

    def _get_obs(self) -> dict:
        n = self.scene.maze.n
        x, y = self._cell
        machines = sorted(self.scene.state_machines)
        return {
            "cell": int(y * n + x),
            "heading": int(self._heading),
            "identified": np.array(
                [m in self._interaction.identified for m in machines], dtype=np.int8),
            "delivered": np.array(
                [m in self._interaction.delivered for m in machines], dtype=np.int8),
        }

    def _info(self, events: List[Event], scan: Optional[ScanResult] = None) -> dict:
        # R1 自检：此处不得加入 scene.maze / state_machines / delivery_map 等场景内部字段
        info = {
            "steps": int(self._steps),
            "events": events,
            "identified": sorted(self._interaction.identified),
            "delivered": sorted(self._interaction.delivered),
        }
        if scan is not None:
            # 接通雷达时附加传感器观测（M2.2）：量测数据 = 算法侧合法信息通道
            pose = pose_at(self._cell, HEADINGS[self._heading])
            info["ranges"] = scan.ranges
            info["angles"] = scan.angles
            # 世界系点云与束一一对应（(n,2)，NaN=无回波）：与 ranges/angles 对齐，
            # 便于 RL/存档定长处理；展示层渲染时自行过滤 NaN。（丢弃语义随 M2.6 定稿）
            info["points"] = scan_to_world(scan, pose, drop_no_echo=False)
        return info

    def _scan_now(self) -> Optional[ScanResult]:
        """当前规范位姿（格中心）的一帧全扫描；未接雷达返回 None。"""
        if self._lidar is None:
            return None
        pose = pose_at(self._cell, HEADINGS[self._heading])
        return self._lidar.scan(self.scene.maze.has_wall, pose)
