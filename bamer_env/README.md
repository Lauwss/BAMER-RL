# bamer_env — 场景构建模块

按 PPT《机器人与人工智能》第一页场景图示构建的 16×16 迷宫场景，提供题目要求的三大接口。

## 三大接口

```python
from bamer_env import Maze, build_scene, render_scene

# 默认场景（PPT 图示原样复刻）
scene = build_scene()

# 1) 迷宫结构接口：Maze 实例
maze = Maze.load("my_maze.json")            # 从 JSON 加载
maze = Maze.random_maze(seed=42, braiding=0.1)  # 随机生成（训练多样化）

# 2) 状态机位置接口
# 3) 目的地终点接口
scene = build_scene(
    maze=maze,
    state_machines={"quyuan": (5, 3), "zhaojun": (9, 13)},
    destinations={"chuguo": (8, 8), "xihan": (7, 7)},
    start=(0, 0),
)
scene.validate()        # 坐标合法性 + 任务点可达性校验
render_scene(scene)     # 渲染成图（PPT 风格，图标取自原 PPT 素材）
```

## Gymnasium 环境（M1.2，占位契约）

```python
from bamer_env import BamerEnv

env = BamerEnv()                      # 默认 PPT 场景
env = BamerEnv(randomize=True)        # 每次 reset 随机迷宫 + 随机布点
obs, info = env.reset(seed=42)        # 可复现（迷宫/布点全走 np_random 流）
obs, r, term, trunc, info = env.step(0)   # 0=F 1=TL 2=TR 3=TB（D5 原语占位）
env.reset(options={"state_machines": {"quyuan": (5,3), "zhaojun": (9,13)}})
                                      # R2 代码冻结注入接口（不改代码换布点）
```

- 占位观测：`cell`（y*n+x）、`heading`（0=N 1=E 2=S 3=W）、`identified`/`delivered`
  （MultiBinary，按状态机名字典序）——只含机器鼠自身状态与交互反馈（R1）；
- `info`：steps / events / identified / delivered，无场景内部字段；
- `terminated` = 两项派送达成；`truncated` = 超 `max_steps`；每动作恰计 1 步（含旋转/碰撞）；
- 占位奖励 -1/步（D8 待用户设计）；派送触发/碰撞语义占位（见 `docs/CHANGELOG.md`）。

## 交互协议与展示层（M1.1 / M1.3）

```python
from bamer_env import EpisodeAnimator, record_episode

def my_policy(obs, info) -> int:
    ...  # 演示用占位策略示例见 tools/demo_exploration.py（仅用合法自有信息）

env = BamerEnv(max_steps=1500)
frames = record_episode(env, my_policy)
EpisodeAnimator(env.scene, frames, max_steps=env.max_steps).save_gif(
    "docs/_demo_exploration.gif", fps=12, stride=5)
```

- `interaction.py`：6 类事件（Reported/ImageFeed/Arrived/Collision/TaskComplete/Timeout），
  经 `info["events"]` 透传；
- `viewer.py`：动画 + 步数 HUD + 识别图像面板 + 点云挂点（`Frame.points`，
  阶段二 M2.2/M2.3 接通后传入即叠加）。

## 坐标约定

与 PPT 一致：16×16 格子，x 向右、y 向上，0~15；起点 (0,0) 在左下角，朝北。

## 文件说明

| 文件 | 内容 |
| --- | --- |
| `maze.py` | `Maze` 数据模型：墙线存储、感知接口 `has_wall`、连通性、随机生成、序列化 |
| `scene.py` | `Scene` / `build_scene`：三大接口组装 + 校验（越界/重合/不可达） |
| `interaction.py` | M1.1 状态机双向交互协议：事件定义 + 上报/图像下发 + 识别/派送簿记 |
| `env.py` | M1.2 Gymnasium 封装 `BamerEnv`：占位 obs/action、随机化、R2 注入接口 |
| `viewer.py` | M1.3 展示层：逐帧动画 / 步数 HUD / 识别图像面板 / 点云挂点 |
| `render.py` | PPT 风格静态渲染（迷宫墙线、图标、起点箭头） |
| `default_maze.json` | 官方 2022 日本东北大会 Fresher 迷宫底图（程序化提取自 PPT 内嵌原图） |
| `default_maze_ppt.json` | **PPT 最终场景** = 底图 + PPT 虚线标记的 13 处开口（`Maze.default()` 加载此文件） |

## 重要背景（复现细节）

- PPT 迷宫底图是 2022 年日本东北地区大会 Classic Mouse Fresher 官方赛图，
  图注"西回 45 步 16 折"已用于交叉验证提取精度（BFS 46 步 16 折，折数一致）。
- PPT 第一页叠加的 13 条虚线段 = 在官方图上开通的墙口（迷宫图纸惯例），
  开通后起点→(7,7) 最短 16 步。提取脚本见 `../tools/extract_maze.py` 与
  `../tools/extract_openings.py`。
- 官方赛图本身含多处**装饰性封闭区域**（不可达格块），这是迷宫设计的一部分，
  非提取错误；场景校验只要求任务点（起点/状态机/终点）互相可达。

## 运行测试

```
python tools/test_scene.py          # 场景验收（既有）
python tests/test_interaction.py    # M1.1 交互协议单测
python tests/test_env.py            # M1.2 Env 单测（含 100 种子验收）
python tools/demo_exploration.py    # M1.3 探索录像 GIF → docs/_demo_exploration.gif
python tools/demo_observation_samples.py   # D7 三方案样例 → docs/d7_samples/
```
