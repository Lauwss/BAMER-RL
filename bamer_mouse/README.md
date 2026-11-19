# bamer_mouse — 机器鼠层（PLAN.md 阶段二）

机器鼠的物理与传感仿真：运动学、激光雷达、点云/占据栅格、数据 IO、识别挂接。
本包**不 import bamer_env**（单向依赖）：墙体查询以 `wall_query(x, y, direction)->bool`
回调注入，传感器输出即算法侧唯一合法信息通道（AGENT.md R1 信息封闭），
且便于阶段四 MATLAB 逐行移植。

| 模块 | 里程碑 | 内容 |
| --- | --- | --- |
| `kinematics.py` | M2.1 | `Pose` 位姿、D5 原语 {F,TL,TR,TB}、碰撞判定 `try_move`、动画插值 `interpolate`（格单位，D3） |
| `lidar.py` | M2.2 | `LidarSpec` 参数表（D4 默认：360°/180 束/2.0 格/噪声关）、DDA raycast、`Lidar.scan` 全扫描（机体系输出，NaN=无回波） |
| `pointcloud.py` | M2.3 | 坐标链（极坐标→机体系→世界系）、局部占据栅格 `occupancy_grid`（D7-B，仅凭量测复算）、呈现（极坐标视图/点云残影） |
| `dataio.py` | M2.4 | `EpisodeRecord`（meta+逐步）、npz+json 存取、迷宫哈希 |
| `camera.py` | M2.5 | `Classifier` 协议（可插拔）、GT 直通占位（D9 待定）、`CameraUnit` 识别流水线 |

## 用法示例

```python
from bamer_env.env import BamerEnv          # 场景层（Env 侧接通雷达）
from bamer_mouse import LidarSpec

env = BamerEnv(lidar=LidarSpec())           # 默认关闭；传入 spec 即接通
obs, info = env.reset(seed=42)
obs, r, term, trunc, info = env.step(0)     # info["ranges"/"angles"/"points"] 传感器观测
```

```python
from bamer_env import Maze
from bamer_mouse import Lidar, LidarSpec, pose_at, occupancy_grid, try_move

maze = Maze.default()                       # wall_query 回调注入（解耦）
lidar = Lidar(LidarSpec(), seed=0)
pose = pose_at((2, 11), "N")
scan = lidar.scan(maze.has_wall, pose)      # 机体系 ranges/angles/hit_xy
grid = occupancy_grid(scan.ranges, scan.angles, pose, LidarSpec().max_range, ego=True)
out = try_move(pose, "F", maze.has_wall)    # out.moved / out.direction（碰撞语义）
```

## 验收与测试

- `tests/test_kinematics.py`：原语序列解析解对拍、逻辑格同步、与 BamerEnv 交叉验证、碰撞、插值；
- `tests/test_lidar.py`：raycast 解析解（空房/内墙/30°/45° 斜射/角点 tie）、开口穿透、噪声可复现、性能；
- `tests/test_pointcloud.py`：坐标链、占据栅格三值语义、**量测数据 ↔ 墙查询逐格等价（R1 自洽实证）**、ego 旋转、D7 复核；
- `tests/test_dataio.py`：写→读往返逐字段一致、迷宫哈希；
- `tests/test_camera.py`：协议挂接、GT 直通、误识如实记录；
- `tests/test_env_sensor.py`：Env 接雷达的 info 键集/扫描时机/信息封闭/含噪可复现/默认关闭零侵入；
- `tools/demo_lidar_pointcloud.py` → `docs/lidar_*.png`（墙前贴墙 / 开口穿透 / 走廊纵深 / 点云残影）+ `docs/_demo_pointcloud.gif`（带实时点云完整任务录像）。

## 契约

物理/几何语义与传感器默认值见 [../docs/contract_v1.md](../docs/contract_v1.md)（M2.6，
观测/动作/奖励三章待用户填写后冻结）。
