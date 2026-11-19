# CHANGELOG

> 契约（M2.6）冻结后的任何语义变更须先经用户确认，再记入本文件并同步单测。

## 2026-11-20 — 阶段三算法设计定稿（D6/D7 参数/D8/D10）+ 契约三章填写

### 新增

- `docs/rl_algorithm_design.md`（阶段三开工前置 #2：用户设计决策的书面规格）：
  - **D10 算法**：SMDP/options Dueling DQN——高层三动作 {Explore(点)/Pick(标签)/Deliver}，
    259 合法选项 + 动态掩码（dueling 均值只取合法动作）；执行器 (格,朝向) 1024 态 BFS
    （转向计步）+ 每步重规划 + 不可达中止；v1 commit-until-done；
  - **D7 定稿参数**：16 通道全图信念张量 (16,16,16) uint8（4 墙位 + 4 位姿 one-hot +
    2 机器 0/1/2 + 2 终点 + 2 carrying 广播 + 2 轨迹），替代局部 7×7 窗口；账本派生
    （探测账本 + 轨迹账本），墙值不存储、每步纯派生；
  - **D8 奖励**：−1/步 + 首次识别 +100/台 + 终点发现 +100/个 + 送达 +300/台；碰撞/超时无罚；
  - 辅助任务 = 补全未知墙位（GT 仅训练监督、λ 消融）；GRU 延后为 `use_gru` 开关
    （轨迹通道为轻量替代）；评估 = held-out 随机种子 + 官方图分套报告 + 全知 BFS 参照；
- `tools/bench_belief_derive.py`：信念派生性能基准（循环版 107.7 µs/次，训练全程外推 ~10 s）；
- `tools/decrypt_pdfs.py` / `pdf_libcheck.py`：论文 PDF 解锁副本（`reference/unlocked/`）。

### 契约语义确认（5 项全部经用户拍板，修订随 M2.6 冻结落地）

1. **终点公开性**：终点 (8,8)/(7,7) 固定但对机器鼠封闭、须自主发现 → contract §3 新增
   `DestinationFeed` 终点发现事件（图像下发，算法侧仅消费像素）；
2. **派送触发**：与显式 Pick 不冲突——Pick 生效 carrying → 进终点格自动派送（Arrived）；
   contract §3 补携带前置 + 新增 `Pickup` 事件（carrying 的合法信息通道）；
3. **单机负载**：一次只能背一台（Pick 掩码需空手）；
4. **ImageFeed 载荷可见性**：算法侧合法字段仅 `image`，身份走分类器（M2.5 接口，GT 直通占位）；
5. **碰撞/超时**：均无额外惩罚。

### 文件修改

- `docs/contract_v1.md`：§3 事件语义确认 + `Pickup`/`DestinationFeed` 登记；§4/§5/§6 三章
  按定稿填写；状态 草案 → 定稿候选（v1.0-rc，待新事件实现 + 单测同步后整体冻结）；
- `PLAN.md`：D6/D7/D8/D10 登记表更新、阶段三开工前置 #2 勾销、M2.6 状态更新；
- `docs/rl_algorithm_design.md`：重构为最终算法规格（v1.0 定稿）——训练课程改为**三臂消融**
  （A 搜索预训练→微调 / B 熵预训练→微调 / C 从头直训），切换机制与风险对策入 §6.2；
  移除契约簿记与决策沿革章节（该信息归 contract_v1.md / CHANGELOG 所有）。

## 2026-11-19 — 阶段二 M2.1–M2.5 交付 + M2.6 契约草案

### 新增

- `bamer_mouse/` 包（对阶段一验收代码零破坏；不 import `bamer_env`，墙体查询以
  `wall_query` 回调注入，R1 单向依赖 + MATLAB 可移植）：
  - `kinematics.py`（M2.1）：`Pose` / D5 原语 / `try_move` 碰撞判定 / `interpolate`
    动画插值；规范位姿 90° 整倍数方向向量精确快照（消除 IEEE 残差漂移）；
  - `lidar.py`（M2.2）：`LidarSpec`（D4 默认 360°/180 束/2.0 格/噪声关）+ DDA
    raycast（Amanatides & Woo，角点 tie 先竖直后水平）+ `Lidar.scan` 全扫描；
    独立随机流，可由环境种子派生重置；解析解对拍/开口穿透/噪声复现/性能单测
    （官方迷宫 180 束 0.31 ms/帧）；
  - `pointcloud.py`（M2.3）：极坐标→机体系→世界系齐次变换链；局部占据栅格
    `occupancy_grid`（D7-B，**仅凭量测数据复算，无墙查询**——与墙查询参考实现
    在官方迷宫 527 条回波束上逐格等价，R1 自洽实证）；呈现层（极坐标视图/点云
    残影衰减/ego 车头朝上旋转）；
  - `dataio.py`（M2.4）：`EpisodeRecord`（meta 含迷宫 sha256/种子/场景配置/雷达
    参数）+ npz+json 存取，往返一致性单测；目录 `data/episodes/`；
  - `camera.py`（M2.5）：`Classifier` 协议 + `GTPassthroughClassifier` GT 直通
    占位（D9 待定）+ `CameraUnit` 识别流水线（只消费像素，GT 仅评估口径）；
- `tests/test_kinematics.py / test_lidar.py / test_pointcloud.py / test_dataio.py /
  test_camera.py / test_env_sensor.py`（6 个新测试文件，42 项断言组全部通过）；
- `tools/demo_lidar_pointcloud.py` → `docs/lidar_wall_front.png`（墙前贴墙）/
  `lidar_opening.png`（开口穿透）/ `lidar_corridor.png`（走廊纵深）/ 
  `lidar_ghost_trail.png`（点云残影）/ `_demo_pointcloud.gif`（带实时点云完整任务
  录像 578 步：探索→双识别→双派送→TaskComplete）；4 张验收图目检通过；
- `docs/contract_v1.md`（M2.6 草案）：物理/几何/传感器/事件/数据格式章节已整理；
  观测（D7 定稿参数）/动作（D6）/奖励（D8）三章【待用户填写】；
- `bamer_mouse/README.md`。

### 既有文件修改（均为加法式，默认行为与阶段一验收版本逐位一致）

- `bamer_env/env.py`：新增可选构造参数 `lidar=LidarSpec(...)`（默认 None=不接雷达，
  info 键集不变）；接通时每 step 末尾 + reset 第 0 帧在规范位姿采全扫描，info 附加
  `ranges`/`angles`/`points`（束对齐 NaN 填充）；雷达流由 reset 种子派生（含噪可复现）。
- `bamer_env/viewer.py`：`_frame_from_env` 填充 `Frame.points=info.get("points")`
  （M1.3 预留挂点正式启用）。
- `PLAN.md`：M2.1–M2.5 状态、阶段二验收 4/5 勾选、D4 备注更新。

### 需用户拍板（登记于 contract_v1.md 与 AGENT.md §5）

1. **ImageFeed 载荷可见性（R1 相关）**：payload 的 `machine`/`label` 算法侧可见
   等于免识别获知身份；建议契约限定算法侧合法字段为 `image`（像素）。
2. `info["points"]` 束对齐 NaN 填充 vs 无回波丢弃（PLAN 原注"默认丢弃"，实现选
   对齐——利于 RL 定长与存档，展示层自行过滤）。
3. M2.6 三章待填：观测窗参/对齐/堆叠（D7 定稿）、决策级动作空间（D6）、奖励（D8）。

### 备注

- `tools/demo_observation_samples.py` 的 D7 复核已完成：正式实现占据栅格与样例式
  墙查询栅格在 4 位姿逐位一致（`tests/test_pointcloud.py::test_occupancy_measurement_vs_wallquery`）。
- 探索录像策略为纯传感器启发式（束角可通行判据 + 左手定则 + 新格优先），仅作展示，
  不是算法基线；正式策略由阶段三 RL 产出。

## 2026-11-19 — D7 决策：观测表示选定方案 B

- 用户拍板：算法侧观测表示 = **局部占据栅格（方案 B）**（样例见 `docs/d7_samples/`）；
- 实现路径：M2.2 雷达 → M2.3 点云/栅格化 → M2.6 契约冻结时定稿窗参/维度/多帧堆叠；
- 原始点云（A）与极坐标视图（C）保留为展示层渲染选项，不进入算法侧契约。

## 2026-11-19 — M1.1–M1.3 交付 + D7 样例

### 新增

- `bamer_env/interaction.py`（M1.1）：事件类型 `Reported / ImageFeed / Arrived /
  Collision / TaskComplete / Timeout`；机器鼠每步实时上报格坐标；进入状态机格
  下发人物图像（事件 + `docs/_pptx_media` 资产引用）；识别/派送簿记
  （`InteractionManager.identified / delivered / task_complete`）。
- `bamer_env/env.py`（M1.2）：`BamerEnv(gymnasium.Env)`；占位动作 `Discrete(4)`
  = D5 原语 {F, TL, TR, TB}；占位观测 `Dict(cell, heading, identified, delivered)`；
  `randomize=True` 随机迷宫 + 随机布点（种子全走 `reset(seed=)` 的 np_random 流）；
  `reset(options={"maze"/"state_machines"/"destinations"})` 为 R2 代码冻结注入接口；
  `terminated` = 两项派送达成，`truncated` = `max_steps` 超时；每动作恰计 1 步。
- `bamer_env/viewer.py` + `tools/demo_exploration.py`（M1.3）：逐帧动画（墙线 +
  鼠位姿 + 轨迹尾迹 + 步数 HUD + 识别图像面板 + `Frame.points` 点云挂点）；
  默认场景 1500 步完整探索录像 → `docs/_demo_exploration.gif`（点云未接通先无点云）。
- `tools/demo_observation_samples.py` → `docs/d7_samples/`（D7）：三方案样例
  （原始点云 / 局部占据栅格 / 降采样极坐标向量，2 位姿 × 3 方案）。
- `tests/test_interaction.py`、`tests/test_env.py`（含 100 种子 validate 验收）。

### 占位语义（均待用户确认，随 M2.6 契约冻结定稿）

| 项 | 当前占位 | 归属决策 |
| --- | --- | --- |
| 决策级动作空间 | D5 物理原语 F/TL/TR/TB 直通 | D6（用户） |
| 观测表示 | 格子级 Dict（cell/heading/identified/delivered） | D7（用户，样例见 `docs/d7_samples/`） |
| 奖励 | -1.0/步纯时间惩罚 | D8（用户） |
| 派送触发 | 进入已识别状态机映射的终点格 → 自动完成派送；识别前经过终点不派送 | 用户（M2.6 确认） |
| 碰撞语义 | 位置不变 + 计 1 步 + Collision 事件，不终止 | 用户（M2.6 确认） |

### 备注

- `tools/demo_observation_samples.py` 内嵌样例专用临时 raycast（未对拍），
  M2.2 正式 `LidarSpec` + raycast 落地后复核 D7 样例。
- 既有已验收代码（`maze.py` / `scene.py` / `render.py` / 迷宫 JSON）未改动，
  仅 `bamer_env/__init__.py` 追加导出。
