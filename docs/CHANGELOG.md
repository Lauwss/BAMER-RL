# CHANGELOG

> 契约（M2.6）冻结后的任何语义变更须先经用户确认，再记入本文件并同步单测。

## 2026-11-21 — λ_aux=0 消融（seed 0，shaping+γ=0.999+Graph 基线上单变量）

用户提出"辅助损失似乎没什么用"。设计：shaping+γ=0.999+Graph 基线上仅去
λ_aux（0.5→0），A/C 两臂各 160k options，对照同基线 λ_aux=0.5：

| | λ_aux=0 | λ_aux=0.5 |
|---|---|---|
| A 训练期送达（stage1） | **5** | **11** |
| A stage1 探索率均值 / 峰值 | 18.4% / 40.6% | 19.1% / 42.6% |
| A 8-seed 贪心 探索/墙知识 | 5.76% / 12.77% | 7.13% / 15.65% |
| C 训练期送达 | **7** | **9** |
| C 探索率均值 / 峰值 | 16.8% / 39.8% | 16.8% / 40.2% |
| C 8-seed 贪心 探索/墙知识 | 8.11% / 16.70% | 9.81% / 20.53% |

**结论（补同栈对照后更新）**：同栈（Graph+4线程+确定模式）下 λ_aux=0.5 vs 0
全面占优——A 送达 **16 vs 5**（3.2×）、探索率 20.5% vs 18.4%、8-seed 探索
10.16% vs 5.76%；C 送达 9 vs 7、8-seed 探索 8.89% vs 8.11%。辅助损失有正贡献，
**无去除动机**；λ=0.5 与 0 的差距远大于同配置换栈的波动（A 同配置 λ=0.5 新旧栈
11 vs 16 次送达 → 单 seed 方差大，λ 结论需多种子确认）。§4.5 消融计划
{0, 0.1, 0.5, 1.0} × 多种子仍待执行。吞吐：noaux 133 vs aux05 122-126 options/s
（2 并行同栈）——辅助头计算开销 ~5-8%，与预估一致。附带修复：CUDA Graph 学习体
在 λ_aux=0 下构建崩溃（辅助头 grad=None → zero_ 失败），改为只纳入有梯度缓冲的
参数 + 构建失败锁存。

## 2026-11-21 — CUDA Graph 固化 act/学习体 + cuDNN 确定算法（agent.py，吞吐 ~1.9×，复现升级为参数级）

### 改动（`algo/agent.py`，`TrainConfig.cuda_graph: bool = False` 门控默认关）

1. **act 图**：捕获 [H2D→前向→dueling→统计→D2H(pinned)]。动态合法集 gather
   等价改写为固定形状——argmax 全向量（非法=−inf，结果相同）+ 掩码均值
   `where(m,q,0).sum()/m.sum()`（仅监控量，~1ulp 序差）；
2. **学习图**：捕获 [H2D→3 前向→损失→反传→损失 D2H]。**clip_grad_norm_ 与
   Adam.step 留在图外**（fused Adam eager 读静态梯度缓冲）——Adam 状态首次
   创建、step 计数、stage1_lr 变更全部保持原语义，图无需重捕获；
3. **warmup 无副作用**：侧流 3 次预热只跑前向+反向（无 optimizer.step），参数
   与 Adam 状态零污染 → 图开关轨迹与参数逐位一致；
4. **梯度缓冲静态化**：图路径 `zero_grad(set_to_none=False)` + 图内 `grad.zero_()`
   起手（初版漏 `backward()` 的 bug 由等价性测试抓出并修复）；
5. **cuDNN 确定算法**（CUDA 时强制 `cudnn.deterministic=True, benchmark=False`）：
   默认 False 时 conv 反传选非确定核，同 seed 两次运行参数有 ~1e-8 漂移（轨迹
   靠 argmax 鲁棒性才一致）；开启后复现从"轨迹级"升为"参数级"，且是 Graph 与
   eager 逐位一致的前提；
6. 构建失败 try/except 自动回退 eager；CPU 设备不启用。

### 等价性验证（tests/test_trainer_smoke.py::test_cuda_graph_equivalence）

同 seed 图开/关完整训练：option 轨迹、episode 核心列、**checkpoint 参数与优化器
状态张量逐位一致**；14/14 测试过。

### 实测（干净 GPU，确定模式，2500/500-option 差分法）

| 路径 | 边际吞吐 | 单臂 160k |
|---|---|---|
| eager + 确定模式 | 13.6 ms/option（74/s） | ~36 分钟 |
| **CUDA Graph + 确定模式** | **5.10 ms/option（196/s）** | **~16 分钟** |
| （改前最快：非确定 eager + pinned） | 9.61 ms/option（104/s） | ~26 分钟 |

Graph 相对确定模式 eager 2.7×、相对改前最快路径 ~1.9×。剩余热点为环境侧
Python（雷达/账本/状态机），进一步压榨需 numba 化或 CUDA Graph 扩展到环境循环
（形状动态，难度高），性价比低。

## 2026-11-21 — 学习/决策路径 H2D 提速：pinned 常驻暂存 + 同步合并（agent.py，吞吐 8×）

### 热点定位（cProfile + 分相微基准，500-option 短跑）

`torch.as_tensor(numpy, device=cuda)` 逐字段上传占 46%（学习步 10 字段 10.7ms/
次 + act 3 次，pageable 内存每次隐式中转/同步 1–2ms）；叠加每 option ~13 个
GPU 同步点（10 H2D + q 统计 2×.item() + 损失 tolist），同步排队构成第一瓶颈
（GPU 纯算仅 ~6%）。注意：剖析期间本机 GPU 同时被外部图形进程占用，绝对值
被放大，但前后同条件对比有效。

### 改动（`algo/agent.py`，数值与语义不变）

1. **pinned 常驻暂存**：`__init__` 预分配 batch 级 pinned 缓冲（s/s2/gt×2/r/a/
   tau/terminal/mask×2）与 act 级暂存（按 in_channels）；学习步先 CPU memcpy
   写入 pinned，再每字段一次阻塞 pinned H2D（免 pageable 中转）；
2. **act 合法集 GPU 端索引**：mask 上传后 `legal_idx[gpu_mask]` 在 GPU 上取
   合法集，免第三次小 H2D；
3. **单次同步**：act 的 q_mean/q_max/argmax 合并为一次 `torch.stack().tolist()`
  （原 3 次 `.item()` 各自全管线同步）；学习步损失上报合并保持不变；
4. **Adam fused=True**（仅 CUDA）：优化器步合并进单 kernel。

### 实测（两套口径）

- 学习步 H2D 微基准：10.68 → **0.46 ms**（23×）；
- **干净 GPU（真实训练条件）**：边际吞吐 53–65 → **104 options/s**
  （15.4–18.8 → **9.61 ms/option**，5000-option 差分法精确测量）——对实际训练
  **增益 ~1.6–2.0×**，单臂 160k options 约 50 → 26 分钟；
- （参考）GPU 争抢条件下前后对比 70.3 → 8.8 ms/option（8×）——争抢放大每个
  同步点的排队代价，该数字不代表训练条件，仅说明同步优化在争抢下收益超线性；
- 数值一致性：同种子复现测试逐行一致通过（14/14），shaping 测试 episode 奖励
  与改前逐位相同（fused Adam 未翻转任何决策）；算法语义零变化。

### 备注

干净 GPU 剖析（优化后，9.61 ms/option）：学习步（3 前向+反向+Adam）~4ms、
环境步进（env.step python + 雷达 0.33 + 账本 0.29 + has_wall 0.11 ms/步）~3ms、
act ~1.5ms、其余 ~0.5ms。剩余优化空间 ~30%：CUDA Graph 固化学习步/act（形状
静态可捕获，估 −1.5~2.5ms）、环境 DDA numba/Cython 化（估 −1~1.5ms）、env.step
python 微优化——成本递增，当前性价比不高。**并行线程限额**：torch 默认按全核开
intra-op 线程，3 进程并行超订阅；实测每进程限 3-4 线程聚合吞吐 181 → 226
options/s（+25%），已支持环境变量 `BAMER_TORCH_THREADS`（tools/train.py，不设
则保持默认）。正式网格仍建议服务器 run 级并行。

## 2026-11-21 — A/C 探索塌缩根因修复：势能拉势 shaping（默认关，转正待拍板）

完整分析过程见 **docs/ac_collapse_analysis.md**。要点：

### 根因（2026-11-20 三臂首训取证）

−1/步 + 未采样里程碑构成反探索梯度 → Q 收敛到"最短 option"偏好 → 贪心策略
退化为起点旁 3 格死循环（rollout 探针实证：唯一 option 数=3、重访率 1.0）→
里程碑永不采样（bonus>0 的 episode 仅 11–19%，B 为 62.7%）→ 因果闭环无法自愈。

### 改动（`algo/trainer.py`，配置门控默认关）

1. `TrainConfig` 新键：`shape_potential: bool = False`、`shape_scale: float = 2000.0`
   （validate 拒绝非法 scale）；
2. `Trainer._shape_reward`：势能拉势 F = γ^τ·Φ(s') − Φ(s)，Φ = 已知墙槽位比例
   （Ng et al. 1999 / Grzes 2017 SMDP 多步动作形式，与 TD target 的 γ^τ 同构 →
   最优策略不变）；terminated 时 Φ(s')=0（定理严格条件）；零增益 option 恒负
   （信息时间代价），正增益只能来自真实新增墙位知识；
3. 主循环在 `reward_fn` 之后按门控叠加；默认关路径与冻结契约逐位一致。

### 测试

`test_potential_shaping`（公式逐项/terminated 分支/门控校验/端到端确定性）；
全套测试通过；同种子复现不变。

### 实测（seed 0 重跑 160k options）

训练期送达 0→**14 次（A）/6 次（C）**（B 参照 6）；stage1 探索率坍缩→爬升
（A 11.4%→18.0%，C 7.2%→14.1%）；贪心死循环打破（唯一 option 3→8–13，
8-seed 贪心探索 1.07%→9.81%、1.32%→5.71%）。A_shaped 训练期已超 B 臂。

### 待拍板

D8 奖励新增可选拉势项是否随正式实验转正（默认关保持冻结行为）。

## 2026-11-20 — 学习步 host 开销优化（用户拍板"先做①"）

### 改动（`algo/agent.py`，学习语义不变）

1. **γ 标量张量缓存**：`__init__` 预建 `self._gamma_t`，学习步不再逐次
   `torch.tensor(gamma, device)`（省一次 H2D + 对象构造）；
2. **`gt_directional_walls` 设备无关化**：接受 numpy 或 torch 张量，纯 0/1 重排
   改 torch 切片赋值（数值逐位一致）——调用侧先把 GT 传上 GPU 再换算，免去
   CPU 中转 + (B,4,n,n) float 的 H2D 传输；
3. **辅助 BCE 无同步化**：`unknown.any()` 中途 D2H 同步 → 全 GPU 掩码均值
   `sum(bce·mask)/cnt.clamp(min=1)`（数学等价：子集均值 vs 掩码均值；cnt=0 时
   bce=0 与"跳过"等值，常数 0 无梯度。浮点求和序差 ~1e-7，traj 可能微移，
   契约无路径锁定）；
4. **单次 D2H 损失上报**：td/aux/total 三损失 `torch.stack().tolist()` 一次
   同步（原 td/total 各一次 `.float()`）；
5. **`compute_aux=self.lambda_aux>0` 守卫**：λ_aux=0 消融臂不再白算辅助头。

### 实测（RTX 4060 / 正式配置量级）

- 学习步：10.84 → **7.11 ms/call（−34%）**（先插桩定位：GPU 纯算 2.7ms，
  host 开销 ~7ms——逐损失同步/逐步建 gamma 张量/GT CPU 中转是主犯）
- 端到端 episode：3.78 → **3.16 s（−16%）**；**相对本日起点 4.25s 累计 −26%**
- 14/14 测试全过（td_loss 与改前逐位一致；aux_loss ~1e-7 浮点序差）
- GPU 上限探针（顺带登记）：batch 64→1024 学习步 2.7→32ms——卡未吃饱，
  3090 级换卡预计端到端 <5%，不值得；剩余瓶颈=单线程 Python 数据供给。

## 2026-11-20 — 雷达/信念账本向量化提速（性能优化，语义逐位一致）

### 改动（束间并行 DDA，结果与标量版逐位一致）

1. `bamer_env/maze.py`：构造时缓存 numpy 墙数组（墙线不可变）；新增
   `has_wall_batch(xs, ys, dirs)`（语义逐条镜像 `has_wall`，N/E 越界=外边框墙）。
2. `bamer_mouse/lidar.py`：新增 `_raycast_parallel`（束间锁步 DDA，数学全向量化；
   三个逐位一致保证：tie 同式 tx<=ty、每束查询序列不变、t>max_range 严格大于）；
   `Lidar.scan` 新增可选 `wall_query_batch`（缺省退化标量回调，结果一致）。
   标量 `raycast` 保留（对拍参照/单测用）。
3. `algo/belief.py`：`update_scan` 重写为锁步 DDA——每轮每束恰跨一条边界，穿越
   侧直接给出规范槽位（E/N=外侧格带护栏、W/S=内侧格，与 `_slot_between` 逐条
   对齐）；标记收集后批量写入，`_mark` 单调+wall 恒胜 → 先 free 后 wall 的批量
   写入与逐束交错写入终态逐位一致。顺带消除标量版噪声越界场景的负索引回绕
   隐患（穿出网格的束即刻终止；外边框恒有墙，正常束不受影响）。不再 import
   walk_cells（pointcloud.occupancy_grid 仍用）。
4. `bamer_env/env.py`：`_scan_now` 传入 `has_wall_batch`。

### 实测（RTX 4060 Laptop / CUDA / 正式配置量级 180 束）

- 雷达单帧：0.799 → **0.332 ms**（**2.4×**，test_lidar 性能项）
- `update_scan`/帧：~2.0 → <0.4 ms（**≥5×**，剖析榜除名）
- Python 函数调用/episode：386k → **88.6k**（**−77%**）
- 端到端 episode（1000 步跑满）：均值 4.25 → **3.78 s**（−11%——瓶颈已转移到
  学习步 GPU 小批次传输 ~45% 与 BFS 执行器纯 Python 规划；进一步提速方向是
  executor BFS 向量化/网络 CPU 化，本轮未做）

### BFS 前沿扩展：尝试后回退（负结果登记）

executor BFS 的 numpy 前沿扩展版已实现并用 1795 例随机场景验证**步数零不一致**
（4.8% 等长不同路，tie-breaking 差异，均最优），但微基准 **0.85ms vs 标量
0.56ms（0.7×，更慢）**——标量版提前终止只访问路径半径内 ~150 态，逐层 numpy
的固定开销（fancy indexing / unique / 分配）在 1024 态规模摊不平。**已回退**，
`Executor.plan` 维持标量 `bfs_path`；教训：向量化不是普适提速，小状态空间 +
提前终止的场景 Python 顺序循环反而占优。

### 回归

14/14 测试全过，含逐位一致类：lidar 解析解/噪声逐位复现、D7 复核、账本手算、
同种子训练复现逐行一致。修复过程中账本手算测试曾抓出 S 穿越槽位误镜像
（外侧格误用 → 负索引回绕），已修正为与 `_slot_between` 逐条一致。

## 2026-11-20 — 探索质量诊断指标 + Mean Q-value 监控（用户提出补充）

### 新增监控指标（episodes.csv 新列 + swanlab 曲线）

1. **重复探索率** `repeat_ratio` → `explore/repeat_ratio`：
   ρ_repeat = 落入已探索格的位移步数 / 实际位移步数。位移 = `OptionResult.path`
   相邻格变化（碰撞/转向天然不计）；已访集取**本 option 开始前**的信念 visited
   （BFS 路径无环，option 内自环不存在）。纯函数 `trainer._repeat_moves` 有单测。
2. **单位探索成本的信息增益** `info_gain_per_move` → `explore/info_gain_per_move`：
   η_info = (episode 末已知墙槽位 − 初始已知) / 实际位移步数（槽位 = 2n²）。
3. **Mean/Max Q-value** `q_mean`/`q_max` → `learn/q_mean`/`learn/q_max`：
   `agent.act` 贪心分支零额外成本顺带统计（合法动作 Q 均值 / 全局 max），
   ε-random 步无前向 → 不计入。q_max 用于观察价值异常增长。

### 对照用户指标表

Episode Reward ✓（train/reward）· Success Rate ✓（mini eval eval/success_rate +
train/success）· Episode Length ✓（train/steps）· TD Loss ✓（learn/td_loss）·
Mean Q-value ✓（本次新增 learn/q_mean，附带 q_max）。

### 文件

`algo/agent.py`（act 暴露 last_q_stats）、`algo/trainer.py`（诊断累计 + 行字段 +
`_repeat_moves`）、`algo/monitor.py`（4 条新曲线映射）、
`tests/test_trainer_smoke.py`（+`test_repeat_moves_unit`；CSV schema 断言扩充）。
14/14 测试过（同种子复现不变——诊断量只读）。

## 2026-11-20 — SwanLab 实验管理 + 训练中 mini 评估（用户拍板：云端 + 加 mini 套件）

### 新增

1. **`algo/monitor.py`**：`NullMonitor`（no-op 替身）+ `SwanLabMonitor` +
   `build_monitor(cfg)`。可选依赖——`cfg.swanlab=False`（默认）或未安装 swanlab
   自动降级，冒烟/回归零影响。step 轴 = 全局 option 计数（跨 stage 连续）；
   指标分组 `train/ explore/ learn/ stage/ eval/ summary/`；init 携带全量
   TrainConfig 供跨实验对比。云端模式需先 `swanlab login`。
2. **TrainConfig 新键**（默认值均=关闭，不破坏复现）：
   `swanlab: false` / `swanlab_project: BAMER-RL` / `swanlab_mode: cloud|local|offline`
   （validate 拒绝非法值）；`eval_every_options: 0`（0=关）+ `eval_mini_seeds: 3`。
3. **训练中 mini 评估 hook**：每 `eval_every_options` 跑一次 greedy held-out
   mini 套件（恒用 stage1 拓扑、种子由 master seed 派生固定可比）——`evaluate()`
   纯贪心只读，不写 buffer、不占训练种子流（同种子复现测试仍逐行一致）。
   正式三臂配置 `eval_every_options: 10000`，smoke 保持 0。
4. **episodes.csv 补 loss 分项列**：新增 `td_loss` / `aux_loss`（episode 内学习步
   均值；原只有 total 的 `mean_loss`）——补 monitor 前的数据缺口。
5. `summary.json` 新增 `options_per_sec` 吞吐字段。

### 文件

- 新增 `algo/monitor.py`；修改 `algo/trainer.py`（4 个 hook + `_run_mini_eval` +
  loss 分项）、`algo/configs/*.yaml`（5 键）、`requirements.txt`（swanlab==0.10.6，
  可选）、`tests/test_trainer_smoke.py`（+2 测试：monitor 关闭路径/校验、mini
  hook 端到端 + CSV schema）。

## 2026-11-20 — 训练迷宫拓扑升级：braided 赛图匹配（用户确认）

### 背景（实测，非破坏契约——全部新参数有默认值 = 旧行为）

- 验收图 `default_maze_ppt.json` 是 **braided 有环迷宫**：可达子图 215/256 格、
  **37 环**、非起点死胡同仅 8 个（环/格 0.172）。
- 此前训练 `Maze.random_maze(seed)` 默认 `braiding=0.0` → **全用 perfect maze
  （0 环、~26 死胡同）**，与验收图拓扑失配。
- 生成器死胡同开洞上限只有 ~0.10 环/格（braiding=1.0 → 27 环），追不上 0.172——
  官方赛图的环来自人为平行长廊，不经死胡同。

### 改动（4 处）

1. `bamer_env/maze.py`：`random_maze` 新增 `target_loops` 参数——先按 `braiding`
   开死胡同，再随机敲内部墙补环到目标（连通迷宫中敲墙恒 +1 环、不破坏连通性）；
   新增 `_open_interior_edges` 辅助。默认 None = 旧行为。
2. `bamer_env/env.py`：`BamerEnv` 新增 `maze_braiding`（默认 0.0）与
   `maze_loops_range`（默认 None，非法区间构造时拒绝）构造参数，randomize 分支
   用 `np_random` 采样目标环数（同 reset 种子逐墙可复现）。
3. `algo/trainer.py` + `algo/configs/*.yaml`：`TrainConfig` 新增
   `maze_braiding_stage0/stage1`、`maze_loops_stage0/stage1`（默认 0.0/None =
   旧行为）；`_build_env` 按 stage 取参（臂 C 的 "full" 归入 stage1 档）。三臂配置
   统一：**stage0 = braiding 0.0 + 补环 U[0,30]（练死胡同回溯与系统覆盖），
   stage1 = braiding 0.8 + 补环 U[25,50]（锚定验收图 37 环/215 格）**；臂 C/smoke
   用 stage1 值。
4. `tools/train.py`：`_topology_kwargs(cfg)` 使 held-out 评估工厂与场景复现 env
   使用与 stage1 相同拓扑（两处必须一致，否则参照场景失配）。

### 测试

- `tests/test_env.py` 新增 `test_random_maze_target_loops`（精确补环/连通/复现/
  默认旧行为）与 `test_env_randomize_topology`（区间生效且可复现/固定场景不受
  影响/非法区间拒绝）；全套 14/14 文件通过（含同种子训练复现、官方图全知参照
  仍 72 步）。

## 2026-11-20 — **契约 v1.0 冻结**（M2.6 门禁关闭）+ D9 拍板

### 用户决议（4 项，同日拍板）

1. **契约 v1.0 整体冻结**：contract_v1.md 版本号升 v1.0，全部七章 ✅ 已冻结；
   此后任何语义变更走 §8 流程（用户确认 → CHANGELOG → 单测 → 版本递增）。
2. **D9 = 维持 GT 直通直到验收**：真实分类模型（数据集/训练）不在当前范围，
   `CameraUnit`/`GTAssetClassifier` 占位贯穿管线；验收前如有需要另行立项。
3. **ImageFeed 载荷可见性（R1）写入冻结契约 §3**：算法侧合法字段仅 `image`
   （像素）+ 事件步号，身份走分类器输出（`CameraUnit`/`algo.belief.GTAssetClassifier`
   已按此实现）。
4. **info["points"] 无回波语义 = 束对齐 NaN 填充**（写入冻结契约 §2）：
   利于 RL 定长/存档，展示层自行过滤 NaN。

### 文件修改

- `docs/contract_v1.md`：标题/状态 → v1.0 已冻结；七章状态表更新；§1 碰撞语义
  行、§2 无回流行、§3 ImageFeed 可见性行同步定稿表述；
- `PLAN.md`：M2.6 → ✅ 已冻结（阶段二验收 5/5 全勾）；D9 登记表 → 已定；
- `AGENT.md` §5：待拍板清单清空（无待办；默认项登记：雷达降束、臂 B 熵奖励
  权重 1.0、随机迷宫 braiding=0）。

## 2026-11-20 — 规则 A 用户确认 + 全知参照语义修复与升级

### 语义确认

- **Pickup 首达语义（规则 A）经用户拍板确认**：进入状态机格且空手且未送达即拾取，
  首达 = ImageFeed+Pickup 同步。已知代价（树形迷宫中机器挡在唯一路径上时强制
  拾取序可能多一趟折返）由全知参照如实计入；contract_v1.md §3 登记为已确认语义，
  AGENT.md §5 清单勾销。替代方案（显式 PICK 原语）否决。

### 修复（全知参照两处建模 bug，数值修正）

1. **F 边漏朝向约束**：参照 BFS 的移动边遍历 4 个方向且不改朝向——等于把 F
   建模成"任意方向平移一格"，退化为无转向代价的格子 BFS（executor 的 bfs_path
   无此 bug，其 F 边只取当前朝向）。修复后移动边仅沿 HEADINGS[h]，转向由
   TL/TR/TB 旋转边承担（R4 计步）。**官方图参照值由错误的 50 修正为 72**
   （(格,朝向) 全语义最优；旧值不可作为参照上限）。
2. **强制事件语义缺失**：参照原为"自由任务序 staged BFS"（2 条序取最小），未
   建模规则 A 的强制拾取/送达。升级为 (格,朝向,携带,送达) 状态自动机 BFS
   （8192 态），与 env 交互语义逐条一致。程序化拓扑搜索（45 个走廊+支路案例）
   实证：7 例强制序代价 > 自由序（gap=1 步，约 15%），其余全部相等——规则 A
   的全局影响真实存在、有界且已在参照中如实计入，评估差距严格公平。

### 新增

- `tests/test_eval_reference.py`：参照语义三测试（无冲突拓扑 forced==free /
  挡路拓扑 forced=28 > free=27（搜索锁定实例）/ 官方图参照=72）。

## 2026-11-20 — 阶段三开工：契约新事件实现 + algo/ 全模块构建（§8 冒烟 1–5 通过）

### 新增

- **`bamer_env/interaction.py` 契约事件实现（contract §3 定稿语义落地）**：
  - `Pickup`：进入状态机所在格且空手且未送达 → carrying 生效（首达即识别+拾取
    同步发生；识别时手持他机仅识别、事后 Pick 回取——259 选项中 2 个 Pick 的
    真实语义；已送达机器不重复拾取；单机负载）；
  - `DestinationFeed`：进入终点所在格下发终点图像（first 标记首次发现；终点
    封闭须自主发现——算法侧唯一终点发现通道）；
  - 派送前置从"已识别"改为"**已携带**"（2026-11-20 拍板）：进入已携带且映射的
    终点格 → Arrived 自动派送；`search_complete` 属性（课程臂 A 阶段 0 判据）；
  - 终点图像资产 `DESTINATION_ASSETS`（与 render.py 同源 image3/5）。
- **`bamer_env/env.py`**：构造参数 `curriculum="search"`（臂 A 阶段 0：terminated
  = 4 POI 定位，交互机制不变）；只读属性 `env.carrying`。
- **`algo/` 包（阶段三全模块，docs/rl_algorithm_design.md §8 定稿）**：
  - `belief.py`：探测账本（规范形式感知记录，walk_cells 量测复算、只增不减、
    有墙优先）+ 轨迹账本 + 16 通道信念张量派生（[C,x,y] 布局与 bench 一致；
    墙值不存储、每次全量重派生）；GTAssetClassifier 直通占位（只消费合法
    image 字段）；R1 自洽（不 import scene/Maze）；
  - `actions.py`：259 选项枚举（Explore 256 + Pick 2 + Deliver 1）+ 动态合法掩码
    （信念图可达集；Explore 排除当前格防 τ=0 死循环；已送达机器不可 Pick）；
  - `executor.py`：(格,朝向) 1024 态 BFS（转向计步 R4）+ 每步重规划 + 不可达
    中止；OptionResult 全量记录（SMDP 转移材料）；
  - `network.py`：FCQN 三分支（3 层卷积主干 32/64/64 + Explore score map +
    Pick/Deliver/Value 池化头）+ 辅助补墙头 (4,16,16) logits；dueling_q 动态
    掩码组合（均值只取合法动作）；use_gru 显式 NotImplementedError 预留；
  - `replay.py`：option 级 uint8 环形经验池（含双端合法掩码 + 辅助 GT 墙线）；
  - `agent.py`：ε-greedy + Double-DQN 学习步（γ^τ SMDP 折扣；terminated 不
    自举、truncated 自举）+ 辅助 BCE（unknown 槽位）+ 梯度裁剪 + checkpoint；
  - `trainer.py`：训练循环 / 种子管理（master seed → 环境+agent 独立流）/
    checkpoint 恢复 / CSV 日志 / 三臂课程（A 搜索预训练 / B 熵预训练 / C 基线，
    阶段切换 = buffer 清空 + target 重置 + ε 重启 0.05）；D8 奖励实现于奖励
    包装层（−1/步 + 首次识别/终点发现 +100 + 送达 +300；臂 A 阶段 0 去送达、
    臂 B 阶段 0 = 信息增益精确计数）；
  - `eval.py`：固定 seed 套件评估（贪心）+ **全知 BFS 参照**（(格,朝向,任务
    进度) 状态空间 BFS，两条合法任务序取最小）+ 指标（成功率/搜寻/派送/总步数/
    探索区域率/墙知识率/参照差距）；
  - `configs/`：smoke.yaml（本地冒烟）+ armA/armB/armC.yaml（服务器正式配置）。
- **`tools/train.py`**：训练/评估统一入口（--config/--resume/--eval/--eval-seeds），
  正式训练交私人服务器执行（BAMER 环境）。
- **tests/ 新增 5 个（§8 冒烟顺序）**：`test_belief.py`（账本手算对拍 + R1 自检
  GT-free 输入流）/ `test_executor.py`（解析解对拍含转向计步、重规划/中止、
  Pick/Deliver 全流程）/ `test_network.py`（掩码语义、手工场景 argmax 合规、
  200 组随机掩码模糊）/ `test_agent_replay.py`（uint8 往返、ε-greedy 100% 合法、
  warmup 边界、γ^τ 对拍、阶段切换、GT 派生）/ `test_trainer_smoke.py`（端到端
  分钟级冒烟、**同种子复现逐行一致**、评估套件 + 全知参照 50 步）。

### 修复（实现期测试抓出的真 bug）

- `actions.legal_mask`：已送达机器仍可 Pick（补 machine_delivered 排除）；
- `executor.run`：到达目标格的 break 早于 terminated 标志记录 → "送达即终止"
  的 terminated 丢失（先行记录后 break）；
- `eval.run_eval_episode`：一次不可达中止即终止整条评估 episode → 改为连续
  中止 8 次判死锁（失败信号应交还高层）。

### 语义登记（随本次实现一并落地，请用户随 M2.6 冻结确认）

1. **Pickup 首达语义**：env 层"进入状态机格且空手且未送达"即拾取（首达 =
   ImageFeed+Pickup 同步）。推导依据：执行器只产出 D5 原语（§3.2）env 无法
   感知 option 意图；若首达不拾取则一切首次拾取需二次到访，与 R3 步数最优
   矛盾；"识别时手持他机 → Pick 回取"恰好保持 Pick 选项的真实语义。
2. **DestinationFeed 正交性**：进入终点格恒触发（first 标记首次），与 Arrived
   同一进入可并存；未携带（或携带非映射机器）不派送。
3. **评估循环死锁判据**：连续 8 次不可达中止 = episode 提前失败（工程保险丝）。

### 契约状态

- contract_v1.md §3 两个新事件 **已实现 + 单测同步** → v1.0-rc 达到可冻结状态，
  **整体冻结待用户最终确认**（冻结后版本号升 v1.0）。

### 冒烟验收（§8 顺序，全部通过）

1. belief：账本手算对拍 / 张量通道 / R1 自检 ✅
2. executor：BFS 解析解（含 TB 计 1 步）/ 重规划中止 / Pick-Deliver 全流程 ✅
3. network：dueling 掩码语义 / 手工场景 argmax 合规 / 随机掩码模糊 ✅
4. agent+replay：学习无 NaN / 合法动作 100% / γ^τ / 阶段切换 ✅
5. trainer+eval：端到端 1.7s 冒烟 / **同种子复现逐行一致** / 全知参照 50 步 ✅

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
