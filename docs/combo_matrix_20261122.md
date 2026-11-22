# 组合臂矩阵实验（2026-11-22）：rf 转正种子方差 + learn2 新栈复核

> 执行 3b9a913 的 5-run 矩阵：{dist+dilate+learn2}×seeds 0/1/2 + 基线 frontier×seeds 1/2。
> 训练 ~31-33 min/160k（combo 吞吐 83.8-85.1 options/s 与基线 81.2-84.5 持平——距离场
> 不构成瓶颈）；swanlab 已登记（run 名 = A-armA_frontier_{combo_s*,s*}-seed*）。
> **评测口径升级**：commit 备注允许"各自套件对照或统一套件复评"，本批直接采用
> **统一套件复评**——全部 6 个 checkpoint（含基线 s0）在 seed_base=0 的同一套
> **100-seed** held-out 套件上贪心评估，全矩阵单值可比，无跨套件噪声。

## 统一套件结果矩阵（held-out 100 seeds，seed_base=0）

| Run | 栈 | seed | 成功率 | POI 定位率 | 总步数 | 探索率 | 墙知识率 | 参照差距 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 基线 frontier_s0* | learn1 | 0 | 31.0% | 34.0% | 607.7 | 56.3% | 77.6% | 525.4 |
| 基线 frontier_s1 | learn1 | 1 | 41.0% | 44.0% | 662.2 | 65.3% | 86.2% | 572.3 |
| 基线 frontier_s2 | learn1 | 2 | 46.0% | 47.0% | 667.1 | 62.8% | 84.2% | 578.9 |
| 组合 combo_s0 | dist+dilate+learn2 | 0 | 53.0% | 54.0% | 679.0 | 67.2% | 86.7% | 581.1 |
| 组合 combo_s1 | dist+dilate+learn2 | 1 | **57.0%** | **58.0%** | 626.3 | 64.2% | 83.7% | 532.0 |
| 组合 combo_s2 | dist+dilate+learn2 | 2 | 41.0% | 43.0% | 625.5 | 64.0% | 84.1% | 536.0 |

\* 基线 s0 用上一批 `armA_frontier_lidar90` checkpoint（与原 frontier s0 逐位等效，
见 docs/ablation_followup_20261122.md 降束零信息损失节）。

## 判读矩阵逐条

### ① rf 转正训练种子方差

- 基线行 31/41/46%：**均值 39.3%，种子间极差 15pp**——frontier 基线本身的种子
  方差就很大，单 run（此前 s0 的 40%/50% 读数）不可作定论依据，本批实测坐实；
- 组合行 53/57/41%：均值 **50.3%**，配对差 s0 +22pp / s1 +16pp / **s2 −5pp**；
- 判读：**组合臂平均 +11pp，2/3 种子优于配对基线**，方向与 93434eb 的
  40→58%（p=0.016）一致；但 s2 反转 + n=3 种子样本量小（种子层 sd ~8pp），
  "rf 转正"证据成立但强度中等，建议以 5 种子矩阵或逐步配对检验收口；
- 组合行最低种子（41%）≥ 基线行均值（39.3%）——即最差组合种子也不输基线
  平均水平。

### ② learn2 新栈复核（combo_s0 vs dist_dilate s0 配对）

- combo_s0（learn2）= 53.0%（统一套件）；dist_dilate（learn1，你们侧 run）=
  58%（**其自建套件**，checkpoint 不在本机，无法统一套件复评）；
- 差 5pp ≈ 两套件各自 ±5pp 采样噪声范围内 → **learn2 在新栈上无降级证据**，
  与首批 frontier 栈上的 +25pp 方向一致；严格配对定论需 dist_dilate
  checkpoint 在统一套件复评（补跑或传 checkpoint 即可闭环）。

### ③ 组合臂多种子（自身稳健性）

- 53/57/41%：三种子全部 ≥41%、均值 50.3%、sd ~8.3pp——与基线行 sd（~7.6pp）
  相当，组合臂没有放大种子方差；41% 低点与 s2 训练曲线（swanlab）可对照看
  是否早期 ε/学习塌陷。

## 与 93434eb 数字的关系（重要校准）

你们侧文档口径：基线 s0=40%、组合 s0=58%（100-seed，自建套件）。本批统一
套件下基线 s0=31%、combo_s0=53%——同一 checkpoint 跨套件差 ~9pp 与 n=100
二项采样噪声（σ≈4.7pp，~1.9σ）相容；**跨文档引用数字时必须注明套件口径**，
后续定论建议统一走本批的 seed_base=0 / 100-seed 口径（脚本与套件派生已固化）。

## 注意事项

1. POI 定位率与成功率仍高度同步（差 1-3pp）：失败 episode 全部为 POI 未齐
   超时，派送段（BFS 支撑）未见失败；
2. 探索率：combo 三种子 64-67% 全面高于基线 s0（56%），与 dist 通道引导
   覆盖的机制假设一致；
3. 参照差距 combo_s1/s2（532/536）优于基线 s1/s2（572/579），s0 行反向
   （581 vs 525）——步数指标与成功率不同步，rf 转正以成功率为先（题目铁律
   R3 的核心是"搜寻+派送"达成的可靠性）；
4. 训练吞吐 combo ≈ 基线（距离场 BFS 在 option 边界一次性计算，非逐步开销），
   dist 通道无墙钟代价。

## 复现

- 训练：`python tools/train.py --config algo/configs/armA_frontier_{combo_s0,combo_s1,combo_s2,s1,s2}.yaml`；
- 统一套件评估：`BAMER_TORCH_THREADS=4 python /tmp/bamer_eval_ext.py <config> <ckpt> 100 0`
  （临时脚本随本批使用后删除；如需固化统一套件口径，可将 seed_base 参数
  并入 tools/train.py run_eval）；
- 评估日志：`algo/runs/<name>/eval_heldout100.log`。
