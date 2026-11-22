# 后续消融 batch（2026-11-22）：rf_long / lidar90 / lidar60 / learn2 / fixppt_f

> 5 组实验（6 个 run）在 Train 分支完成，swanlab 项目 BAMER-RL（云端，账号 Lee）。
> 训练吞吐 ~80-93 options/s；rf_long 280k ≈ 57 分钟，其余 160k ≈ 29-33 分钟。
> 评估协议：held-out 固定 seed 套件 8 seeds，贪心 rollout + 全知 BFS 参照
> （`tools/train.py --eval` 同路径，另计 POI 定位率 = search_steps 非 None 占比）。

## 结果总表（held-out 8 seeds）

| Run | 配置要点 | 成功率 | POI 定位率 | 总步数 | 搜寻步数 | 派送步数 | 探索率 | 墙知识率 | 与参照差距 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rf_long | frontier + stage1 240k（180 束） | 50.0% | 50.0% | 726.2 | 684.5 | 41.8 | 61.2% | 81.3% | 625.2 |
| lidar90 | frontier + 90 束 | 50.0% | 50.0% | 766.8 | 746.5 | 20.3 | 57.5% | 79.9% | 671.2 |
| lidar60 | frontier + 60 束 | 50.0% | 50.0% | 766.8 | 746.5 | 20.3 | 57.5% | 79.9% | 671.2 |
| learn2 | frontier + learn_every=2 | **75.0%** | **75.0%** | 750.3 | 712.2 | 38.2 | 64.0% | 83.1% | 649.5 |
| fixppt_f_s0 | fixppt + frontier_mask, seed 0 | 37.5% | 37.5% | 547.3 | 498.0 | 49.3 | 59.8% | 88.4% | 468.7 |
| fixppt_f_s1 | fixppt + frontier_mask, seed 1 | 50.0% | 50.0% | 569.8 | 550.8 | 19.0 | 54.9% | 81.7% | 503.5 |

对照基线的可用性：原始 frontier-160k（seed 0）与本批 lidar90/lidar60 **逐位等效**
（见下"降束零信息损失"，配置除束数外与 `armA_g0999_aux05_frontier.yaml` 全同、
同 seed → 训练流相同），故 lidar90/60 行即 frontier-160k 基线读数；原始
fixppt（固定图无 frontier）的 checkpoint 不在本机，A/B 对照待补（旧 run 数据
或在 swanlab 历史，或需重跑一臂）。

## 判据逐条对答

1. **rf_long（预算天花板）**：held-out 成功率 240k = 50% 与 160k（=50%）持平，
   未"仍在升"；总步数略降（766.8→726.2，−5.3%）、探索率升（57.5%→61.2%）。
   按判据字面（成功率仍在升才改 240k）→ **held-out 口径不支持改 240k 预算**；
   多花的 120k options 换来的是步数/探索质量边际改善。最终裁量归用户
   （训练曲线 mini-eval 六分位走势见 swanlab rf_long run）。
2. **lidar90 / lidar60（降束代价）**：两者与 180 束基线**逐位等效**（见下），
   降幅 = 0，远优于 ≤3-5pp 门槛 → **两档均达标**；信息无损前提下激进档 60 束
   更省（墙钟收益见注意事项 3）。
3. **learn2（批学习间隔）**：held-out 75% > 基线 50%（+25pp），步数持平略优 →
   **"不降"判据满足且反升，建议采纳**（训练流与基线逐位相同、权重不同，
   提升可信来源 = 学习间隔改变而非数据差异；8-seed 二项噪声 ±15-18pp，
   差异约 1.4σ，方向明确但量级待多种子确认）。
4. **fixppt_f（官方图 + frontier）**：seed 0 = 37.5%、seed 1 = 50%（两 seed
   方差大）；与 fixppt 无 frontier 对照的 A/B 判读待基线补齐。

## 降束零信息损失（本批意外发现，实测非推导）

lidar90 与 lidar60 两次**独立训练**最终 80/80 张量逐位相同、held-out 全指标
相同。机制：belief 是槽位级二值知识（free/wall，`algo/belief.py update_scan`），
束方向 6°/4°/2° 间距嵌套（60 ⊂ 90 ⊂ 180），DDA 在栅格对齐墙体 + 量程 2.0 下
60 束标记槽位集合与 180 束完全一致 → 观测流逐位相同 → 策略/网络/评估全同。

含义：**束数在 belief 表示层不承载额外信息**，"观测分辨率减半的代价"在当前
表示下恒为零；束数选择纯剩墙钟问题。边界条件：该等价是本迷宫分布 + 量程
2.0 + 槽位二值表示上的实证，若 belief 升级为连续占据概率或量程加大需重验。

## 注意事项与不确定性

1. 8-seed 二项噪声：50% 读数的 1σ ≈ ±17.7pp，75% ≈ ±15.3pp——单 run 间
   ±25pp 内的差异不足以强定论，采纳建议均需多种子复核（惯例同 λ_aux 消融）；
2. fixppt_f_s1 的评估 seed 套件由 seed=1 派生（rng(778)），与 seed 0 系
   列 run 不同套，跨 run 对比按同分布理解；
3. 墙钟收益未在本批干净测得：lidar90/60 共享 GPU 5、rf_long 独占 GPU 3，
   吞吐 84.9/83.9/82.5 options/s 被 co-tenancy 混淆；60 束理论省投射
   ~20%+（投射占墙钟 33.7%），如需量化须单卡独占复测；
4. POI 定位率在全部 run 中与成功率相等：失败 episode 均为超时截断
   （POI 未齐），未见"定位成功但派送失败"case——派送阶段在已知图上
   由 BFS 支撑，瓶颈全在搜寻阶段；
5. 全知参照差距 468-671 步：贪心策略离全知下限仍远，主要差在搜寻段
   的覆盖效率，与 frontier 收窄动机一致。

## 复现

- 配置：`algo/configs/armA_frontier_{rf_long,lidar90,lidar60,learn2}.yaml`、
  `algo/configs/armA_fixppt_frontier_{s0,s1}.yaml`；
- 训练：`python tools/train.py --config <上述yaml>`（swanlab cloud 已登记）；
- 评估：`python tools/train.py --config <yaml> --eval algo/runs/<name>/checkpoint_stage1_final.pt --eval-seeds 8`；
- POI 定位率：评估脚本为 train.py run_eval 附加 `search_steps is not None`
  占比（本批用临时脚本 `/tmp/bamer_eval_ext.py`，已删；如需固化为正式指标
  可并入 `algo/eval.py`）。
- swanlab run id：rf_long 6go0llub / lidar90 acduj7me / lidar60 b3h5tqcs /
  learn2 plbrvyy9 / fixppt_f_s0 pog1qjiz / fixppt_f_s1 9koa27k9（另有首轮
  缓冲启动的 6 个 <1 分钟 stub 与 1 个冒烟 run，可忽略）。
