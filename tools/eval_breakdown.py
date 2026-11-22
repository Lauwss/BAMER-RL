# -*- coding: utf-8 -*-
"""评估 + 失败模式分解：成功率 / 搜索超时 / 找到未送达（deliver-fail）。

rf 消融表"搜索超时 / 找到未送达"两列的独立工具化（train.py --eval 只打印
汇总）；与 train.py --eval 同一评估函数与 seed 派生（rng(cfg.seed+777)），
数字逐位可比。判据口径（EpisodeMetrics 语义，eval.py:161-169）：
  success        = terminated（全任务完成）
  search-timeout = search_steps is None（4 POI 未全定位即截断）
  deliver-fail   = search_steps 有值 且 非 success（找全 POI 后未完成派送）

用法：python tools/eval_breakdown.py --config algo/configs/<yaml> \
          --checkpoint algo/runs/<exp>/checkpoint_stage1_final.pt [--seeds 100]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="评估 + 失败模式分解（held-out 贪心）")
    ap.add_argument("--config", required=True, help="algo/configs/*.yaml（含评估口径 env_max_steps）")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seeds", type=int, default=100)
    args = ap.parse_args()

    import torch
    from algo.agent import SMDPDQNAgent
    from algo.belief import BeliefState
    from algo.eval import evaluate, summarize_with_reference
    from algo.executor import Executor
    from algo.trainer import TrainConfig, build_fixed_maze
    from bamer_env.env import BamerEnv
    from bamer_mouse.lidar import LidarSpec

    cfg = TrainConfig.from_yaml(args.config)
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    agent = SMDPDQNAgent(in_channels=14 + 2 * int(cfg.use_traj) + int(cfg.dist_channel)
                         + int(cfg.dir_channel),
                         n=cfg.maze_n, aux_head=cfg.aux_head,
                         dist_channel=cfg.dist_channel, dilated=cfg.dilated_backbone,
                         dir_channel=cfg.dir_channel,
                         device="cpu", seed=cfg.seed)
    agent.load_state_dict(ckpt["agent"])

    seeds = [int(s) for s in np.random.default_rng(cfg.seed + 777).integers(
        0, 2 ** 31 - 1, args.seeds)]
    belief = BeliefState(n=cfg.maze_n, max_range=cfg.lidar_max_range,
                         scan_memo=cfg.scan_memo, use_dist=cfg.dist_channel,
                         use_dir=cfg.dir_channel)
    executor = Executor(belief, max_option_steps=cfg.env_max_steps, lazy_replan=cfg.lazy_replan)

    topology = dict(
        maze=build_fixed_maze(cfg) if cfg.maze_fixed is not None else None,
        maze_braiding=cfg.maze_braiding_stage1,
        maze_loops_range=tuple(cfg.maze_loops_stage1) if cfg.maze_loops_stage1 is not None else None,
    )

    def factory():
        return BamerEnv(
            randomize=cfg.env_randomize, max_steps=cfg.env_max_steps,
            **topology,
            lidar=LidarSpec(n_beams=cfg.lidar_n_beams, max_range=cfg.lidar_max_range, noise_sigma=0.0),
        )

    summary = evaluate(agent, factory, seeds, belief, executor, suite="held-out",
                       frontier_only=cfg.frontier_mask)
    # 逐 seed 场景复现（参照计算需要布点 GT：评估侧特权，与 run_eval 同构）
    scenes = {}
    for seed in seeds:
        env = BamerEnv(randomize=cfg.env_randomize, max_steps=cfg.env_max_steps,
                       **topology,
                       lidar=LidarSpec(n_beams=cfg.lidar_n_beams, max_range=cfg.lidar_max_range))
        env.reset(seed=seed)
        scenes[seed] = env.scene
        env.close()
    summary = summarize_with_reference(summary, scenes)

    eps = summary.per_episode
    n = len(eps)
    succ = [e for e in eps if e.success]
    search_to = [e for e in eps if e.search_steps is None]
    dfail = [e for e in eps if e.search_steps is not None and not e.success]

    print(f"\n[评估分解] cfg={os.path.basename(args.config)}  n={n}  "
          f"env_max_steps={cfg.env_max_steps}")
    print(f"  成功率          {len(succ)}/{n} = {len(succ) / n:.1%}")
    print(f"  搜索超时        {len(search_to)}/{n} = {len(search_to) / n:.1%}"
          f"   （4 POI 未全定位即截断）")
    print(f"  找到未送达      {len(dfail)}/{n} = {len(dfail) / n:.1%}"
          f"   （找全 POI 后未完成派送）")
    if succ:
        print(f"  成功集平均总步数 {np.mean([e.total_steps for e in succ]):.1f}   "
              f"搜寻 {np.mean([e.search_steps for e in succ if e.search_steps is not None]):.1f}   "
              f"派送 {np.mean([e.delivery_steps for e in succ if e.delivery_steps is not None]):.1f}")
    print(f"  全体平均总步数   {np.mean([e.total_steps for e in eps]):.1f}")
    print(f"  探索区域率      {np.mean([e.explored_ratio for e in eps]):.2%}")
    if summary.mean_gap_to_reference is not None:
        print(f"  与全知参照差距  {summary.mean_gap_to_reference:.1f} 步（成功集）")


if __name__ == "__main__":
    main()
