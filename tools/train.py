# -*- coding: utf-8 -*-
"""训练入口：一次实验一个 yaml 配置（PLAN.md 阶段三设施）。

本地冒烟（分钟级，验证管线通）：
    python tools/train.py --config algo/configs/smoke.yaml

服务器正式训练（三臂消融，BAMER 环境）：
    python tools/train.py --config algo/configs/armA_search_pretrain.yaml
    python tools/train.py --config algo/configs/armB_entropy_pretrain.yaml
    python tools/train.py --config algo/configs/armC_baseline.yaml
    # 断点恢复：python tools/train.py --config ... --resume algo/runs/<exp>/checkpoint_stage1_final.pt
    # 训后评估：python tools/train.py --config algo/configs/smoke.yaml \
    #              --eval algo/runs/smoke/checkpoint_full_final.pt --eval-seeds 8

复现性：同 seed 两次训练曲线一致（CPU 严格；GPU 允许非严格误差，需声明）。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np


def _topology_kwargs(cfg):
    """评估环境拓扑与训练 stage1 同分布（braiding + 补环，见 TrainConfig 注释）。

    返回 kwargs 供环境工厂与场景复现共用——两处必须一致，否则评估场景与
    全知参照场景不匹配。maze_fixed 时返回固定实例（与训练共享同一张）。
    """
    from algo.trainer import build_fixed_maze

    return dict(
        maze=build_fixed_maze(cfg) if cfg.maze_fixed is not None else None,
        maze_braiding=cfg.maze_braiding_stage1,
        maze_loops_range=tuple(cfg.maze_loops_stage1) if cfg.maze_loops_stage1 is not None else None,
    )


def build_env_factory(cfg):
    """评估用环境工厂（held-out 固定种子套件；官方图套件 env_randomize=False）。"""
    from bamer_env.env import BamerEnv
    from bamer_mouse.lidar import LidarSpec
    from algo.trainer import TrainConfig

    def factory():
        return BamerEnv(
            randomize=cfg.env_randomize, max_steps=cfg.env_max_steps,
            **_topology_kwargs(cfg),
            lidar=LidarSpec(n_beams=cfg.lidar_n_beams,
                            max_range=cfg.lidar_max_range, noise_sigma=0.0),
        )
    return factory


def run_eval(config_path: str, checkpoint: str, n_seeds: int) -> None:
    """载入 checkpoint，在固定 seed 套件上贪心评估 + 全知参照差距。"""
    import torch

    from algo.actions import legal_mask
    from algo.agent import SMDPDQNAgent
    from algo.belief import BeliefState
    from algo.eval import evaluate, omniscient_reference, summarize_with_reference
    from algo.executor import Executor
    from algo.trainer import TrainConfig

    cfg = TrainConfig.from_yaml(config_path)
    ckpt = torch.load(checkpoint, weights_only=False, map_location="cpu")
    agent = SMDPDQNAgent(in_channels=14 + 2 * int(cfg.use_traj) + int(cfg.dist_channel)
                         + int(cfg.dir_channel),
                         n=cfg.maze_n, aux_head=cfg.aux_head,
                         dist_channel=cfg.dist_channel, dilated=cfg.dilated_backbone,
                         dir_channel=cfg.dir_channel,
                         device="cpu", seed=cfg.seed)
    agent.load_state_dict(ckpt["agent"])

    seeds = [int(s) for s in np.random.default_rng(cfg.seed + 777).integers(0, 2**31 - 1, n_seeds)]
    belief = BeliefState(n=cfg.maze_n, max_range=cfg.lidar_max_range, scan_memo=cfg.scan_memo,
                         use_dist=cfg.dist_channel, use_dir=cfg.dir_channel)
    executor = Executor(belief, max_option_steps=cfg.env_max_steps, lazy_replan=cfg.lazy_replan)
    summary = evaluate(agent, build_env_factory(cfg), seeds, belief, executor, suite="held-out",
                       frontier_only=cfg.frontier_mask,
                       fallback_unvisited=cfg.fallback_unvisited)

    # 逐 seed 场景复现（参照计算需要布点 GT：评估侧特权）
    from bamer_env.env import BamerEnv
    from bamer_mouse.lidar import LidarSpec
    scenes = {}
    for seed in seeds:
        env = BamerEnv(randomize=cfg.env_randomize, max_steps=cfg.env_max_steps,
                       **_topology_kwargs(cfg),
                       lidar=LidarSpec(n_beams=cfg.lidar_n_beams, max_range=cfg.lidar_max_range))
        env.reset(seed=seed)
        scenes[seed] = env.scene
        env.close()
    summary = summarize_with_reference(summary, scenes)

    print(f"\n[评估套件 {summary.suite}] {summary.n_episodes} seeds")
    print(f"  成功率        {summary.success_rate:.2%}")
    if summary.mean_total_steps is not None:
        print(f"  平均总步数    {summary.mean_total_steps:.1f}")
        print(f"  平均搜寻步数  {summary.mean_search_steps}")
        print(f"  平均派送步数  {summary.mean_delivery_steps}")
        print(f"  平均 option 数 {summary.mean_options:.1f}")
    print(f"  探索区域率    {summary.mean_explored_ratio:.2%}")
    print(f"  墙知识率      {summary.mean_wall_knowledge_ratio:.2%}")
    if summary.mean_gap_to_reference is not None:
        print(f"  与全知参照差距 {summary.mean_gap_to_reference:.1f} 步")


def main() -> None:
    # 可选 CPU 线程限额（BAMER_TORCH_THREADS，如 3）：多进程并行训练时 torch
    # 默认按全核开 intra-op 线程，N 进程 × 全核 → 超订阅；实测 3 并行各限 3-4
    # 线程聚合吞吐 +25%（2026-11-21）。不设则保持 torch 默认。
    threads = os.environ.get("BAMER_TORCH_THREADS")
    if threads:
        import torch
        torch.set_num_threads(int(threads))

    ap = argparse.ArgumentParser(description="BAMER-RL 训练/评估入口")
    ap.add_argument("--config", required=True, help="algo/configs/*.yaml")
    ap.add_argument("--resume", default=None, help="checkpoint 路径（恢复训练）")
    ap.add_argument("--eval", default=None, help="checkpoint 路径（只评估不训练）")
    ap.add_argument("--eval-seeds", type=int, default=8, help="评估 seed 数")
    args = ap.parse_args()

    if args.eval:
        run_eval(args.config, args.eval, args.eval_seeds)
        return

    from algo.trainer import TrainConfig, Trainer

    cfg = TrainConfig.from_yaml(args.config)
    trainer = Trainer(cfg)
    print(f"[训练开始] arm={cfg.arm} out_dir={cfg.out_dir} seed={cfg.seed} "
          f"device={trainer.agent.device}")
    summary = trainer.run(resume_from=args.resume)
    print(f"[训练结束] {summary}")


if __name__ == "__main__":
    main()
