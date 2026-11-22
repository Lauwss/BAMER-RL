# -*- coding: utf-8 -*-
"""Explore 决策距离敏感度探针（2026-11-22 感受野批评核实实验，tools/train.py 评估同口径）。

对给定 checkpoint 的贪心评估逐 option 记录：
  - 合法前沿集（legal_mask frontier_only，与策略评估同掩码语义）
  - 各前沿 BFS 距离（belief.distance_field 同源：开放边界=已探明无墙、悲观可达）
  - 各前沿 A_explore 优势值（argmax 等价口径：dueling 的 V/mean_legal 为状态常数）
  - 各前沿局部未知槽位数（半径 1，"局部信息量"）

检验指标（感受野修复消融的验收口径）：
  选中=最近比例 / 选中距离秩(0=最近) —— 代价侧近偏形态
  Spearman(距离, 优势)                —— 代价侧：代理(旧栈 −0.304) vs 精确(距离通道)
  Spearman(局部未知槽, 优势)          —— 收益侧：旧栈 +0.04 ≈ 零信号，膨胀主干预期转正
  Spearman(对齐度, 优势)              —— 方向连续性预检 L1（2026-11-22）：a(c)=cos∠(c−P, P−B)，
                                        打分是否表达"候选在我来路方向"；≈0 = 排序信号真空，
                                        方向特征前提成立；显著正 = 前提死（特征计划叫停）

用法：
    python tools/explore_probe.py --run-dir algo/runs/armA_frontier_dist [--seeds 30]
                                  [--checkpoint checkpoint_stage1_final.pt]
"""
import argparse
import math
import os
import sys
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch

from algo.actions import index_cell, legal_mask
from algo.agent import SMDPDQNAgent
from algo.belief import HEADINGS, W_UNKNOWN, BeliefState
from algo.executor import Executor
from algo.trainer import TrainConfig, build_fixed_maze
from bamer_env.env import BamerEnv
from bamer_mouse.lidar import LidarSpec

OFFS = ((0, 1), (1, 0), (0, -1), (-1, 0))   # N E S W，与 belief._DIRS/actions 同序


def rankcorr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman 秩相关（常数向量返回 nan，调用方过滤）。"""
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def run_probe(cfg: TrainConfig, checkpoint: str, seeds) -> None:
    torch.manual_seed(int(cfg.seed))
    ckpt = torch.load(checkpoint, weights_only=False, map_location="cpu")
    agent = SMDPDQNAgent(in_channels=14 + 2 * int(cfg.use_traj) + int(cfg.dist_channel)
                         + int(cfg.dir_channel),
                         n=cfg.maze_n, aux_head=cfg.aux_head,
                         dist_channel=cfg.dist_channel, dilated=cfg.dilated_backbone,
                         dir_channel=cfg.dir_channel,
                         device="cpu", seed=cfg.seed)
    agent.load_state_dict(ckpt["agent"])
    net = agent.online

    belief = BeliefState(n=cfg.maze_n, max_range=cfg.lidar_max_range,
                         scan_memo=cfg.scan_memo, use_dist=cfg.dist_channel,
                         use_dir=cfg.dir_channel)
    executor = Executor(belief, max_option_steps=cfg.env_max_steps, lazy_replan=cfg.lazy_replan)

    def factory():
        return BamerEnv(
            randomize=cfg.env_randomize, max_steps=cfg.env_max_steps,
            maze=build_fixed_maze(cfg) if cfg.maze_fixed is not None else None,
            maze_braiding=cfg.maze_braiding_stage1,
            maze_loops_range=tuple(cfg.maze_loops_stage1) if cfg.maze_loops_stage1 is not None else None,
            lidar=LidarSpec(n_beams=cfg.lidar_n_beams, max_range=cfg.lidar_max_range, noise_sigma=0.0),
        )

    def unknown_slots(cell):
        return sum(1 for d in HEADINGS
                   if belief.ledger.wall_value(cell[0], cell[1], d) == W_UNKNOWN)

    rows = []          # (d_chosen, d_min, d_mean, rank_norm, ρ(距离,优势), ρ(局部,优势), n_front,
                       #  ρ(对齐度,优势), 搜索期)
    n_options = n_explore = 0
    for seed in seeds:
        env = factory()
        obs, info = env.reset(seed=seed)
        belief.reset(env.scene.maze.n)
        belief.observe(obs, info)
        n = belief.n
        abort_guard = 0
        while True:
            s = belief.tensor(use_traj=cfg.use_traj)
            mask = legal_mask(belief, frontier_only=cfg.frontier_mask)
            with torch.no_grad():
                adv = net(torch.from_numpy(s).unsqueeze(0), compute_aux=False)["advantage"][0].numpy()
            action = int(np.argmax(np.where(mask, adv, -np.inf)))     # 贪心等价（常数消去）
            n_options += 1
            if action < 256 and mask[action]:
                n_explore += 1
                front_idx = [a for a in range(256) if mask[a]]
                front = [index_cell(a, n) for a in front_idx]
                dist = belief.distance_field()
                ds = np.array([dist[c] for c in front], dtype=float)
                avs = np.array([adv[a] for a in front_idx])
                loc = np.array([unknown_slots(c) for c in front], dtype=float)
                # 对齐度（方向连续性预检 L1）：a(c)=cos∠(c−P, P−B)。B=上个决策边界格
                # （commit_option 登记，belief.prev_boundary_cell）；首边界/零位移 option
                # → 整列 nan（报告期过滤）。frontier 掩码排除自身格，dn>0 恒成立。
                ali = np.full(len(front), np.nan)
                if belief.prev_boundary_cell is not None:
                    vx = belief.cell[0] - belief.prev_boundary_cell[0]
                    vy = belief.cell[1] - belief.prev_boundary_cell[1]
                    if vx != 0 or vy != 0:
                        vn = math.hypot(vx, vy)
                        for i, (cx, cy) in enumerate(front):
                            dx, dy = cx - belief.cell[0], cy - belief.cell[1]
                            dn = math.hypot(dx, dy)
                            if dn > 0:
                                ali[i] = (dx * vx + dy * vy) / (dn * vn)
                if len(front) > 1 and (ds > -1).all():
                    d_chosen = int(dist[index_cell(action, n)])
                    rank = float((ds < d_chosen).sum()) / (len(ds) - 1)   # 0=最近（并列偏保守）
                    ali_rc = rankcorr(ali, avs) if np.isfinite(ali).all() else float("nan")
                    search = 0 if belief.poi_complete() else 1            # 搜索期=POI 未齐
                    rows.append((d_chosen, float(ds.min()), float(ds.mean()), rank,
                                 rankcorr(ds, avs), rankcorr(loc, avs), len(front),
                                 ali_rc, search))
            result = executor.run(env, action)
            belief.commit_option(result.path, result.start_cell)
            if result.steps == 0:
                abort_guard += 1                     # 不可达中止保险丝（与 eval 同判据）
                if abort_guard >= 8:
                    break
            else:
                abort_guard = 0
            if result.terminated or result.truncated:
                break
        env.close()

    arr = np.array(rows)
    print(f"\n[Explore 探针] run={cfg.out_dir}  n_seeds={len(seeds)}  options={n_options}  "
          f"explore 决策={n_explore}  可比较={len(arr)}")
    if len(arr) == 0:
        return
    print(f"  选中的前沿 = 最近前沿比例            : {np.mean(arr[:, 0] == arr[:, 1]):.1%}")
    print(f"  选中目标距离 vs 全前沿平均距离        : {arr[:, 0].mean():.2f} vs {arr[:, 2].mean():.2f}")
    print(f"  选中目标的距离秩（0=最近 1=最远）     : 均值 {arr[:, 3].mean():.3f}")
    cd = arr[:, 4][~np.isnan(arr[:, 4])]
    cl = arr[:, 5][~np.isnan(arr[:, 5])]
    print(f"  逐 option Spearman(距离, 优势)       : 均值 {np.nanmean(cd):+.3f}（n={len(cd)}/{len(arr)}）")
    print(f"  逐 option Spearman(局部未知槽, 优势) : 均值 {np.nanmean(cl):+.3f}（n={len(cl)}/{len(arr)}）")
    ok_a = ~np.isnan(arr[:, 7])
    ca = arr[ok_a, 7]
    cs = arr[ok_a & (arr[:, 8] == 1), 7]
    cs_txt = f"{np.nanmean(cs):+.3f}（n={len(cs)}）" if len(cs) else "—（无样本）"
    print(f"  逐 option Spearman(对齐度, 优势)     : 均值 {np.nanmean(ca):+.3f}（n={len(ca)}/{len(arr)}；"
          f"搜索期 {cs_txt}）")
    d = arr[:, 0]
    print(f"  选中距离分布: p25={np.percentile(d, 25):.0f} 中位={np.median(d):.0f} "
          f"p75={np.percentile(d, 75):.0f} max={d.max():.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Explore 决策距离敏感度探针")
    ap.add_argument("--run-dir", required=True, help="algo/runs/<exp>（含 config.json）")
    ap.add_argument("--checkpoint", default="checkpoint_stage1_final.pt")
    ap.add_argument("--seeds", type=int, default=30, help="held-out seed 数（cfg.seed+777 派生）")
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(os.path.join(args.run_dir, "config.json"))
    seeds = [int(s) for s in np.random.default_rng(cfg.seed + 777).integers(
        0, 2 ** 31 - 1, args.seeds)]
    run_probe(cfg, os.path.join(args.run_dir, args.checkpoint), seeds)


if __name__ == "__main__":
    main()
