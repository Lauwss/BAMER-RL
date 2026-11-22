# -*- coding: utf-8 -*-
"""repeat 解剖：behind/ahead 选择的供给、份额、代价与转移（2026-11-22）。

问题：若 tie-break 是均匀噪声，repeat 方向的选择权重应与其他方向相同，
repeat 不应占主导。本工具把"主导"分解为四个可测量（同一贪心 rollout）：

  1. 供给份额 —— 决策点上 behind/ahead/lateral 候选的构成（几何供给不对称）
  2. 选择份额 —— 被选前沿的 behind/ahead 分布；若 ≈ 供给份额 → 裁决均匀；
     若 behind 超额 → 距离近偏 × 几何的系统性偏置（Q 为状态的确定性函数，
     同一几何每次裁决同向——"固定偏置"而非"逐次重掷"）
  3. 代价不对称 —— 各类 option 的 τ / repeat 步数 / repeat 质量占比：
     即使选择对半，repeat 的步数质量仍可向 behind 集中（重走走廊 vs 踩新地）
  4. 转移矩阵 —— P(behind|behind) vs P(behind|ahead)：后退是否自锁（正反馈）

分类：a = cos∠(c−P, P−B)（P=当前格，B=上个决策边界格，commit_option 登记）；
behind a<−0.15、ahead a>+0.15、lateral 其余；首边界/零位移不定义。
转移链仅在连续 Explore option 间累计，Pick/Deliver 打断。

用法：python tools/repeat_anatomy.py --run-dir algo/runs/armA_frontier_dist_dilate [--seeds 30]
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
from algo.belief import BeliefState
from algo.executor import Executor
from algo.trainer import TrainConfig, build_fixed_maze, _repeat_moves
from bamer_env.env import BamerEnv
from bamer_mouse.lidar import LidarSpec

THRESH = 0.15        # |a| 阈值：behind(<−0.15) / ahead(>+0.15) / lateral


def classify(a: float) -> str:
    if a < -THRESH:
        return "behind"
    if a > THRESH:
        return "ahead"
    return "lateral"


def main() -> None:
    ap = argparse.ArgumentParser(description="repeat 解剖：behind/ahead 不对称分解")
    ap.add_argument("--run-dir", required=True, help="algo/runs/<exp>（含 config.json）")
    ap.add_argument("--checkpoint", default="checkpoint_stage1_final.pt")
    ap.add_argument("--seeds", type=int, default=30, help="held-out seed 数（cfg.seed+777 派生）")
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(os.path.join(args.run_dir, "config.json"))
    seeds = [int(s) for s in np.random.default_rng(cfg.seed + 777).integers(
        0, 2 ** 31 - 1, args.seeds)]

    torch.manual_seed(int(cfg.seed))
    ckpt = torch.load(os.path.join(args.run_dir, args.checkpoint),
                      weights_only=False, map_location="cpu")
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

    supply = {"behind": 0, "ahead": 0, "lateral": 0}       # 候选计数（所有决策点合计）
    supply_n = 0
    choices = {c: [] for c in supply}                      # 每类: (tau, repeat, moves, dist)
    n_explore = 0
    seq: list = []                                         # 连续 Explore 的类别链
    trans = {}                                             # (prev,cur) → 计数
    prev_was_explore = False

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
            action = int(np.argmax(np.where(mask, adv, -np.inf)))
            is_explore = action < 256 and bool(mask[action])

            defined = belief.prev_boundary_cell is not None
            if defined:
                vx = belief.cell[0] - belief.prev_boundary_cell[0]
                vy = belief.cell[1] - belief.prev_boundary_cell[1]
                defined = (vx != 0 or vy != 0)
            if defined and is_explore:
                front_idx = [a for a in range(256) if mask[a]]
                dist = belief.distance_field()
                px, py = belief.cell
                bx, by = belief.prev_boundary_cell
                vn = math.hypot(vx, vy)
                ali = {}
                for a in front_idx:
                    cx, cy = index_cell(a, n)
                    dx, dy = cx - px, cy - py
                    dn = math.hypot(dx, dy)
                    if dn > 0:
                        ali[a] = (dx * vx + dy * vy) / (dn * vn)
                if ali:
                    for a, val in ali.items():
                        supply[classify(val)] += 1
                    supply_n += len(ali)
                    chosen_cls = classify(ali[action])
                    n_explore += 1
                    if not prev_was_explore:
                        seq.clear()                        # Pick/Deliver 打断转移链
                    seq.append(chosen_cls)
                    if len(seq) >= 2:
                        key = (seq[-2], seq[-1])
                        trans[key] = trans.get(key, 0) + 1

            visited_before = set(belief.visited)
            result = executor.run(env, action)
            belief.commit_option(result.path, result.start_cell)
            if defined and is_explore and ali and action in ali:
                moves = max(len(result.path) - 1, 0)
                choices[classify(ali[action])].append(
                    (result.steps, _repeat_moves(result.path, visited_before), moves,
                     int(dist[index_cell(action, n)])))
            if result.steps == 0:
                abort_guard += 1
                if abort_guard >= 8:
                    break
            else:
                abort_guard = 0
            if result.terminated or result.truncated:
                break
            prev_was_explore = is_explore
        env.close()
        seq.clear()          # episode 边界打断转移链
        prev_was_explore = False

    total_choice = sum(len(v) for v in choices.values())
    total_rep = sum(r for v in choices.values() for (_, r, _, _) in v)
    print(f"\n[repeat 解剖] run={args.run_dir}  n_seeds={len(seeds)}  "
          f"explore 对齐可定义 option={total_choice}  候选={supply_n}")
    if total_choice == 0:
        return
    print(f"  候选供给份额 : " + "  ".join(
        f"{k} {supply[k] / supply_n:.1%}" for k in ("behind", "ahead", "lateral")))
    print(f"  选择份额     : " + "  ".join(
        f"{k} {len(choices[k]) / total_choice:.1%}" for k in ("behind", "ahead", "lateral")))
    print(f"  {'类别':<8} {'τ̄':>8} {'repeat̄':>8} {'repeat/move':>12} {'距离̄':>7} {'repeat质量占比':>14}")
    for k in ("behind", "ahead", "lateral"):
        rows = choices[k]
        if not rows:
            print(f"  {k:<8} {'—':>8}")
            continue
        arr = np.array(rows, dtype=float)
        mass = arr[:, 1].sum() / total_rep if total_rep else float("nan")
        print(f"  {k:<8} {arr[:, 0].mean():>8.1f} {arr[:, 1].mean():>8.2f} "
              f"{(arr[:, 1].sum() / max(arr[:, 2].sum(), 1)):>12.3f} "
              f"{arr[:, 3].mean():>7.1f} {mass:>14.1%}")
    n_trans = sum(trans.values())
    if n_trans:
        print(f"  转移（n={n_trans}）: "
              f"P(B|B)={trans.get(('behind', 'behind'), 0) / max(sum(v for (p, _), v in trans.items() if p == 'behind'), 1):.2f}  "
              f"P(B|A)={trans.get(('ahead', 'behind'), 0) / max(sum(v for (p, _), v in trans.items() if p == 'ahead'), 1):.2f}  "
              f"P(A|A)={trans.get(('ahead', 'ahead'), 0) / max(sum(v for (p, _), v in trans.items() if p == 'ahead'), 1):.2f}  "
              f"P(A|B)={trans.get(('behind', 'ahead'), 0) / max(sum(v for (p, _), v in trans.items() if p == 'behind'), 1):.2f}")


if __name__ == "__main__":
    main()
