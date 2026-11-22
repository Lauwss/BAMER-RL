# -*- coding: utf-8 -*-
"""轨迹通道因果干预探针（2026-11-22 方向连续性预检 L3）。

问题：L1（explore_probe 对齐度列）测得 ρ(对齐度, 优势)≈+0.05——相关性层面的信号
真空，但相关性 ≠ 因果。本工具在评估期直接干预输入通道，同种子贪心对比行为：

  baseline    原样输入
  zero-traj   置零 CH_TRAIL_PREV_OPTION / CH_POSE_PREV_BOUNDARY（14/15）
              —— 若行为不变 → 历史通道因果死亡（方向特征前提的最强单一数字）
  zero-dist   置零 CH_DIST（16）—— 阳性对照：L1 同款探针已测得 ρ(距离,优势)=−0.445
              （网络确在使用距离通道），此臂必须产生显著行为变化；否则干预管线
              本身无效，zero-traj 的阴性结果不可采信

指标与 eval.py 同口径：success=terminated；贪心 ε=0；abort 连续 8 次零步判死锁。
repeat 口径与 trainer._run_episode 一致（option 前快照 visited，路径重走计数）。

用法：python tools/intervene_probe.py --run-dir algo/runs/armA_frontier_dist_dilate [--seeds 30]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch

from algo.actions import legal_mask
from algo.agent import SMDPDQNAgent
from algo.belief import CH_DIST, CH_POSE_PREV_BOUNDARY, CH_TRAIL_PREV_OPTION, BeliefState
from algo.executor import Executor
from algo.trainer import TrainConfig, build_fixed_maze, _repeat_moves
from bamer_env.env import BamerEnv
from bamer_mouse.lidar import LidarSpec


def run_arm(cfg: TrainConfig, checkpoint: str, seeds, zero) -> list:
    """zero=None 原样；否则为要置零的通道索引元组。返回逐 seed 行。"""
    torch.manual_seed(int(cfg.seed))
    ckpt = torch.load(checkpoint, weights_only=False, map_location="cpu")
    agent = SMDPDQNAgent(in_channels=14 + 2 * int(cfg.use_traj) + int(cfg.dist_channel),
                         n=cfg.maze_n, aux_head=cfg.aux_head,
                         dist_channel=cfg.dist_channel, dilated=cfg.dilated_backbone,
                         device="cpu", seed=cfg.seed)
    agent.load_state_dict(ckpt["agent"])

    belief = BeliefState(n=cfg.maze_n, max_range=cfg.lidar_max_range,
                         scan_memo=cfg.scan_memo, use_dist=cfg.dist_channel)
    executor = Executor(belief, max_option_steps=cfg.env_max_steps, lazy_replan=cfg.lazy_replan)

    def factory():
        return BamerEnv(
            randomize=cfg.env_randomize, max_steps=cfg.env_max_steps,
            maze=build_fixed_maze(cfg) if cfg.maze_fixed is not None else None,
            maze_braiding=cfg.maze_braiding_stage1,
            maze_loops_range=tuple(cfg.maze_loops_stage1) if cfg.maze_loops_stage1 is not None else None,
            lidar=LidarSpec(n_beams=cfg.lidar_n_beams, max_range=cfg.lidar_max_range, noise_sigma=0.0),
        )

    rows = []          # (terminated, steps, search_steps, move_steps, repeat_moves)
    for seed in seeds:
        env = factory()
        obs, info = env.reset(seed=seed)
        belief.reset(env.scene.maze.n)
        belief.observe(obs, info)
        steps = move_steps = repeat_moves = 0
        search_steps = None
        abort_guard = 0
        while True:
            s = belief.tensor(use_traj=cfg.use_traj)
            if zero is not None:
                s = s.copy()
                s[list(zero)] = 0
            mask = legal_mask(belief, frontier_only=cfg.frontier_mask)
            action = agent.act(s, mask, greedy=True)
            visited_before = set(belief.visited)
            result = executor.run(env, action)
            steps += result.steps
            move_steps += max(len(result.path) - 1, 0)
            repeat_moves += _repeat_moves(result.path, visited_before)
            belief.commit_option(result.path, result.start_cell)
            if search_steps is None and belief.poi_complete():
                search_steps = steps
            if result.steps == 0:
                abort_guard += 1        # 不可达中止：连续 8 次零步判死锁（与 eval 同判据）
                if abort_guard >= 8:
                    break
            else:
                abort_guard = 0
            if result.terminated or result.truncated:
                break
        env.close()
        rows.append((int(result.terminated), steps,
                     float("nan") if search_steps is None else float(search_steps),
                     move_steps, repeat_moves))
    return rows


def summarize(rows: list) -> dict:
    arr = np.array(rows, dtype=float)
    ok = arr[:, 0] == 1
    mean = lambda v: float(np.mean(v)) if len(v) else float("nan")   # noqa: E731
    so = ok & ~np.isnan(arr[:, 2])
    return {
        "n": len(arr),
        "success": float(np.mean(ok)) if len(arr) else 0.0,
        "steps_ok": mean(arr[ok, 1]),
        "steps_all": mean(arr[:, 1]),
        "search_ok": mean(arr[so, 2]),
        "repeat": mean(arr[:, 4]),
        "repeat_ratio": mean(arr[:, 4] / np.maximum(arr[:, 3], 1.0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="轨迹通道因果干预探针（L3）")
    ap.add_argument("--run-dir", required=True, help="algo/runs/<exp>（含 config.json）")
    ap.add_argument("--checkpoint", default="checkpoint_stage1_final.pt")
    ap.add_argument("--seeds", type=int, default=30, help="held-out seed 数（cfg.seed+777 派生）")
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(os.path.join(args.run_dir, "config.json"))
    if not cfg.use_traj:
        raise SystemExit("zero-traj 干预要求 use_traj=True（无轨迹通道可干预）")
    seeds = [int(s) for s in np.random.default_rng(cfg.seed + 777).integers(
        0, 2 ** 31 - 1, args.seeds)]
    ckpt = os.path.join(args.run_dir, args.checkpoint)

    arms = {
        "baseline": None,
        "zero-traj": (CH_TRAIL_PREV_OPTION, CH_POSE_PREV_BOUNDARY),
        "zero-dist": (CH_DIST,),
    }
    raws = {}
    results = {}
    for name, zero in arms.items():
        raws[name] = run_arm(cfg, ckpt, seeds, zero)
        results[name] = summarize(raws[name])

    print(f"\n[干预探针] run={args.run_dir}  n_seeds={len(seeds)}")
    print(f"  {'arm':<10} {'success':>8} {'steps(ok)':>10} {'steps(all)':>11} "
          f"{'repeat':>8} {'repeat/move':>12} {'search(ok)':>11}")
    for name, m in results.items():
        print(f"  {name:<10} {m['success']:>8.1%} {m['steps_ok']:>10.1f} {m['steps_all']:>11.1f} "
              f"{m['repeat']:>8.2f} {m['repeat_ratio']:>12.3f} {m['search_ok']:>11.1f}")

    base = raws["baseline"]
    print("\n  配对差值（干预 − baseline，同种子逐条）:")
    for name in ("zero-traj", "zero-dist"):
        raw = raws[name]
        d_succ = sum(1 for a, b in zip(raw, base) if a[0] > b[0])
        d_fail = sum(1 for a, b in zip(raw, base) if a[0] < b[0])
        d_steps = np.mean([a[1] - b[1] for a, b in zip(raw, base)])
        d_rep = np.mean([a[4] - b[4] for a, b in zip(raw, base)])
        print(f"    {name:<10} Δsuccess {d_succ:+d}成/{d_fail:+d}败   "
              f"Δsteps {d_steps:+.1f}   Δrepeat {d_rep:+.2f}")


if __name__ == "__main__":
    main()
