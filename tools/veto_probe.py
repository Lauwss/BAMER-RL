# -*- coding: utf-8 -*-
"""behind 名额否决探针（2026-11-22，用户提案"对 behind 名额消减"的零训练检验）。

动机：repeat 解剖显示 behind 以 49.2% 的选择份额占 70.2% 的 repeat 步数质量，
且裁决对 behind/ahead 实质公平（选择份额≈供给份额）。若 behind 超额名额
是净有害的，评估期强制否决应改善 held-out 表现——无需训练即可为
dir_channel 提供上限证据（headroom）。

三臂设计（同种子贪心，逐 option 干预）：
  baseline     原样 argmax
  veto-behind  argmax 选中 behind（a<−0.15）时改选 ahead/lateral 中优势最高者；
               无非 behind 候选则保留原选（计数 no-alt）
  veto-ahead   对称对照：选中 ahead（a>+0.15）时改选 behind/lateral 中最优者

判读（混淆控制）：veto-behind 同时混杂"远离回头路"与"趋向更近目标"两种效应
（ahead 被选距离 4.8 vs behind 8.2）。若 veto-behind 改善且 veto-ahead 恶化
→ 方向特异性效应（behind 超额确有害）；若两臂同向 → 只是"别走深"的近距效应，
名额消减与方向无关，dir_channel 的预期收益需重估。

注意：veto 作用于**未重训**的策略（其 Q 从未学过系统性方向偏好），测得的是
当前策略下 behind/ahead 分配的净价值，不是 field 重训后的上限。

用法：python tools/veto_probe.py --run-dir algo/runs/armA_frontier_dist_dilate [--seeds 30]
"""
import argparse
import math
import os
import sys

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

THRESH = 0.15


def classify(a: float) -> str:
    if a < -THRESH:
        return "behind"
    if a > THRESH:
        return "ahead"
    return "lateral"


def run_arm(cfg: TrainConfig, checkpoint: str, seeds, mode: str) -> dict:
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

    n_success = n_steps = n_repeat = 0
    n_explore = 0
    cls_count = {"behind": 0, "ahead": 0, "lateral": 0}
    n_veto = n_no_alt = 0
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
                adv = net_adv = None
                adv = agent.online(torch.from_numpy(s).unsqueeze(0),
                                   compute_aux=False)["advantage"][0].numpy()
            action = int(np.argmax(np.where(mask, adv, -np.inf)))
            is_explore = action < 256 and bool(mask[action])

            defined = belief.prev_boundary_cell is not None
            ali = {}
            if defined and is_explore:
                vx = belief.cell[0] - belief.prev_boundary_cell[0]
                vy = belief.cell[1] - belief.prev_boundary_cell[1]
                if vx == 0 and vy == 0:
                    defined = False
            if defined and is_explore:
                px, py = belief.cell
                bx, by = belief.prev_boundary_cell
                vn = math.hypot(vx, vy)
                for a in range(256):
                    if mask[a]:
                        cx, cy = index_cell(a, n)
                        dx, dy = cx - px, cy - py
                        dn = math.hypot(dx, dy)
                        if dn > 0:
                            ali[a] = (dx * vx + dy * vy) / (dn * vn)
            if defined and is_explore and ali:
                cls = classify(ali.get(action, 0.0))
                target = "behind" if mode == "veto_behind" else "ahead"
                if mode != "baseline" and cls == target:
                    others = [a for a in ali if classify(ali[a]) != target]
                    if others:
                        action = max(others, key=lambda a: adv[a])
                        n_veto += 1
                    else:
                        n_no_alt += 1
                n_explore += 1
                cls_count[classify(ali.get(action, 0.0))] += 1

            visited_before = set(belief.visited)
            result = executor.run(env, action)
            belief.commit_option(result.path, result.start_cell)
            n_steps += result.steps
            n_repeat += _repeat_moves(result.path, visited_before)
            if result.steps == 0:
                abort_guard += 1
                if abort_guard >= 8:
                    break
            else:
                abort_guard = 0
            if result.terminated or result.truncated:
                n_success += int(result.terminated)
                break
        env.close()
    return {"success": n_success / len(seeds), "steps": n_steps / len(seeds),
            "repeat": n_repeat / len(seeds), "explore": n_explore,
            "behind_share": cls_count["behind"] / max(n_explore, 1),
            "ahead_share": cls_count["ahead"] / max(n_explore, 1),
            "veto": n_veto, "no_alt": n_no_alt}


def main() -> None:
    ap = argparse.ArgumentParser(description="behind 名额否决探针（零训练）")
    ap.add_argument("--run-dir", required=True, help="algo/runs/<exp>（含 config.json）")
    ap.add_argument("--checkpoint", default="checkpoint_stage1_final.pt")
    ap.add_argument("--seeds", type=int, default=30, help="held-out seed 数（cfg.seed+777 派生）")
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(os.path.join(args.run_dir, "config.json"))
    seeds = [int(s) for s in np.random.default_rng(cfg.seed + 777).integers(
        0, 2 ** 31 - 1, args.seeds)]
    ckpt = os.path.join(args.run_dir, args.checkpoint)

    results = {}
    for mode in ("baseline", "veto_behind", "veto_ahead"):
        results[mode] = run_arm(cfg, ckpt, seeds, mode)

    print(f"\n[否决探针] run={args.run_dir}  n_seeds={len(seeds)}")
    print(f"  {'arm':<12} {'success':>8} {'steps':>9} {'repeat':>8} "
          f"{'behind份额':>10} {'ahead份额':>10} {'veto次数':>9}")
    for mode, m in results.items():
        print(f"  {mode:<12} {m['success']:>8.1%} {m['steps']:>9.1f} {m['repeat']:>8.2f} "
              f"{m['behind_share']:>10.1%} {m['ahead_share']:>10.1%} "
              f"{m['veto']:>9d}" + (f"（无替代 {m['no_alt']}）" if m["no_alt"] else ""))

    base = results["baseline"]
    print("\n  配对基线差（仅 success 为逐 seed 可配对均值口径，steps/repeat 为总量均值差）:")
    for mode in ("veto_behind", "veto_ahead"):
        m = results[mode]
        print(f"    {mode:<12} Δsuccess {m['success'] - base['success']:+.1%}  "
              f"Δsteps {m['steps'] - base['steps']:+.1f}  Δrepeat {m['repeat'] - base['repeat']:+.2f}")


if __name__ == "__main__":
    main()
