# -*- coding: utf-8 -*-
"""表征探针：f(c) 能否线性解码对齐度 a(c)（2026-11-22 方向连续性预检 L2）。

L1（决策级 ρ(对齐度,优势)=+0.05）与 L3（因果干预：zero-traj 噪声级、zero-dist
阳性对照通过）已证：打分不表达、也不因果使用历史/对齐信号。本工具下探一层——
主干特征 f(c)（backbone 输出 64 维，取自候选格自身位置）里，对齐度是否"在场"：

  - 线性岭回归探针 f(c) → a(c)，按锚点可达性分桶（Chebyshev 半径 R = RF/2，
    dilated→7 / 标准→3）：
      both_in   P、B 都在 c 窗口内      —— 比较物理上可计算的集合
      p_only / b_only                    —— 单锚点在场（走廊残迹仍可携带相关证据）
      arch-zero  双锚点窗外且窗口内无走廊像素 —— 信息论保证不可解码（架构零点）
  - 按 seed 划分 train/test（同 episode 状态强相关，禁止随行随机划分）
  - 置换对照：打乱 y 训练的探针 → R² 应 ≤0（管线有效性检验）

解读：both_in 内 R²>0 而桶外≈0 = "在场但未用"；全场 ≈0 = 比较从未被计算。
桶占比本身即"两锚点几何稀缺度"的实测（方向讨论的定量版）。

用法：python tools/align_probe.py --run-dir algo/runs/armA_frontier_dist_dilate [--seeds 30]
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
from algo.belief import CH_TRAIL_PREV_OPTION, BeliefState
from algo.executor import Executor
from algo.trainer import TrainConfig, build_fixed_maze
from bamer_env.env import BamerEnv
from bamer_mouse.lidar import LidarSpec


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1.0):
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    Z = (X - mu) / sd
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    return mu, sd, np.linalg.solve(A, Z.T @ y)


def r2(X: np.ndarray, y: np.ndarray, mu, sd, w) -> float:
    Z = (X - mu) / sd
    sse = float(((y - Z @ w) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    return 1.0 - sse / sst if sst > 0 else float("nan")


def collect(cfg: TrainConfig, checkpoint: str, seeds):
    """贪心 rollout 采集 (特征, 对齐度, dP, dB, 无历史证据标志) 行。"""
    torch.manual_seed(int(cfg.seed))
    ckpt = torch.load(checkpoint, weights_only=False, map_location="cpu")
    agent = SMDPDQNAgent(in_channels=14 + 2 * int(cfg.use_traj) + int(cfg.dist_channel),
                         n=cfg.maze_n, aux_head=cfg.aux_head,
                         dist_channel=cfg.dist_channel, dilated=cfg.dilated_backbone,
                         device="cpu", seed=cfg.seed)
    agent.load_state_dict(ckpt["agent"])
    net = agent.online

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

    R = 7 if cfg.dilated_backbone else 3
    feats, ys, dps, dbs, no_hist, seed_ids = [], [], [], [], [], []
    for si, seed in enumerate(seeds):
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
                out = net(torch.from_numpy(s).unsqueeze(0), compute_aux=False)
            action = int(np.argmax(np.where(mask, out["advantage"][0].numpy(), -np.inf)))
            front_idx = [a for a in range(256) if mask[a]]
            if action < 256 and mask[action] and len(front_idx) > 1 \
                    and belief.prev_boundary_cell is not None:
                vx = belief.cell[0] - belief.prev_boundary_cell[0]
                vy = belief.cell[1] - belief.prev_boundary_cell[1]
                if vx != 0 or vy != 0:
                    vn = math.hypot(vx, vy)
                    feat = out["features"][0].numpy()          # (64,16,16)
                    trail = s[CH_TRAIL_PREV_OPTION]
                    px, py = belief.cell
                    bx, by = belief.prev_boundary_cell
                    ok = True
                    for a in front_idx:
                        cx, cy = index_cell(a, n)
                        dx, dy = cx - px, cy - py
                        dn = math.hypot(dx, dy)
                        if dn == 0:
                            ok = False
                            break
                    if ok:                                     # 掩码已排除自身格，防御
                        dp = np.array([max(abs(index_cell(a, n)[0] - px),
                                           abs(index_cell(a, n)[1] - py)) for a in front_idx])
                        db = np.array([max(abs(index_cell(a, n)[0] - bx),
                                           abs(index_cell(a, n)[1] - by)) for a in front_idx])
                        ali = np.array([( (index_cell(a, n)[0] - px) * vx +
                                          (index_cell(a, n)[1] - py) * vy) /
                                        (math.hypot(index_cell(a, n)[0] - px,
                                                    index_cell(a, n)[1] - py) * vn)
                                        for a in front_idx])
                        for i, a in enumerate(front_idx):
                            cx, cy = index_cell(a, n)
                            wx = trail[max(0, cx - R):cx + R + 1, max(0, cy - R):cy + R + 1]
                            feats.append(feat[:, cx, cy])
                            ys.append(ali[i])
                            dps.append(dp[i])
                            dbs.append(db[i])
                            no_hist.append(bool(dp[i] > R and db[i] > R and not wx.any()))
                            seed_ids.append(si)
            result = executor.run(env, action)
            belief.commit_option(result.path, result.start_cell)
            if result.steps == 0:
                abort_guard += 1
                if abort_guard >= 8:
                    break
            else:
                abort_guard = 0
            if result.terminated or result.truncated:
                break
        env.close()
    return (np.array(feats, dtype=np.float64), np.array(ys, dtype=np.float64),
            np.array(dps), np.array(dbs), np.array(no_hist, dtype=bool),
            np.array(seed_ids))


def main() -> None:
    ap = argparse.ArgumentParser(description="表征探针 f(c)→a(c)（L2）")
    ap.add_argument("--run-dir", required=True, help="algo/runs/<exp>（含 config.json）")
    ap.add_argument("--checkpoint", default="checkpoint_stage1_final.pt")
    ap.add_argument("--seeds", type=int, default=30, help="held-out seed 数（cfg.seed+777 派生）")
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(os.path.join(args.run_dir, "config.json"))
    seeds = [int(s) for s in np.random.default_rng(cfg.seed + 777).integers(
        0, 2 ** 31 - 1, args.seeds)]
    X, y, dp, db, no_hist, sid = collect(
        cfg, os.path.join(args.run_dir, args.checkpoint), seeds)
    if len(y) < 100:
        raise SystemExit(f"样本过少（n={len(y)}），检查 use_traj/对齐前提")

    n_train_seeds = max(1, int(round(0.7 * len(seeds))))
    tr = sid < n_train_seeds
    te = ~tr
    mu, sd, w = ridge_fit(X[tr], y[tr])
    y_perm = y[tr].copy()
    np.random.default_rng(0).shuffle(y_perm)
    _, _, w_perm = ridge_fit(X[tr], y_perm)   # mu/sd 只依赖 X[tr]，与 w 的相同，直接复用

    r = 7 if cfg.dilated_backbone else 3
    both = (dp <= r) & (db <= r)
    p_only = (dp <= r) & (db > r)
    b_only = (dp > r) & (db <= r)

    def pct(mask):
        return f"{mask.mean():.1%}"

    print(f"\n[表征探针] run={args.run_dir}  R={r}  样本 n={len(y)}"
          f"（train seeds={n_train_seeds} / test seeds={len(seeds) - n_train_seeds}）")
    print(f"  桶占比: both_in {pct(both)}  p_only {pct(p_only)}  "
          f"b_only {pct(b_only)}  arch_zero(双锚点窗外且无走廊像素) {pct(no_hist)}")
    print(f"  Test R² overall        : {r2(X[te], y[te], mu, sd, w):+.3f}")
    for name, m in (("both_in", both), ("p_only", p_only),
                    ("b_only", b_only), ("arch_zero", no_hist)):
        sel = te & m
        if sel.sum() >= 30:
            print(f"  Test R² {name:<10}  : {r2(X[sel], y[sel], mu, sd, w):+.3f}（n={int(sel.sum())}）")
        else:
            print(f"  Test R² {name:<10}  : —（n={int(sel.sum())} 样本不足）")
    print(f"  置换对照 R²（应 ≤0）    : {r2(X[te], y[te], mu, sd, w_perm):+.3f}")


if __name__ == "__main__":
    main()
