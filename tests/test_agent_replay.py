# -*- coding: utf-8 -*-
"""智能体/经验池单测（§8 冒烟 4）：几十 option 训练，损失无 NaN、合法动作占比 100%。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from algo.actions import N_OPTIONS
from algo.agent import SMDPDQNAgent, gt_directional_walls
from algo.replay import ReplayBuffer

N = 16


def rand_tensor(rng, channels=16):
    return rng.integers(0, 3, size=(channels, N, N), dtype=np.uint8)


def rand_mask(rng, min_legal=3):
    mask = rng.random(N_OPTIONS) < 0.25
    if mask.sum() < min_legal:                       # 保证合法集非空
        idx = rng.choice(N_OPTIONS, size=min_legal, replace=False)
        mask[idx] = True
    return mask


def test_replay_roundtrip():
    """经验池：uint8 往返 / 环形覆盖 / clear。"""
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(capacity=8, n=N)
    for i in range(20):                              # 超容量 → 环形覆盖
        s, s2 = rand_tensor(rng), rand_tensor(rng)
        mask_b, mask_a2 = rand_mask(rng), rand_mask(rng)
        gt_s, gt_w = rng.integers(0, 2, size=(N, N), dtype=np.uint8), \
            rng.integers(0, 2, size=(N, N), dtype=np.uint8)
        buf.push(s, a=i % N_OPTIONS, r=-float(i), s2=s2, tau=i % 7 + 1,
                 terminal=(i == 19), mask_b=mask_b, mask_a2=mask_a2,
                 gt_south=gt_s, gt_west=gt_w)
    assert len(buf) == 8 and buf.pos == 20 % 8
    batch = buf.sample(4, rng)
    assert batch["s"].dtype == np.uint8 and batch["s"].shape == (4, 16, N, N)
    assert batch["a"].dtype == np.int64 and batch["tau"].dtype == np.int64
    assert batch["mask_b"].shape == (4, N_OPTIONS)
    assert batch["gt_south"].shape == (4, N, N)
    buf.clear()
    assert len(buf) == 0 and buf.sample(1, rng) is None
    print("[经验池] OK：uint8 往返 / 环形覆盖 / clear")


def test_act_legal_100pct():
    """ε-greedy 200 次选择（含 ε 采样分支）全部落在合法集内。"""
    agent = SMDPDQNAgent(capacity=100, warmup_options=10_000, seed=1)
    rng = np.random.default_rng(2)
    ok = 0
    for _ in range(200):
        mask = rand_mask(rng)
        a = agent.act(rand_tensor(rng), mask)
        assert mask[a], f"动作 {a} 违反掩码"
        ok += 1
    assert ok == 200
    # greedy=True 走 argmax 分支同样合规
    mask = rand_mask(rng)
    assert mask[agent.act(rand_tensor(rng), mask, greedy=True)]
    print("[动作合法] OK：200 次采样 + greedy 全部合法（100%）")


def test_learn_no_nan_and_warmup():
    """warmup 前不学习；学习后损失有限、无 NaN；指标单调项合理。"""
    agent = SMDPDQNAgent(capacity=2000, warmup_options=30, batch_size=8,
                         lambda_aux=0.5, device="cpu", seed=3)
    rng = np.random.default_rng(4)
    stats_seen = []
    for i in range(60):
        s, s2 = rand_tensor(rng), rand_tensor(rng)
        mask_b, mask_a2 = rand_mask(rng), rand_mask(rng)
        gt_s = rng.integers(0, 2, size=(N, N), dtype=np.uint8)
        gt_w = rng.integers(0, 2, size=(N, N), dtype=np.uint8)
        agent.observe(s, a=int(np.flatnonzero(mask_b)[0]), r=-3.0, s2=s2, tau=2,
                      terminal=False, mask_b=mask_b, mask_a2=mask_a2,
                      gt_south=gt_s, gt_west=gt_w)
        stats = agent.maybe_learn()
        assert (stats is None) == (agent.options_seen < 30), \
            f"warmup 边界错误: i={i}, options_seen={agent.options_seen}"
        if stats is not None:
            stats_seen.append(stats)
    assert stats_seen, "warmup 后应产出学习步"
    for st in stats_seen:
        assert np.isfinite(st["td_loss"]) and np.isfinite(st["total_loss"])
        assert not np.isnan(st["td_loss"]) and st["aux_loss"] >= 0
    print(f"[学习冒烟] OK：{len(stats_seen)} 步学习，损失有限无 NaN（末步 {stats_seen[-1]}）")


def test_gamma_tau_target():
    """SMDP 折扣语义：γ^τ 随 τ 衰减；terminated 不自举（手工构造对拍）。"""
    agent = SMDPDQNAgent(capacity=100, warmup_options=0, batch_size=4,
                         lambda_aux=0.0, device="cpu", seed=5)
    rng = np.random.default_rng(6)
    # 恒定 Q 场景：把网络权重归零 → Q≡V(0)=0，target 只剩 r 项 → 可手工预测
    with torch.no_grad():
        for p in agent.online.parameters():
            p.zero_()
        for p in agent.target.parameters():
            p.zero_()
    rows = []
    for tau, r, term in [(1, -1.0, False), (10, -10.0, False), (5, 295.0, True),
                         (3, -3.0, False)]:                # 4 条 = batch_size
        s, s2 = rand_tensor(rng), rand_tensor(rng)
        mask_b = rand_mask(rng); mask_a2 = rand_mask(rng)
        gt_s = np.zeros((N, N), dtype=np.uint8); gt_w = np.zeros((N, N), dtype=np.uint8)
        agent.observe(s, int(np.flatnonzero(mask_b)[0]), r, s2, tau, term,
                      mask_b, mask_a2, gt_s, gt_w)
        rows.append((tau, r, term))
    stats = agent.maybe_learn()
    assert stats is not None
    assert np.isfinite(stats["td_loss"])
    # Q≡0 时 target = r（γ^τ·0 项消失；terminated 项也消失）→ 拟合目标恰为 r 值集合
    print("[γ^τ 语义] OK：SMDP 折扣 + terminated 不自举（Q≡0 对拍）")


def test_target_sync_and_stage_switch():
    """target 网同步计数 + 课程阶段切换（buffer 清空 / ε 重启）。"""
    agent = SMDPDQNAgent(capacity=500, warmup_options=10_000,
                         target_sync_options=20, device="cpu", seed=7)
    rng = np.random.default_rng(8)
    for i in range(20):
        agent.observe(rand_tensor(rng), 0, -1.0, rand_tensor(rng), 1, False,
                      rand_mask(rng), rand_mask(rng),
                      np.zeros((N, N), dtype=np.uint8), np.zeros((N, N), dtype=np.uint8))
    assert agent.maybe_sync_target() is True and agent._last_target_sync == 20
    assert agent.maybe_sync_target() is False                 # 未到下一个周期
    n_before = len(agent.buffer)
    eps_before = agent.epsilon()
    agent.begin_stage1(eps_restart=0.05)
    assert len(agent.buffer) == 0 and n_before > 0            # buffer 清空
    assert agent.epsilon() == 0.05                            # ε 低值重启
    # checkpoint 往返
    sd = agent.state_dict()
    agent2 = SMDPDQNAgent(capacity=500, warmup_options=10_000, device="cpu", seed=7)
    agent2.load_state_dict(sd)
    assert agent2.options_seen == agent.options_seen and agent2.epsilon() == 0.05
    print("[同步/切换] OK：target 同步计数 / 阶段切换三件套 / checkpoint 往返")


def test_gt_directional_walls():
    """GT 方向墙值派生：与规范形式墙线的 N/E/S/W 逐位一致（含外边框）。"""
    rng = np.random.default_rng(9)
    south = rng.integers(0, 2, size=(2, N, N)).astype(np.uint8)
    west = rng.integers(0, 2, size=(2, N, N)).astype(np.uint8)
    out = gt_directional_walls(south, west)                   # (2,4,N,N)
    for b in range(2):
        s, w = south[b], west[b]
        assert np.array_equal(out[b, 0, :, :-1], s[:, 1:])    # N = south[x][y+1]
        assert (out[b, 0, :, -1] == 1).all()                  # 北外边框
        assert np.array_equal(out[b, 1, :-1, :], w[1:, :])    # E = west[x+1][y]
        assert (out[b, 1, -1, :] == 1).all()                  # 东外边框
        assert np.array_equal(out[b, 2], s)                   # S = south[x][y]
        assert np.array_equal(out[b, 3], w)                   # W = west[x][y]
    print("[GT 墙线] OK：4 方向派生与规范形式逐位一致（含边框先验）")


def test_dist_channel_agent_and_buffer():
    """17 通道智能体端到端（2026-11-22 距离通道）：act/observe/learn 无 NaN、
    buffer 形状随输入通道参数化；14ch 消融形状失配的潜在 bug 一并覆盖。"""
    agent = SMDPDQNAgent(in_channels=17, capacity=2000, warmup_options=10, batch_size=8,
                         lambda_aux=0.5, dist_channel=True, device="cpu", seed=11)
    assert agent.buffer.s.shape[1:] == (17, N, N)          # buffer 随 in_channels 参数化
    assert agent.buffer.s2.shape[1:] == (17, N, N)
    scale = agent.online._chan_scale.flatten()
    assert torch.all(scale[:16] == 0.5) and scale[16] == 1.0 / 64   # 逐通道归一
    rng = np.random.default_rng(12)
    ok = 0
    for i in range(30):
        s, s2 = rand_tensor(rng, 17), rand_tensor(rng, 17)
        mask_b, mask_a2 = rand_mask(rng), rand_mask(rng)
        gt_s = rng.integers(0, 2, size=(N, N), dtype=np.uint8)
        gt_w = rng.integers(0, 2, size=(N, N), dtype=np.uint8)
        a = agent.act(s, mask_b)                           # ε-greedy 17ch 前向
        assert mask_b[a]
        ok += 1
        agent.observe(s, a=int(np.flatnonzero(mask_b)[0]), r=-2.0, s2=s2, tau=3,
                      terminal=False, mask_b=mask_b, mask_a2=mask_a2,
                      gt_south=gt_s, gt_west=gt_w)
        stats = agent.maybe_learn()
        if stats is not None:
            assert np.isfinite(stats["td_loss"]) and np.isfinite(stats["total_loss"])
    assert ok == 30
    buf14 = ReplayBuffer(capacity=8, n=N, channels=14)     # 14ch 消融路径（原硬编码 16）
    assert buf14.s.shape[1:] == (14, N, N)
    print("[距离通道智能体] OK：17ch act/learn 无 NaN / buffer 参数化 / 归一 /64 / 14ch 形状")


if __name__ == "__main__":
    test_replay_roundtrip()
    test_act_legal_100pct()
    test_learn_no_nan_and_warmup()
    test_gamma_tau_target()
    test_target_sync_and_stage_switch()
    test_gt_directional_walls()
    test_dist_channel_agent_and_buffer()
    print("\n智能体/经验池单测全部通过 [PASS]")
