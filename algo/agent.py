# -*- coding: utf-8 -*-
"""SMDP Dueling DQN 智能体：ε-greedy + Double-DQN 学习步（γ^τ）+ 辅助补墙任务。

学习件（rl_algorithm_design.md §4.5/§6.1 定稿）：
    Double-DQN：在线网在 s_after 合法集上选动作，target 网估值（确定性环境
    主要防过估）；
    SMDP 贝尔曼：target = Σr + γ^τ · (1−terminated) · Q_target(s₂, a*)；
    truncated（超时截断）自举、terminated 不自举（time-limit aware）；
    损失 = TD 均方误差 + λ · 辅助 BCE（共享主干两路反传），梯度裁剪；
    target 网每 target_sync_options 个 option 硬同步一次。

复现性（§6.1）：ε 采样与 replay 采样走独立 np.random.Generator（构造时由
master seed 派生）；网络初始化走 torch seed。同种子两次训练曲线一致
（CPU 严格一致；GPU 允许非严格误差，需声明）。

R1 边界：本类只消费信念张量 + 合法掩码 + 奖励标量；辅助 GT 由 trainer 从
环境侧特权通道传入 learn()（与 reward 同通道，绝不进入 act() 的策略输入）。
"""
import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .actions import N_OPTIONS
from .network import FCQNet, dueling_q


def gt_directional_walls(gt_south, gt_west) -> "torch.Tensor":
    """规范形式 GT 墙线 (B,2,n,n) → 4 方向 GT 墙值 (B,4,n,n) float（0/1）。

    接受 numpy 数组或 torch 张量（2026-11-20 设备无关化：输入已在 GPU 时全程
    GPU 切片赋值，免去 CPU 中转 + H2D 传输；纯 0/1 重排，数值逐位一致）。
    布局 [B,C,x,y] 与 belief.tensor 一致；N/E 外边框恒 1（有墙，封闭场地先验）。
    """
    south = torch.as_tensor(gt_south, dtype=torch.float32)
    west = torch.as_tensor(gt_west, dtype=torch.float32)
    south = south.squeeze(1) if south.dim() == 4 else south
    west = west.squeeze(1) if west.dim() == 4 else west
    b, _, n = south.shape
    out = torch.zeros((b, 4, n, n), dtype=torch.float32, device=south.device)
    out[:, 0, :, :-1] = south[:, :, 1:]        # wall_N of (x,y) = south[x][y+1]
    out[:, 0, :, -1] = 1.0                     # 北外边框
    out[:, 1, :-1, :] = west[:, 1:, :]         # wall_E of (x,y) = west[x+1][y]
    out[:, 1, -1, :] = 1.0                     # 东外边框
    out[:, 2] = south                          # wall_S = south[x][y]
    out[:, 3] = west                           # wall_W = west[x][y]
    return out


class SMDPDQNAgent:
    """ε-greedy 高层策略 + Double-DQN 学习 + 辅助任务。"""

    def __init__(self, in_channels: int = 16, gamma: float = 0.99, lr: float = 1e-4,
                 batch_size: int = 64, target_sync_options: int = 1000,
                 warmup_options: int = 2000, lambda_aux: float = 0.5,
                 grad_clip: float = 10.0, eps_start: float = 0.10, eps_end: float = 0.02,
                 eps_anneal_options: int = 20000, capacity: int = 100_000,
                 n: int = 16, aux_head: bool = True, use_gru: bool = False,
                 device: Optional[str] = None, seed: int = 0):
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(int(seed))
        self.online = FCQNet(in_channels=in_channels, aux_head=aux_head, use_gru=use_gru).to(self.device)
        self.target = FCQNet(in_channels=in_channels, aux_head=aux_head, use_gru=use_gru).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=float(lr))
        # 复现流：ε 采样 / replay 采样独立（master seed 派生，互不污染）
        self._eps_rng = np.random.default_rng(int(seed) + 1)
        self._sample_rng = np.random.default_rng(int(seed) + 2)
        # 超参（§4.5 默认表）
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_sync_options = int(target_sync_options)
        self.warmup_options = int(warmup_options)
        self.lambda_aux = float(lambda_aux)
        self.grad_clip = float(grad_clip)
        self.eps_start, self.eps_end = float(eps_start), float(eps_end)
        self.eps_anneal_options = max(1, int(eps_anneal_options))
        self.eps_override: Optional[float] = None     # 课程阶段切换时 ε 从低值重启
        # 经验池与计数
        from .replay import ReplayBuffer
        self.buffer = ReplayBuffer(capacity, n=n)
        self.options_seen = 0
        self.learn_steps = 0
        self._last_target_sync = 0
        # γ 标量张量缓存（学习步逐次重建 torch.tensor(gamma, device) 的 H2D 开销）
        self._gamma_t = torch.tensor(float(gamma), device=self.device)
        # 最近一次前向的 Q 统计（监控用，零额外成本；ε-random 步为 None）
        self.last_q_stats: Optional[Dict[str, float]] = None

    # ---------- ε 调度 ----------
    def epsilon(self, options_seen: Optional[int] = None) -> float:
        if self.eps_override is not None:
            return float(self.eps_override)
        k = self.options_seen if options_seen is None else int(options_seen)
        frac = min(1.0, k / self.eps_anneal_options)
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    # ---------- 动作选择 ----------
    @torch.no_grad()
    def act(self, tensor: np.ndarray, mask: np.ndarray, greedy: bool = False,
            ) -> int:
        """ε-greedy 在合法选项上选择（非法动作不参加比较，§4.2）。

        tensor: (16,n,n) uint8（或 14 通道）；mask: (259,) bool（至少一个 True）。
        """
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            raise RuntimeError("合法选项集为空，无法选择动作")
        eps = self.epsilon()
        self.last_q_stats = None                 # ε-random 分支无前向，不留旧值
        if not greedy and self._eps_rng.random() < eps:
            return int(legal[self._eps_rng.integers(legal.size)])
        t = torch.as_tensor(tensor, device=self.device).unsqueeze(0)
        m = torch.as_tensor(mask, device=self.device).unsqueeze(0)
        out = self.online(t, compute_aux=False)
        q = dueling_q(out["advantage"], out["value"], m)          # (1,259)，非法=−inf
        q_legal = q[0, torch.as_tensor(legal, device=self.device)]
        self.last_q_stats = {"q_mean": float(q_legal.mean().item()),
                             "q_max": float(q.max().item())}
        return int(q.argmax(dim=1).item())

    # ---------- 经验与学习 ----------
    def observe(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, tau: int,
                terminal: bool, mask_b: np.ndarray, mask_a2: np.ndarray,
                gt_south: np.ndarray, gt_west: np.ndarray) -> None:
        """登记一条 option 级转移（含该转移的合法掩码与辅助 GT）。"""
        self.buffer.push(s, a, r, s2, tau, terminal, mask_b, mask_a2, gt_south, gt_west)
        self.options_seen += 1

    def maybe_learn(self) -> Optional[Dict[str, float]]:
        """warmup 期满后每 option 学习一步；返回损失统计（或 None）。"""
        if self.options_seen < self.warmup_options:
            return None
        batch = self.buffer.sample(self.batch_size, self._sample_rng)
        if batch is None:
            return None
        return self._learn_step(batch)

    def maybe_sync_target(self) -> bool:
        """每 target_sync_options 个 option 硬同步 target 网。"""
        if self.options_seen - self._last_target_sync >= self.target_sync_options:
            self.sync_target()
            return True
        return False

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())
        self._last_target_sync = self.options_seen

    def _learn_step(self, batch: dict) -> Dict[str, float]:
        dev = self.device
        s = torch.as_tensor(batch["s"], device=dev)
        s2 = torch.as_tensor(batch["s2"], device=dev)
        a = torch.as_tensor(batch["a"], device=dev)
        r = torch.as_tensor(batch["r"], device=dev)
        tau = torch.as_tensor(batch["tau"], device=dev).float()
        terminal = torch.as_tensor(batch["terminal"], device=dev)
        mask_b = torch.as_tensor(batch["mask_b"], device=dev)
        mask_a2 = torch.as_tensor(batch["mask_a2"], device=dev)

        # TD：Q_online(s, a) vs Double-DQN target
        out = self.online(s, compute_aux=self.lambda_aux > 0)
        q = dueling_q(out["advantage"], out["value"], mask_b)
        q_sa = q.gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            out2 = self.online(s2, compute_aux=False)
            q2_online = dueling_q(out2["advantage"], out2["value"], mask_a2)
            a2 = q2_online.argmax(dim=1, keepdim=True)            # 在线网选动作（合法集内）
            out2_t = self.target(s2, compute_aux=False)
            q2_target = dueling_q(out2_t["advantage"], out2_t["value"], mask_a2)
            q_next = q2_target.gather(1, a2).squeeze(1)
            discount = torch.pow(self._gamma_t, tau)              # SMDP γ^τ（缓存标量张量）
            y = r + discount * (~terminal).float() * q_next
        td_loss = nn.functional.mse_loss(q_sa, y)

        # 辅助任务：补全未知墙位（BCE 只在输入 unknown 槽位；共享主干两路反传）。
        # 2026-11-20 无同步化改造：掩码均值 = sum(bce·mask)/cnt（cnt.clamp(min=1)），
        # 全 GPU 完成取代 unknown.any() 的中途 D2H 同步；cnt=0 时 bce=0，与原
        # "跳过"语义等值（常数 0 无梯度）。GT 换算改为 GPU 张量直入（设备无关
        # 版 gt_directional_walls），免去 CPU 中转 + (B,4,n,n) H2D 传输。
        loss = td_loss
        bce_t = torch.zeros_like(td_loss)
        if self.lambda_aux > 0 and "aux_logits" in out:
            aux_logits = out["aux_logits"]                        # (B,4,n,n)
            unknown = (s[:, :4] == 0)                             # (B,4,n,n) 逐方向未知掩码
            gt = gt_directional_walls(torch.as_tensor(batch["gt_south"], device=dev),
                                      torch.as_tensor(batch["gt_west"], device=dev))
            bce_all = nn.functional.binary_cross_entropy_with_logits(
                aux_logits, gt, reduction="none")
            bce_t = (bce_all * unknown).sum() / unknown.sum().clamp(min=1)
            loss = td_loss + self.lambda_aux * bce_t

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optimizer.step()
        self.learn_steps += 1
        # 单次 D2H：三个损失一次 .tolist()（原 td/total 各一次 .float() 同步）
        td_v, aux_v, tot_v = torch.stack(
            [td_loss.detach(), bce_t.detach(), loss.detach()]).tolist()
        return {"td_loss": td_v, "aux_loss": aux_v, "total_loss": tot_v}

    # ---------- 课程切换（§6.2 阶段切换机制） ----------
    def begin_stage1(self, eps_restart: float = 0.05) -> None:
        """阶段 0 → 阶段 1：replay 清空（trainer 负责）+ target 重置 + ε 低值重启。"""
        self.buffer.clear()
        self.sync_target()
        self.eps_override = float(eps_restart)

    def set_lr(self, lr: float) -> None:
        """可选：阶段切换时主干判别式降 lr（§6.2）。"""
        for g in self.optimizer.param_groups:
            g["lr"] = float(lr)

    # ---------- checkpoint ----------
    def state_dict(self) -> dict:
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "options_seen": self.options_seen,
            "learn_steps": self.learn_steps,
            "last_target_sync": self._last_target_sync,
            "eps_override": self.eps_override,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.online.load_state_dict(sd["online"])
        self.target.load_state_dict(sd["target"])
        self.optimizer.load_state_dict(sd["optimizer"])
        self.options_seen = int(sd["options_seen"])
        self.learn_steps = int(sd["learn_steps"])
        self._last_target_sync = int(sd["last_target_sync"])
        self.eps_override = sd["eps_override"]
