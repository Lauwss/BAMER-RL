# -*- coding: utf-8 -*-
"""FCQN 三分支 Dueling 网络（rl_algorithm_design.md §4 定稿实现）。

架构（§4.1）：
    输入 (B, 16, 16, 16) uint8 → float /2 归一化（类别小整数 0/1/2 → [0,1]）
    共享卷积主干：3 层 3×3 conv，s1 p1，通道 32/64/64，ReLU（空间恒 16×16）
    ├──【Explore 分支】1×1 conv → A_explore (1,16,16)：逐格优势（论文 score map 分支）
    ├──【池化分支】GAP → 向量 → 线性头 A_pick(2) / A_deliver(1) / V(s)(1)
    │                 （论文 terminal action 池化分支，1 输出推广为 3）
    └──【辅助头，仅训练】1×1 conv → 墙位补全 logits (4,16,16)
            BCE 只在输入 unknown 槽位计算；GT 仅训练期监督（测试时头丢弃）

合并与选择（§4.2）：259 个优势值拼成联合向量，dueling 组合
    Q(s,o) = V(s) + A(s,o) − mean_legal(A)
mean_legal 只对合法动作取均值（非法位置若先进均值会被 −∞ 污染）；选择
ε-greedy / argmax 在合法动作上进行（agent.py）。

开关（§4.4/§4.5）：
    use_gru：GRU 升级路径预留，本期 False（True 时显式 NotImplementedError）；
    in_channels=14：use_traj=False 消融（裁剪轨迹双通道，belief.tensor 同步裁剪）。
"""
from typing import Dict, Optional

import torch
import torch.nn as nn


class FCQNet(nn.Module):
    """三分支 FCN：主干 + Explore score map + Pick/Deliver/Value 池化头 + 辅助补墙头。"""

    def __init__(self, in_channels: int = 16, aux_head: bool = True, use_gru: bool = False):
        super().__init__()
        if use_gru:
            raise NotImplementedError(
                "use_gru=True 为预留升级路径（§4.4：GRU 插在主干与头之间、时间步=option 边界、"
                "序列化 replay + 段起点隐状态），本期前馈版未实现")
        self.in_channels = int(in_channels)
        self.aux_head = bool(aux_head)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
        )
        self.explore_head = nn.Conv2d(64, 1, kernel_size=1)          # A_explore (1,16,16)
        self.pick_head = nn.Linear(64, 2)                            # A_pick (2)
        self.deliver_head = nn.Linear(64, 1)                         # A_deliver (1)
        self.value_head = nn.Linear(64, 1)                           # V(s) (1)
        if self.aux_head:
            self.wall_completion_head = nn.Conv2d(64, 4, kernel_size=1)  # logits (4,16,16)

    def forward(self, x_uint8: torch.Tensor, compute_aux: bool = True) -> Dict[str, torch.Tensor]:
        """前向。x_uint8: (B, C, 16, 16) uint8/float；返回 advantage (B,259) 等。

        advantage 布局 = [explore 256 (格序 y*16+x，与 actions.explore_index 同构),
                          pick 2 (quyuan, zhaojun), deliver 1]。
        """
        x = x_uint8.float() / 2.0
        feat = self.backbone(x)                                     # (B,64,16,16)
        a_explore = self.explore_head(feat).flatten(1)              # (B,256)
        pooled = feat.mean(dim=(2, 3))                              # GAP → (B,64)
        a_pick = self.pick_head(pooled)                             # (B,2)
        a_deliver = self.deliver_head(pooled)                       # (B,1)
        value = self.value_head(pooled)                             # (B,1)
        advantage = torch.cat([a_explore, a_pick, a_deliver], dim=1)  # (B,259)
        out = {"advantage": advantage, "value": value, "features": feat}
        if self.aux_head and compute_aux:
            out["aux_logits"] = self.wall_completion_head(feat)     # (B,4,16,16)
        return out


def dueling_q(advantage: torch.Tensor, value: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
    """Dueling 组合 + 动态合法掩码（§4.2）。

    advantage (B,259)，value (B,1)，mask (B,259) bool。
    mean_legal = 只对合法动作取均值（先在合法集合上算均值 → 再算各合法 Q →
    非法动作不参加比较）；非法位置 Q 置 −inf（不参加 argmax / 损失）。
    """
    adv = advantage.masked_fill(~mask, 0.0)
    mean_legal = adv.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    q = value + advantage - mean_legal.unsqueeze(1)
    return q.masked_fill(~mask, float("-inf"))
