# -*- coding: utf-8 -*-
"""BAMER RL 算法层（阶段三）——SMDP Dueling DQN + 三分支 FCQN + 16 通道信念。

模块地图（docs/rl_algorithm_design.md §8 定稿）：
    belief    探测账本 + 轨迹账本 + 16 通道信念派生（R1 自洽）
    actions   259 高层选项 + 动态合法性掩码
    executor  (格,朝向) 1024 态 BFS 执行器（每步重规划 + 不可达中止）
    network   三分支 dueling 网络 + 辅助补墙头（use_gru 预留）
    replay    option 级 uint8 经验池
    agent     ε-greedy + Double-DQN（γ^τ）+ 辅助任务
    trainer   训练循环 / checkpoint / 三臂课程（A 搜索预训练 / B 熵预训练 / C 基线）
    eval      固定 seed 套件 + 全知 BFS 参照 + 指标

单向依赖：algo → bamer_env / bamer_mouse（环境与传感器层从不反向 import algo）。
R1 分工：belief/executor/agent 只消费算法侧合法通道；trainer/eval 为环境侧
特权角色（奖励簿记、辅助 GT、参照计算），特权信息绝不进入策略输入。
"""
from .actions import IDX_DELIVER, IDX_PICK, N_OPTIONS, legal_mask, option_label, option_target
from .agent import SMDPDQNAgent
from .belief import BeliefState, GTAssetClassifier, N_CHANNELS, SensingLedger
from .executor import Executor, OptionResult, bfs_path
from .network import FCQNet, dueling_q
from .replay import ReplayBuffer
from .trainer import TrainConfig, Trainer, stage_plan

__all__ = [
    "BeliefState", "GTAssetClassifier", "SensingLedger", "N_CHANNELS",
    "N_OPTIONS", "IDX_PICK", "IDX_DELIVER", "legal_mask", "option_label", "option_target",
    "Executor", "OptionResult", "bfs_path",
    "FCQNet", "dueling_q", "ReplayBuffer", "SMDPDQNAgent",
    "TrainConfig", "Trainer", "stage_plan",
]
