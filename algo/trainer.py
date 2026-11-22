# -*- coding: utf-8 -*-
"""训练循环 / checkpoint / 种子管理 / 三臂课程（rl_algorithm_design.md §4.5/§6 定稿）。

三臂消融（§6.2 训练课程）：
    A 搜索预训练（主臂）  阶段 0：−1/步 + 识别 +100 + 终点发现 +100（无送达 +300），
                          terminated = 4 POI 定位（env curriculum="search"）→ 阶段 1 全任务微调
    B 熵预训练            阶段 0：−1/步 + 信息增益（未知墙位数，精确计数）→ 阶段 1 全任务微调
    C 基线（原型）        无阶段 0，从头直接训全任务

阶段切换机制（A/B → 阶段 1）：replay buffer 清空（阶段 0 transition 奖励语义失效，
混用毒化 target）+ target 网重置 + ε 从低值重启（0.05）+ 可选降 lr。

奖励实现（D8，contract_v1.md §6）：-1/步（每个低层动作）+ 里程碑事件（来源
info["events"]）：首次识别 +100/台（ImageFeed first）+ 终点发现 +100/个
（DestinationFeed first）+ 送达 +300/台（Arrived）；碰撞/超时无额外惩罚。
本模块为环境侧特权角色（与 reward 同通道）：奖励簿记消费事件字段、辅助头 GT
读取 env.scene.maze 墙线——均只作用于训练监督，绝不进入 act() 的策略输入（R1）。

复现性（§6.1）：master seed → 环境种子流（np_random → 迷宫/布点/雷达派生）+
agent 内部流（ε / replay 采样）+ torch 初始化，全部显式登记。同种子两次训练
曲线一致（CPU 严格；GPU 允许非严格误差，需声明）。

性能注记：账本更新 walk_cells 为逐束 Python 循环（180 束 × ~0.5 ms/步），
训练全程外推分钟级，原型阶段可接受；若成热点，按 bench 思路向量化（不引入
新依赖）。
"""
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from bamer_env.env import BamerEnv
from bamer_mouse.lidar import LidarSpec

from .actions import legal_mask
from .agent import SMDPDQNAgent
from .belief import BeliefState
from .executor import Executor
from .monitor import build_monitor

# D8 里程碑（contract_v1.md §6 定稿数值）
REWARD_FIRST_IDENTIFY = 100.0
REWARD_FIRST_DESTINATION = 100.0
REWARD_FIRST_DELIVER = 300.0


@dataclass
class TrainConfig:
    """一次实验一个配置（algo/configs/*.yaml；字段名即 yaml 键）。"""
    # --- 课程（§6.2 三臂消融） ---
    arm: str = "C"                    # "A" 搜索预训练 / "B" 熵预训练 / "C" 从头直训
    stage0_options: int = 40_000      # 阶段 0 option 预算（探索指标收敛即停，宁短勿长）
    stage1_options: int = 120_000     # 阶段 1（或臂 C 全程）option 预算
    stage1_eps_restart: float = 0.05  # 阶段切换 ε 重启值
    stage1_lr: Optional[float] = None  # 可选：阶段切换降 lr（None = 保持）
    max_episodes_per_stage: int = 100000   # 保险丝
    # --- 环境 ---
    env_randomize: bool = True        # 阶段 1 主力（防实例记忆 + 辅助头 GT 合法性前提）
    env_max_steps: int = 1000
    lidar_n_beams: int = 180          # D4 默认；训练可降束提速（登记变更）
    lidar_max_range: float = 2.0
    maze_n: int = 16
    # 随机迷宫拓扑（braided 赛图匹配，2026-11-20）：验收图 215 可达格 37 环、
    # 死胡同仅 8 个（环/格 0.172），而生成器死胡同开洞上限只有 ~0.10 环/格，
    # 故 stage1 用 braiding+补环双机制。stage0 低环练死胡同回溯与系统覆盖。
    # None = 不补环（旧行为）。三臂须同分布，否则探索指标不可对比。
    maze_braiding_stage0: float = 0.0
    maze_braiding_stage1: float = 0.8
    maze_loops_stage0: Optional[List[int]] = None   # [lo, hi] 目标环数均匀采样
    maze_loops_stage1: Optional[List[int]] = None
    # 固定迷宫诊断（2026-11-21，GPT 评审"固定环境 A/B"）：None = 每集随机迷宫
    # （现行）。"generated" = 固定一张随机生成的 braiding-stage1 迷宫（seed 派生，
    # 补环至官方密度 37）；"official" = 固定官方 PPT 赛图（Maze.default()）。
    # 两 stage 共用同一张（课程只剩奖励切换）；机器布点每集随机、终点沿用固定
    # 默认位置；braiding/loops 参数仅用于生成该固定实例。
    maze_fixed: Optional[str] = None
    # frontier 引导（2026-11-21，探索塌缩修复§改动1）：legal_mask 的 Explore
    # 目标收窄为信念前沿格（扫描后信息量必严格增加的格子）。零信息目标被排除
    # 出候选集 → 贪心微步环结构性消除；ε 采样池同步收窄（与§改动2合并实现，
    # 两者在掩码实现下等价）。frontier 为空（可达区域完全观测）回退全可达集。
    frontier_mask: bool = False
    # fallback_unvisited（2026-11-22 盲区修复，评估侧 A/B 见 CHANGELOG 2/3 上限定性）：
    # frontier 空的回退集从全可达收窄为未到访可达格——信息闭合后剩余唯一任务信号
    # 就是踩格（识别须进格）；默认 False = 旧行为逐位不变。训练期开启为候选训练侧
    # 修复（让 ε 采到 ImageFeed→Pickup 转移），本 run 为训练侧消融（用户指令）。
    fallback_unvisited: bool = False
    # --- 感受野修复（2026-11-22 GPT 外部批评核实，均默认关 = 旧行为） ---
    # dist_channel：信念张量追加末通道 = BFS 距离场（belief.distance_field，账本
    # 纯派生、R1 合法）——Q 代价项 −τ/γ^τ 的主特征在目标格自身位置 RF-1 可读，
    # 探针实证旧栈距离仅靠局部观测密度代理（ρ(距离,优势)=−0.304，饱和失真）。
    # 网络侧逐通道归一（距离 /64，persistent=False buffer，旧 checkpoint 兼容）。
    dist_channel: bool = False
    # dir_channel：方向对齐度通道（2026-11-22 方向连续性特征）：末通道 = cos 对齐
    # 度场（belief.alignment_field，上个 option 净位移方向 × 逐格方向的 cos，边界
    # 历史纯派生）——"候选是否在来路方向"这一成对量在候选格自身位置 RF-1 可读。
    # 动机：repeat 主成分 = 前沿往返 = 距离打平候选间的排序失明（回头成本不对称
    # 不可见）；场形式与 dist_channel 同款 trick（预计算绕开 RF 限制）。
    dir_channel: bool = False
    # dilated_backbone：主干 dilation 1→2→4（padding=dilation，空间 16×16 不变，
    # 参数量不变，CUDA Graph 静态形状不变）→ 感受野 7→15——收益侧区域结构
    # （前沿后方黑域规模）可见。探针实证旧栈 ρ(局部信息量,优势)=+0.04 ≈ 零信号。
    dilated_backbone: bool = False
    # --- 每步管线性能等价变换（2026-11-21，逐位一致已单测验证；剖析见 CHANGELOG） ---
    lazy_replan: bool = True          # executor 账本无变化时沿缓存路径执行，跳过逐步 BFS
    scan_memo: bool = True            # belief 同 (cell,heading) 同量测的重复扫描账本标记跳过
    # --- 网络/智能体（§4.5 默认表） ---
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 64
    target_sync_options: int = 1000
    warmup_options: int = 2000
    learn_every: int = 1              # 批学习间隔：每 N 个 option 学一步（>1 省学习墙钟，
                                      # 经验照常累积；全局计数取模，checkpoint 恢复后节奏不变）
    lambda_aux: float = 0.5           # 消融 {0, 0.1, 0.5, 1.0}
    grad_clip: float = 10.0
    eps_start: float = 0.10
    eps_end: float = 0.02
    eps_anneal_options: int = 20000
    capacity: int = 100_000
    aux_head: bool = True
    use_traj: bool = True             # False → 14 通道消融
    use_gru: bool = False             # 本期前馈
    device: Optional[str] = None      # None = 自动
    # --- 实验管理（可选；swanlab 云端/本地，见 algo/monitor.py） ---
    swanlab: bool = False             # True → SwanLabMonitor 上报（未安装自动降级）
    swanlab_project: str = "BAMER-RL"
    swanlab_mode: str = "cloud"       # cloud / local / offline
    # --- 训练中周期评估（mini held-out 套件，greedy 不污染训练流） ---
    eval_every_options: int = 0       # 0 = 关闭；>0 → 每 N options 跑一次
    eval_mini_seeds: int = 3          # mini 套件种子数（seed 派生固定，跨实验可比）
    # --- 势能拉势 shaping（2026-11-21 A/C 探索塌缩修复，默认关 = 冻结契约行为） ---
    # F = γ^τ·Φ(s') − Φ(s)，Φ = 已知墙槽位比例。与 SMDP γ^τ 折扣一致 → 最优策略
    # 不变（Ng et al. 1999 / Grzes 2017 多步动作形式），只加速学习；A/C 塌缩分析
    # 见 docs/ac_collapse_analysis.md。转正需用户确认（契约 D8 语义扩展）。
    shape_potential: bool = False
    shape_scale: float = 2000.0       # Φ∈[0,1] 全程拉势总量上限；2000 ≈ 2× 臂 B 每 bit 量级
    # --- 送达奖励尺度（2026-11-22 消融，用户拍板 600/1000 双臂）：D8 里程碑
    # "首次送达 +300/台" 的参数化。300 = 归档契约行为逐位不变；600/1000 检验
    # "派送激励"假设——预注册判据 = deliver-fail↓ 且 总步数/超时不升（警惕
    # 提前派送绕路打断扫掠连续性的反噬，见 ceiling 诊断链）。
    reward_delivery: float = 300.0
    # --- CUDA Graph（纯性能开关，默认关）：act/学习体内核固化，轨迹与 eager
    # 逐位一致（warmup 无优化器副作用、clip/Adam.step 留在图外）；构建失败
    # 自动回退 eager。CPU 设备无效。
    cuda_graph: bool = False
    # --- 复现与产物 ---
    seed: int = 0
    out_dir: str = "algo/runs/exp"
    checkpoint_every_options: int = 10_000
    smoke: bool = False               # 冒烟模式：更密日志（不影响语义）

    def validate(self) -> None:
        if self.arm not in ("A", "B", "C"):
            raise ValueError(f"arm 仅支持 A/B/C，得到 {self.arm!r}")
        if self.use_gru:
            raise NotImplementedError("use_gru=True 为预留升级路径，本期未实现")
        if self.swanlab_mode not in ("cloud", "local", "offline"):
            raise ValueError(f"swanlab_mode 仅支持 cloud/local/offline，得到 {self.swanlab_mode!r}")
        if self.eval_every_options < 0 or self.eval_mini_seeds < 1:
            raise ValueError("eval_every_options 须 >=0，eval_mini_seeds 须 >=1")
        if self.shape_potential and self.shape_scale <= 0:
            raise ValueError(f"shape_potential=True 时 shape_scale 须 >0，得到 {self.shape_scale!r}")
        if self.learn_every < 1:
            raise ValueError(f"learn_every 须 >=1，得到 {self.learn_every!r}")
        if self.maze_fixed not in (None, "official", "generated"):
            raise ValueError(f"maze_fixed 仅支持 None/'official'/'generated'，得到 {self.maze_fixed!r}")

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        import yaml
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg = cls(**data)
        cfg.validate()
        return cfg


# 固定迷宫补环目标 = 官方赛图可达格环密度（215 可达格 37 环，2026-11-20 普查）
FIXED_MAZE_TARGET_LOOPS = 37


def build_fixed_maze(cfg: "TrainConfig"):
    """按 maze_fixed 配置构建固定迷宫实例（两 stage/评估共用，只建一次）。"""
    from bamer_env.maze import Maze
    if cfg.maze_fixed == "official":
        return Maze.default()
    if cfg.maze_fixed == "generated":
        # seed 取 master seed 派生值（避开 +100 环境流/+555 评估流）；环密度对齐官方
        return Maze.random_maze(n=cfg.maze_n, seed=int(cfg.seed) + 200,
                                braiding=cfg.maze_braiding_stage1,
                                target_loops=FIXED_MAZE_TARGET_LOOPS)
    raise ValueError(f"maze_fixed 非法: {cfg.maze_fixed!r}")


def stage_plan(cfg: TrainConfig) -> List[dict]:
    """三臂课程展开为阶段序列：[{name, options, env_curriculum, reward}]。"""
    if cfg.arm == "A":
        return [
            {"name": "stage0", "options": cfg.stage0_options,
             "env_curriculum": "search", "reward": "search"},
            {"name": "stage1", "options": cfg.stage1_options,
             "env_curriculum": None, "reward": "full"},
        ]
    if cfg.arm == "B":
        return [
            {"name": "stage0", "options": cfg.stage0_options,
             "env_curriculum": None, "reward": "entropy"},
            {"name": "stage1", "options": cfg.stage1_options,
             "env_curriculum": None, "reward": "full"},
        ]
    return [{"name": "full", "options": cfg.stage1_options,
             "env_curriculum": None, "reward": "full"}]


class Trainer:
    """训练驱动：episode 循环 → option 边界 → 执行器展开 → SMDP 转移 → 学习步。"""

    def __init__(self, cfg: TrainConfig):
        cfg.validate()
        self.cfg = cfg
        self.n = int(cfg.maze_n)
        # 种子管理（§6.1）：master seed → 环境/agent 两条独立流
        self._env_seed_rng = np.random.default_rng(int(cfg.seed) + 100)
        self.agent = SMDPDQNAgent(
            in_channels=14 + 2 * int(cfg.use_traj) + int(cfg.dist_channel)
                       + int(cfg.dir_channel),
            gamma=cfg.gamma, lr=cfg.lr,
            batch_size=cfg.batch_size, target_sync_options=cfg.target_sync_options,
            warmup_options=cfg.warmup_options, lambda_aux=cfg.lambda_aux,
            grad_clip=cfg.grad_clip, eps_start=cfg.eps_start, eps_end=cfg.eps_end,
            eps_anneal_options=cfg.eps_anneal_options, capacity=cfg.capacity,
            n=self.n, aux_head=cfg.aux_head, use_gru=cfg.use_gru,
            dist_channel=cfg.dist_channel, dilated=cfg.dilated_backbone,
            dir_channel=cfg.dir_channel,
            device=cfg.device, seed=int(cfg.seed), use_cuda_graph=cfg.cuda_graph)
        self.belief = BeliefState(n=self.n, max_range=cfg.lidar_max_range,
                                  scan_memo=cfg.scan_memo, use_dist=cfg.dist_channel,
                                  use_dir=cfg.dir_channel)
        self.executor = Executor(self.belief, max_option_steps=cfg.env_max_steps,
                                 lazy_replan=cfg.lazy_replan)
        self.env: Optional[BamerEnv] = None
        self._fixed_maze = None    # maze_fixed 时的共享迷宫实例（懒构建）
        self.reward_fn: Callable = self._reward_full
        self.stage_name = "full"
        self._stage_options_budget = 0
        self._last_ckpt_at = 0
        # 实验管理（swanlab，可选）+ mini 评估簿记（种子由 master seed 派生，固定可比）
        self.monitor = build_monitor(cfg)
        self._mini_eval_seeds = [int(s) for s in np.random.default_rng(
            int(cfg.seed) + 555).integers(0, 2 ** 31 - 1, int(cfg.eval_mini_seeds))]
        self._last_eval_at = 0
        # 训练簿记
        self.episodes_done = 0
        self.stage_options_done = 0
        self._episode_rows: List[dict] = []
        self._option_rows: List[dict] = []
        os.makedirs(cfg.out_dir, exist_ok=True)
        with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, ensure_ascii=False, indent=1)

    # ---------- 环境 ----------
    def _build_env(self, env_curriculum: Optional[str], stage_name: str = "stage1") -> BamerEnv:
        # 迷宫拓扑按 stage 取参（arm C 的唯一 stage 名为 "full"，归入 stage1 档）；
        # maze_fixed 时两 stage 共用同一张固定实例（课程只剩奖励切换）
        is_stage0 = stage_name == "stage0"
        braiding = self.cfg.maze_braiding_stage0 if is_stage0 else self.cfg.maze_braiding_stage1
        loops_range = self.cfg.maze_loops_stage0 if is_stage0 else self.cfg.maze_loops_stage1
        fixed_maze = None
        if self.cfg.maze_fixed is not None:
            if self._fixed_maze is None:
                self._fixed_maze = build_fixed_maze(self.cfg)
            fixed_maze = self._fixed_maze
        return BamerEnv(
            maze=fixed_maze,
            randomize=self.cfg.env_randomize, max_steps=self.cfg.env_max_steps,
            curriculum=env_curriculum,
            maze_braiding=braiding,
            maze_loops_range=tuple(loops_range) if loops_range is not None else None,
            lidar=LidarSpec(n_beams=self.cfg.lidar_n_beams,
                            max_range=self.cfg.lidar_max_range, noise_sigma=0.0),
        )

    # ---------- 奖励（D8；事件来源 info["events"] 经 executor 聚合） ----------
    def _milestones(self, events: List[object]) -> Tuple[float, Dict[str, int]]:
        """事件里程碑累计（+100 首次识别/终点发现，+cfg.reward_delivery 首次送达）。"""
        bonus = 0.0
        counts = {"identify": 0, "destination": 0, "deliver": 0}
        for ev in events:
            payload = getattr(ev, "payload", {})
            if getattr(ev, "type", None) == "ImageFeed" and payload.get("first"):
                bonus += REWARD_FIRST_IDENTIFY
                counts["identify"] += 1
            elif getattr(ev, "type", None) == "DestinationFeed" and payload.get("first"):
                bonus += REWARD_FIRST_DESTINATION
                counts["destination"] += 1
            elif getattr(ev, "type", None) == "Arrived":
                bonus += float(self.cfg.reward_delivery)
                counts["deliver"] += 1
        return bonus, counts

    def _reward_full(self, result, unknown_before: int, unknown_after: int) -> float:
        bonus, _ = self._milestones(result.events)
        return -float(result.steps) + bonus

    def _reward_search(self, result, unknown_before: int, unknown_after: int) -> float:
        """臂 A 阶段 0：全任务前半段（去掉送达里程碑）。"""
        bonus, _ = self._milestones(result.events)
        bonus -= float(self.cfg.reward_delivery) * _count_arrived(result.events)
        return -float(result.steps) + bonus

    def _reward_entropy(self, result, unknown_before: int, unknown_after: int) -> float:
        """臂 B 阶段 0：信息增益（未知墙位数精确计数，1 bit = 1 分）。"""
        return -float(result.steps) + float(unknown_before - unknown_after)

    def _shape_reward(self, base: float, unknown_before: int, unknown_after: int,
                      steps: int, terminated: bool) -> float:
        """势能拉势 shaping（Ng et al. 1999；SMDP 多步动作形式 Grzes 2017）。

        F = γ^τ·Φ(s') − Φ(s)，Φ = 已知墙槽位比例（信息势，确定性派生自信念账本，
        是策略可观测状态的函数）。与 TD target 的 γ^τ 折扣同构 → 最优策略不变，
        只加速学习。terminated 时 Φ(s')=0（定理严格条件）；truncated 自举继续，
        保持 Φ(s') 项。逐 episode 遥测和 ΣF = γ^T·Φ(s_T) − Φ(s_0) 有界，不可套利。
        2026-11-21 A/C 探索塌缩修复（docs/ac_collapse_analysis.md）；默认关。
        """
        total = 2 * self.n * self.n
        phi_s = (total - unknown_before) / total
        phi_s2 = 0.0 if terminated else (total - unknown_after) / total
        return base + (math.pow(self.cfg.gamma, steps) * phi_s2 - phi_s) * self.cfg.shape_scale

    # ---------- 主循环 ----------
    def run(self, resume_from: Optional[str] = None) -> Dict[str, object]:
        """按阶段计划训练；返回汇总统计（亦写入 out_dir/summary.json）。

        resume_from：checkpoint 路径——恢复 agent/计数/环境种子流后从该阶段
        剩余预算继续（恢复语义：checkpoint 保存于 stage 边界或中期，恢复即
        重跑当前阶段剩余预算）。
        """
        t0 = time.time()
        plan = stage_plan(self.cfg)
        start_stage = 0
        if resume_from is not None:
            start_stage = self.load_checkpoint(resume_from, plan)
            self._last_eval_at = self.agent.options_seen    # 恢复后 mini 评估从当前进度重新计时
        for si, stage in enumerate(plan):
            if si < start_stage:
                continue
            self._run_stage(si, stage)
            self._save_checkpoint(stage["name"], final=True)
        summary = {
            "wall_time_sec": round(time.time() - t0, 1),
            "episodes": self.episodes_done,
            "options": self.agent.options_seen,
            "learn_steps": self.agent.learn_steps,
            "options_per_sec": round(self.agent.options_seen / max(time.time() - t0, 1e-9), 2),
            "stages": [s["name"] for s in plan],
        }
        with open(os.path.join(self.cfg.out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        self._flush_logs()
        self.monitor.log_summary(summary)
        self.monitor.finish()
        return summary

    def load_checkpoint(self, path: str, plan: List[dict]) -> int:
        """恢复训练状态；返回应开始的阶段下标。"""
        import torch
        ckpt = torch.load(path, weights_only=False)
        saved_cfg = ckpt["cfg"]
        for key in ("arm", "stage0_options", "stage1_options", "seed", "maze_n", "maze_fixed"):
            if saved_cfg.get(key) != getattr(self.cfg, key):
                raise ValueError(f"checkpoint 配置不匹配: {key} "
                                 f"(ckpt={saved_cfg.get(key)}, cfg={getattr(self.cfg, key)})")
        self.agent.load_state_dict(ckpt["agent"])
        self.episodes_done = int(ckpt["episodes_done"])
        self.stage_options_done = int(ckpt["stage_options_done"])
        self._env_seed_rng.bit_generator.state = ckpt["env_seed_state"]
        stage_name = str(ckpt["stage"])
        for si, stage in enumerate(plan):
            if stage["name"] == stage_name:
                return si
        return len(plan)

    def _run_stage(self, stage_idx: int, stage: dict) -> None:
        self.stage_name = stage["name"]
        self._stage_options_budget = int(stage["options"])
        self.reward_fn = {
            "full": self._reward_full, "search": self._reward_search,
            "entropy": self._reward_entropy}[stage["reward"]]
        self.env = self._build_env(stage["env_curriculum"], stage["name"])
        self.monitor.log_stage(stage_idx, stage["name"])
        self.stage_options_done = 0
        self._last_ckpt_at = self.agent.options_seen
        stage_start_options = self.agent.options_seen
        if stage_idx > 0:
            # 阶段切换机制（§6.2）：buffer 清空（trainer 持有 agent）+ target 重置 + ε 重启
            self.agent.begin_stage1(self.cfg.stage1_eps_restart)
            if self.cfg.stage1_lr is not None:
                self.agent.set_lr(self.cfg.stage1_lr)
        episodes_this_stage = 0
        while self.stage_options_done < self._stage_options_budget \
                and episodes_this_stage < self.cfg.max_episodes_per_stage:
            seed = int(self._env_seed_rng.integers(0, 2 ** 31 - 1))
            row = self._run_episode(seed)
            episodes_this_stage += 1
            self.episodes_done += 1
            self._episode_rows.append(row)
            self.monitor.log_episode(row, step=self.agent.options_seen)
            if self.cfg.smoke or self.episodes_done % 10 == 0:
                self._flush_logs()
                print(f"[{self.stage_name}] ep={self.episodes_done} seed={seed} "
                      f"options={row['options']} steps={row['steps']} r={row['reward']:.0f} "
                      f"success={row['terminated']} buf={len(self.agent.buffer)} "
                      f"eps={self.agent.epsilon():.3f}")
            if self.agent.options_seen - self._last_ckpt_at >= self.cfg.checkpoint_every_options:
                self._last_ckpt_at = self.agent.options_seen
                self._save_checkpoint(stage["name"])
            if (self.cfg.eval_every_options > 0
                    and self.agent.options_seen - self._last_eval_at >= self.cfg.eval_every_options):
                self._last_eval_at = self.agent.options_seen
                self._run_mini_eval()
        self._flush_logs()
        if stage_idx == 0 and len(stage_plan(self.cfg)) > 1:
            print(f"[课程切换] {stage['name']}({stage['reward']}) → stage1：buffer 清空、"
                  f"target 重置、ε 重启 {self.cfg.stage1_eps_restart}")

    def _run_episode(self, seed: int) -> dict:
        cfg = self.cfg
        obs, info = self.env.reset(seed=seed)
        self.belief.reset(self.n)
        self.belief.observe(obs, info)
        # 辅助头 GT（训练监督专用，环境侧特权通道）：规范形式墙线 0/1
        maze = self.env.scene.maze
        gt_south = np.asarray(maze.wall_south, dtype=np.uint8)
        gt_west = np.asarray(maze.wall_west, dtype=np.uint8)

        total_reward = 0.0
        total_steps = 0
        search_steps: Optional[int] = None
        zero_step_guard = 0
        stats_rows: List[dict] = []     # maybe_learn 返回的 {td_loss, aux_loss, total_loss}
        options_this_episode = 0
        # 探索质量诊断量（2026-11-20）：重复探索率 + 单位移动步信息增益
        total_slots = 2 * self.n * self.n
        known_start = total_slots - self.belief.ledger.unknown_count()
        move_steps = 0                  # 实际位移步（F 生效；转向/碰撞不计）
        repeat_moves = 0                # 落入已探索格的位移步
        q_rows: List[dict] = []         # 贪心前向的 Q 统计（ε-random 步缺失）

        while True:
            s = self.belief.tensor(use_traj=cfg.use_traj)
            mask = legal_mask(self.belief, frontier_only=cfg.frontier_mask,
                              fallback_unvisited=cfg.fallback_unvisited)
            action = self.agent.act(s, mask)
            unknown_before = self.belief.ledger.unknown_count()
            # 本 option 开始前的已访快照：executor 执行期间每步 observe 都会把
            # 走过的格写入 visited，执行后再取会把本路径自身算作"已探索"。
            visited_before = set(self.belief.visited)

            result = self.executor.run(self.env, action)
            move_steps += max(len(result.path) - 1, 0)
            repeat_moves += _repeat_moves(result.path, visited_before)
            if self.agent.last_q_stats is not None:
                q_rows.append(self.agent.last_q_stats)
            unknown_after = self.belief.ledger.unknown_count()

            s2 = self.belief.tensor(use_traj=cfg.use_traj)
            mask2 = legal_mask(self.belief, frontier_only=cfg.frontier_mask,
                               fallback_unvisited=cfg.fallback_unvisited)
            reward = self.reward_fn(result, unknown_before, unknown_after)
            if cfg.shape_potential:
                reward = self._shape_reward(reward, unknown_before, unknown_after,
                                            result.steps, result.terminated)

            self.agent.observe(s, action, reward, s2, result.steps,
                               result.terminated, mask, mask2, gt_south, gt_west)
            # learn_every>1 时按全局 option 计数取模跳过学习步（缓冲照常写入）
            stats = (self.agent.maybe_learn()
                     if self.agent.options_seen % cfg.learn_every == 0 else None)
            self.agent.maybe_sync_target()
            if stats is not None:
                stats_rows.append(stats)

            self.belief.commit_option(result.path, result.start_cell)
            self.stage_options_done += 1
            options_this_episode += 1
            total_reward += reward
            total_steps += result.steps
            if search_steps is None and self.belief.poi_complete():
                search_steps = total_steps
            self._option_rows.append({
                "stage": self.stage_name, "episode": self.episodes_done, "seed": seed,
                "option": action, "tau": result.steps, "reward": reward,
                "success": result.success, "aborted": result.aborted,
                "terminated": result.terminated, "truncated": result.truncated,
                "eps": round(self.agent.epsilon(), 4),
                "loss": round(stats_rows[-1]["total_loss"], 5) if stats_rows else "",
            })

            if result.steps == 0:
                zero_step_guard += 1     # 到位型 option 防御（掩码语义下不应出现）
                if zero_step_guard >= 8:
                    break
            else:
                zero_step_guard = 0
            if result.terminated or result.truncated:
                break
            # 阶段预算缓冲保险丝：episode 末尾自然越过预算边界即停（避免超预算长尾）
            if self.stage_options_done >= self._stage_options_budget + cfg.env_max_steps:
                break

        n = self.n

        def _mean(key: str):
            return round(sum(s[key] for s in stats_rows) / len(stats_rows), 5) if stats_rows else ""

        known_gain = (total_slots - self.belief.ledger.unknown_count()) - known_start
        return {
            "stage": self.stage_name, "episode": self.episodes_done, "seed": seed,
            "options": options_this_episode, "steps": total_steps,
            "reward": round(total_reward, 1),
            "terminated": bool(result.terminated), "truncated": bool(result.truncated),
            "search_steps": search_steps,
            "explored_ratio": round(len(self.belief.visited) / (n * n), 4),
            "wall_knowledge": round(1.0 - unknown_after / (2 * n * n), 4),
            "move_steps": move_steps,
            "repeat_ratio": round(repeat_moves / move_steps, 4) if move_steps else "",
            "info_gain_per_move": round(known_gain / move_steps, 4) if move_steps else "",
            "mean_loss": _mean("total_loss"),
            "td_loss": _mean("td_loss"),
            "aux_loss": _mean("aux_loss"),
            "q_mean": round(sum(q["q_mean"] for q in q_rows) / len(q_rows), 3) if q_rows else "",
            "q_max": round(max(q["q_max"] for q in q_rows), 3) if q_rows else "",
            "buffer": len(self.agent.buffer),
            "epsilon": round(self.agent.epsilon(), 4),
        }

    # ---------- 训练中 mini 评估（greedy 只读，不写 buffer、不占训练种子流） ----------
    def _mini_eval_env_factory(self) -> BamerEnv:
        # mini 套件恒用 stage1 拓扑：跨阶段/跨臂曲线可比（held-out 语义）
        return self._build_env(None, "stage1")

    def _run_mini_eval(self) -> None:
        from .eval import evaluate

        summary = evaluate(self.agent, self._mini_eval_env_factory, self._mini_eval_seeds,
                           self.belief, self.executor, suite="mini",
                           frontier_only=self.cfg.frontier_mask,
                           fallback_unvisited=self.cfg.fallback_unvisited)
        self.monitor.log_eval({
            "success_rate": summary.success_rate,
            "mean_total_steps": summary.mean_total_steps,
            "mean_search_steps": summary.mean_search_steps,
            "mean_options": summary.mean_options,
            "mean_explored_ratio": summary.mean_explored_ratio,
            "mean_wall_knowledge_ratio": summary.mean_wall_knowledge_ratio,
        }, step=self.agent.options_seen)
        print(f"[mini eval] options={self.agent.options_seen} "
              f"success={summary.success_rate:.2f} "
              f"explored={summary.mean_explored_ratio:.2%} "
              f"steps={summary.mean_total_steps if summary.mean_total_steps is not None else '-'}")

    # ---------- 产物 ----------
    def _save_checkpoint(self, stage: str, final: bool = False) -> None:
        import torch
        name = f"checkpoint_{stage}_final.pt" if final else \
            f"checkpoint_{stage}_{self.agent.options_seen}.pt"
        path = os.path.join(self.cfg.out_dir, name)
        torch.save({"cfg": asdict(self.cfg), "stage": stage,
                    "episodes_done": self.episodes_done,
                    "stage_options_done": self.stage_options_done,
                    "env_seed_state": self._env_seed_rng.bit_generator.state,
                    "agent": self.agent.state_dict()}, path)
        # 只保留最近 3 个中期 checkpoint（final 除外）
        if not final:
            self._prune_checkpoints(stage)

    def _prune_checkpoints(self, stage: str, keep: int = 3) -> None:
        import glob
        prefix = os.path.join(self.cfg.out_dir, f"checkpoint_{stage}_")
        mids = sorted(p for p in glob.glob(prefix + "*.pt") if not p.endswith("_final.pt"))
        for p in mids[:-keep]:
            os.remove(p)

    def _flush_logs(self) -> None:
        self._write_csv("episodes.csv", self._episode_rows)
        self._write_csv("options.csv", self._option_rows)
        self._episode_rows.clear()
        self._option_rows.clear()

    def _write_csv(self, name: str, rows: List[dict]) -> None:
        """增量写 CSV（追加；文件头仅在新建时写入）。"""
        if not rows:
            return
        path = os.path.join(self.cfg.out_dir, name)
        new_file = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if new_file:
                writer.writeheader()
            writer.writerows(rows)


def _count_arrived(events: List[object]) -> int:
    return sum(1 for ev in events if getattr(ev, "type", None) == "Arrived")


def _repeat_moves(path: List[Tuple[int, int]], visited: Set[Tuple[int, int]]) -> int:
    """重复探索步数：本 option 实际位移落入已探索格的次数（纯函数，供单测）。

    path 含起点且只记真实位移（碰撞不入列）；visited 为本 option 开始前的
    信念已访集——BFS 路径自身无环，option 内重复不存在，无需逐步更新。
    """
    return sum(1 for c in path[1:] if c in visited)
