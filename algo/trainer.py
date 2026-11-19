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
import os
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from bamer_env.env import BamerEnv
from bamer_mouse.lidar import LidarSpec

from .actions import legal_mask
from .agent import SMDPDQNAgent
from .belief import BeliefState
from .executor import Executor

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
    # --- 网络/智能体（§4.5 默认表） ---
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 64
    target_sync_options: int = 1000
    warmup_options: int = 2000
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

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        import yaml
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg = cls(**data)
        cfg.validate()
        return cfg


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
            in_channels=16 if cfg.use_traj else 14, gamma=cfg.gamma, lr=cfg.lr,
            batch_size=cfg.batch_size, target_sync_options=cfg.target_sync_options,
            warmup_options=cfg.warmup_options, lambda_aux=cfg.lambda_aux,
            grad_clip=cfg.grad_clip, eps_start=cfg.eps_start, eps_end=cfg.eps_end,
            eps_anneal_options=cfg.eps_anneal_options, capacity=cfg.capacity,
            n=self.n, aux_head=cfg.aux_head, use_gru=cfg.use_gru,
            device=cfg.device, seed=int(cfg.seed))
        self.belief = BeliefState(n=self.n, max_range=cfg.lidar_max_range)
        self.executor = Executor(self.belief, max_option_steps=cfg.env_max_steps)
        self.env: Optional[BamerEnv] = None
        self.reward_fn: Callable = self._reward_full
        self.stage_name = "full"
        self._stage_options_budget = 0
        self._last_ckpt_at = 0
        # 训练簿记
        self.episodes_done = 0
        self.stage_options_done = 0
        self._episode_rows: List[dict] = []
        self._option_rows: List[dict] = []
        os.makedirs(cfg.out_dir, exist_ok=True)
        with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, ensure_ascii=False, indent=1)

    # ---------- 环境 ----------
    def _build_env(self, env_curriculum: Optional[str]) -> BamerEnv:
        return BamerEnv(
            randomize=self.cfg.env_randomize, max_steps=self.cfg.env_max_steps,
            curriculum=env_curriculum,
            lidar=LidarSpec(n_beams=self.cfg.lidar_n_beams,
                            max_range=self.cfg.lidar_max_range, noise_sigma=0.0),
        )

    # ---------- 奖励（D8；事件来源 info["events"] 经 executor 聚合） ----------
    @staticmethod
    def _milestones(events: List[object]) -> Tuple[float, Dict[str, int]]:
        """事件里程碑累计（+100 首次识别/终点发现，+300 首次送达）。"""
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
                bonus += REWARD_FIRST_DELIVER
                counts["deliver"] += 1
        return bonus, counts

    def _reward_full(self, result, unknown_before: int, unknown_after: int) -> float:
        bonus, _ = self._milestones(result.events)
        return -float(result.steps) + bonus

    def _reward_search(self, result, unknown_before: int, unknown_after: int) -> float:
        """臂 A 阶段 0：全任务前半段（去掉送达 +300）。"""
        bonus, _ = self._milestones(result.events)
        bonus -= REWARD_FIRST_DELIVER * _count_arrived(result.events)
        return -float(result.steps) + bonus

    def _reward_entropy(self, result, unknown_before: int, unknown_after: int) -> float:
        """臂 B 阶段 0：信息增益（未知墙位数精确计数，1 bit = 1 分）。"""
        return -float(result.steps) + float(unknown_before - unknown_after)

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
            "stages": [s["name"] for s in plan],
        }
        with open(os.path.join(self.cfg.out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        self._flush_logs()
        return summary

    def load_checkpoint(self, path: str, plan: List[dict]) -> int:
        """恢复训练状态；返回应开始的阶段下标。"""
        import torch
        ckpt = torch.load(path, weights_only=False)
        saved_cfg = ckpt["cfg"]
        for key in ("arm", "stage0_options", "stage1_options", "seed", "maze_n"):
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
        self.env = self._build_env(stage["env_curriculum"])
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
            if self.cfg.smoke or self.episodes_done % 10 == 0:
                self._flush_logs()
                print(f"[{self.stage_name}] ep={self.episodes_done} seed={seed} "
                      f"options={row['options']} steps={row['steps']} r={row['reward']:.0f} "
                      f"success={row['terminated']} buf={len(self.agent.buffer)} "
                      f"eps={self.agent.epsilon():.3f}")
            if self.agent.options_seen - self._last_ckpt_at >= self.cfg.checkpoint_every_options:
                self._last_ckpt_at = self.agent.options_seen
                self._save_checkpoint(stage["name"])
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
        losses: List[float] = []
        options_this_episode = 0

        while True:
            s = self.belief.tensor(use_traj=cfg.use_traj)
            mask = legal_mask(self.belief)
            action = self.agent.act(s, mask)
            unknown_before = self.belief.ledger.unknown_count()

            result = self.executor.run(self.env, action)
            unknown_after = self.belief.ledger.unknown_count()

            s2 = self.belief.tensor(use_traj=cfg.use_traj)
            mask2 = legal_mask(self.belief)
            reward = self.reward_fn(result, unknown_before, unknown_after)

            self.agent.observe(s, action, reward, s2, result.steps,
                               result.terminated, mask, mask2, gt_south, gt_west)
            stats = self.agent.maybe_learn()
            self.agent.maybe_sync_target()
            if stats is not None:
                losses.append(stats["total_loss"])

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
                "loss": round(losses[-1], 5) if losses else "",
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
        return {
            "stage": self.stage_name, "episode": self.episodes_done, "seed": seed,
            "options": options_this_episode, "steps": total_steps,
            "reward": round(total_reward, 1),
            "terminated": bool(result.terminated), "truncated": bool(result.truncated),
            "search_steps": search_steps,
            "explored_ratio": round(len(self.belief.visited) / (n * n), 4),
            "wall_knowledge": round(1.0 - unknown_after / (2 * n * n), 4),
            "mean_loss": round(sum(losses) / len(losses), 5) if losses else "",
            "buffer": len(self.agent.buffer),
            "epsilon": round(self.agent.epsilon(), 4),
        }

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
