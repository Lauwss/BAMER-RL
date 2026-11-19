# -*- coding: utf-8 -*-
"""训练器/评估单测（§8 冒烟 5）：本地分钟级端到端冒烟 → 同种子复现 → 固定 seed 套件评估 + 全知 BFS 参照。"""
import csv
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from algo.trainer import TrainConfig, Trainer, stage_plan

SMOKE_YAML = os.path.join(os.path.dirname(__file__), "..", "algo", "configs", "smoke.yaml")


def tiny_cfg(out_dir, seed=0, arm="C"):
    """冒烟级配置（压低预算到分钟内）：臂 C 从头直训。"""
    cfg = TrainConfig.from_yaml(SMOKE_YAML)
    cfg.out_dir = out_dir
    cfg.seed = seed
    cfg.arm = arm
    cfg.stage1_options = 40          # option 预算（复现测试压到最小可学习规模）
    cfg.warmup_options = 10
    cfg.batch_size = 8
    cfg.target_sync_options = 15
    cfg.max_episodes_per_stage = 30
    cfg.env_max_steps = 200
    cfg.lidar_n_beams = 60
    cfg.checkpoint_every_options = 1000   # 期间不产中期 ckpt
    return cfg


def test_stage_plan():
    """三臂课程展开：A/B 两阶段、C 单阶段；env curriculum 只在臂 A 阶段 0 打开。"""
    cfg = TrainConfig(arm="A", stage0_options=100, stage1_options=200)
    plan = stage_plan(cfg)
    assert [s["name"] for s in plan] == ["stage0", "stage1"]
    assert plan[0]["env_curriculum"] == "search" and plan[0]["reward"] == "search"
    assert plan[1]["env_curriculum"] is None and plan[1]["reward"] == "full"
    cfg_b = TrainConfig(arm="B", stage0_options=100, stage1_options=200)
    assert stage_plan(cfg_b)[0]["reward"] == "entropy"
    cfg_c = TrainConfig(arm="C")
    assert [s["name"] for s in stage_plan(cfg_c)] == ["full"]
    try:
        TrainConfig(arm="X").validate()
        raise AssertionError("非法 arm 未报错")
    except ValueError:
        pass
    print("[课程计划] OK：三臂展开与切换语义")


def _read_options_csv(out_dir):
    rows = []
    with open(os.path.join(out_dir, "options.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((int(row["option"]), int(row["tau"]), row["reward"]))
    return rows


def test_end_to_end_smoke(tmp_root="algo/runs/_test_e2e"):
    """端到端：臂 C 小预算训练 → 产物齐备；option 全部合法索引；损失无 NaN。"""
    shutil.rmtree(tmp_root, ignore_errors=True)
    cfg = tiny_cfg(tmp_root)
    trainer = Trainer(cfg)
    t0 = time.time()
    summary = trainer.run()
    dt = time.time() - t0
    assert summary["options"] >= cfg.stage1_options
    assert summary["learn_steps"] > 0
    assert os.path.exists(os.path.join(tmp_root, "episodes.csv"))
    assert os.path.exists(os.path.join(tmp_root, "config.json"))
    assert os.path.exists(os.path.join(tmp_root, "checkpoint_full_final.pt"))
    rows = _read_options_csv(tmp_root)
    assert len(rows) >= cfg.stage1_options
    assert all(0 <= opt < 259 for opt, _, _ in rows), "选项索引越界"
    assert all(tau >= 0 for _, tau, _ in rows)
    # 无 NaN 的损失列（有学习步时）
    with open(os.path.join(tmp_root, "options.csv"), encoding="utf-8") as f:
        losses = [float(r["loss"]) for r in csv.DictReader(f) if r["loss"] != ""]
    assert losses and all(loss == loss and abs(loss) < 1e6 for loss in losses)
    print(f"[端到端冒烟] OK：{summary['options']} options / {summary['learn_steps']} 学习步 "
          f"/ {dt:.1f}s / 产物齐备")


def test_same_seed_reproducibility():
    """复现性（§6.1）：同 seed 两次训练 → option 序列（动作/τ/奖励）逐行一致（CPU）。"""
    root_a, root_b = "algo/runs/_test_repro_a", "algo/runs/_test_repro_b"
    shutil.rmtree(root_a, ignore_errors=True)
    shutil.rmtree(root_b, ignore_errors=True)
    Trainer(tiny_cfg(root_a)).run()
    Trainer(tiny_cfg(root_b)).run()
    ra, rb = _read_options_csv(root_a), _read_options_csv(root_b)
    assert len(ra) == len(rb) and len(ra) >= 40
    assert ra == rb, "同种子两次训练 option 序列不一致"
    print(f"[同种子复现] OK：{len(ra)} 条 option 转移逐行一致（CPU 严格确定性）")


def test_eval_suite_and_reference(tmp_root="algo/runs/_test_eval"):
    """评估套件：贪心策略跑固定 seeds + 全知 BFS 参照为正整数且 ≤ 保守上界。"""
    shutil.rmtree(tmp_root, ignore_errors=True)
    cfg = tiny_cfg(tmp_root)
    Trainer(cfg).run()                       # 训练一个（很弱的）策略
    import torch

    from algo.agent import SMDPDQNAgent
    from algo.belief import BeliefState
    from algo.eval import evaluate, omniscient_reference, summarize_with_reference
    from algo.executor import Executor
    from tools.train import build_env_factory

    ckpt = torch.load(os.path.join(tmp_root, "checkpoint_full_final.pt"),
                      weights_only=False, map_location="cpu")
    agent = SMDPDQNAgent(in_channels=16, n=cfg.maze_n, device="cpu", seed=cfg.seed)
    agent.load_state_dict(ckpt["agent"])

    seeds = [11, 22, 33]
    belief = BeliefState(n=cfg.maze_n, max_range=cfg.lidar_max_range)
    executor = Executor(belief, max_option_steps=cfg.env_max_steps)
    summary = evaluate(agent, build_env_factory(cfg), seeds, belief, executor)
    assert summary.n_episodes == len(seeds)
    assert 0.0 <= summary.success_rate <= 1.0
    for m in summary.per_episode:
        assert m.total_steps > 0 and 0.0 < m.explored_ratio <= 1.0
        assert 0.0 < m.wall_knowledge_ratio <= 1.0

    # 全知参照：官方图 + 冒烟种子复现场景
    from bamer_env.scene import build_scene
    scene = build_scene()
    ref = omniscient_reference(scene)
    assert ref is not None and ref > 0
    # 参照 ≤ 全知直线任务的合理上界（曼哈顿环最差 < 4*16*2+若干转向）
    assert ref < 4 * 16 * 3, ref
    summary = summarize_with_reference(summary, {s: None for s in seeds})
    assert summary.mean_gap_to_reference is None     # 场景缺失时不核算
    print(f"[评估套件] OK：{summary.n_episodes} seeds 贪心评估 + 全知参照 = {ref} 步")


if __name__ == "__main__":
    test_stage_plan()
    test_end_to_end_smoke()
    test_same_seed_reproducibility()
    test_eval_suite_and_reference()
    print("\n训练器/评估单测全部通过 [PASS]")
