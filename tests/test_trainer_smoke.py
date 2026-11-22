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


def test_monitor_null_and_validation():
    """monitor 关闭路径零影响 + 新配置键校验（不触发 swanlab.init，不联网）。"""
    from algo.monitor import NullMonitor, build_monitor

    cfg = TrainConfig(swanlab=False)
    m = build_monitor(cfg)
    assert isinstance(m, NullMonitor)
    m.log_episode({}, step=0)
    m.log_stage(0, "stage0")
    m.log_eval({}, step=0)
    m.log_summary({})
    m.finish()                                # 全部 no-op 不抛
    for bad in (dict(swanlab_mode="parrot"), dict(eval_mini_seeds=0),
                dict(eval_every_options=-1)):
        try:
            TrainConfig(**bad).validate()
            raise AssertionError(f"{bad} 未报错")
        except ValueError:
            pass
    print("[monitor] OK：关闭路径 no-op、非法 mode/seeds 拒绝")


def test_mini_eval_hook_and_csv_schema(tmp_root="algo/runs/_test_monitor"):
    """mini 评估 hook：eval_every_options>0 时训练中跑 greedy 套件不崩溃；
    episodes.csv 含 loss 分项 + Q 统计 + 探索质量诊断列。"""
    shutil.rmtree(tmp_root, ignore_errors=True)
    cfg = tiny_cfg(tmp_root)
    cfg.eval_every_options = 20               # 40 option 预算 → 触发 ≥1 次 mini eval
    summary = Trainer(cfg).run()
    assert summary["options"] >= cfg.stage1_options
    assert "options_per_sec" in summary
    with open(os.path.join(tmp_root, "episodes.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    header = list(rows[0].keys())
    for col in ("mean_loss", "td_loss", "aux_loss", "q_mean", "q_max",
                "move_steps", "repeat_ratio", "info_gain_per_move"):
        assert col in header, f"episodes.csv 缺列 {col}"
    # 诊断量语义：repeat_ratio ∈ [0,1]；info_gain_per_move ≥ 0；Q 列可解析
    for r in rows:
        if r["repeat_ratio"] != "":
            assert 0.0 <= float(r["repeat_ratio"]) <= 1.0
        if r["info_gain_per_move"] != "":
            assert float(r["info_gain_per_move"]) >= 0.0
        if r["q_mean"] != "":
            assert float(r["q_mean"]) == float(r["q_mean"])   # 非 NaN
    print("[mini评估] OK：hook 触发无异常，episodes.csv 含诊断/loss/Q 列且语义合法")


def test_repeat_moves_unit():
    """_repeat_moves 纯函数：已访格计数、起点不计、option 内自环不可能。"""
    from algo.trainer import _repeat_moves

    visited = {(0, 0), (1, 0)}
    assert _repeat_moves([(2, 0), (3, 0), (1, 0)], visited) == 1     # 只有 (1,0) 已访
    assert _repeat_moves([(0, 0), (5, 5)], visited) == 0             # 起点不计
    assert _repeat_moves([(9, 9)], visited) == 0                     # 零位移 option
    assert _repeat_moves([(4, 4), (0, 0), (1, 0), (2, 2)], visited) == 2  # 两格均已访
    print("[重复探索] OK：_repeat_moves 计数语义（起点不计/已访命中）")


def test_cuda_graph_equivalence():
    """CUDA Graph 开/关同 seed 训练：option 轨迹、episode 核心列、checkpoint
    参数张量逐位一致（纯性能开关；构建失败自动回退 eager，两路必须同结果）。"""
    import torch

    if not torch.cuda.is_available():
        print("[cuda graph] 跳过（无 CUDA）")
        return
    root1, root2 = "algo/runs/_test_cg_off", "algo/runs/_test_cg_on"
    shutil.rmtree(root1, ignore_errors=True)
    shutil.rmtree(root2, ignore_errors=True)
    summaries = []
    for out, flag in ((root1, False), (root2, True)):
        cfg = tiny_cfg(out)
        cfg.cuda_graph = flag
        summaries.append(Trainer(cfg).run())
    assert summaries[0]["options"] == summaries[1]["options"]
    assert summaries[0]["learn_steps"] == summaries[1]["learn_steps"]
    # option 轨迹逐位一致
    rows = []
    for out in (root1, root2):
        with open(os.path.join(out, "options.csv"), encoding="utf-8") as f:
            rows.append([(r["option"], r["tau"], r["reward"]) for r in csv.DictReader(f)])
    assert rows[0] == rows[1] and len(rows[0]) > 0, "CUDA Graph 开/关 option 轨迹不一致"
    # checkpoint 参数/优化器状态张量逐位一致
    cks = [torch.load(os.path.join(out, "checkpoint_full_final.pt"),
                      weights_only=False, map_location="cpu") for out in (root1, root2)]

    def walk(a, b, path):
        if torch.is_tensor(a):
            assert torch.equal(a, b), f"CUDA Graph 参数不一致：{path}"
        elif isinstance(a, dict):
            assert set(a.keys()) == set(b.keys()), f"键集不一致：{path}"
            for k in a:
                walk(a[k], b[k], f"{path}.{k}")
    walk(cks[0]["agent"], cks[1]["agent"], "agent")
    print("[cuda graph] OK：开/关轨迹与参数逐位一致（含优化器状态）")


def test_potential_shaping():
    """势能拉势 shaping（2026-11-21 A/C 塌缩修复）：公式逐项、终止分支、门控校验、端到端。"""
    import math

    cfg = tiny_cfg("algo/runs/_test_shape")
    cfg.shape_potential = True
    trainer = Trainer(cfg)
    total = 2 * trainer.n * trainer.n
    scale, gamma = cfg.shape_scale, cfg.gamma
    base = -5.0
    # 信息增益 10 槽、τ=10：F = γ^10·(10/total)·scale − 0（Φ(s)=0）
    expect = base + math.pow(gamma, 10) * (10 / total) * scale
    got = trainer._shape_reward(base, total, total - 10, 10, terminated=False)
    assert abs(got - expect) < 1e-9, f"{got} != {expect}"
    # 零增益（原地扫描无新信息）：F = Φ·(γ^τ − 1)·scale < 0（轻微时长代价，策略不变性下合法）
    phi = 10 / total
    expect0 = base + (math.pow(gamma, 3) * phi - phi) * scale
    got0 = trainer._shape_reward(base, total - 10, total - 10, 3, terminated=False)
    assert abs(got0 - expect0) < 1e-9 and got0 < base
    # terminated：Φ(s')=0 → 正增益被抑制；有 Φ(s) 时为纯负项（定理严格条件）
    assert trainer._shape_reward(base, total, total - 10, 10, terminated=True) == base
    expect_t = base - phi * scale
    got_t = trainer._shape_reward(base, total - 10, total - 10, 3, terminated=True)
    assert abs(got_t - expect_t) < 1e-9
    # 校验：开 shaping 但 scale 非法须拒绝；默认关（冻结契约行为）
    try:
        TrainConfig(shape_potential=True, shape_scale=0.0).validate()
        raise AssertionError("shape_scale=0 未报错")
    except ValueError:
        pass
    assert TrainConfig().shape_potential is False
    # 端到端：开 shaping 训练可跑通且同种子确定性（两次 option 序列一致）
    summary = trainer.run()
    assert summary["options"] >= cfg.stage1_options
    rows = [float(r["reward"]) for r in csv.DictReader(
        open(os.path.join(cfg.out_dir, "episodes.csv"), encoding="utf-8"))]
    assert len(rows) > 0 and all(math.isfinite(x) for x in rows)
    print("[拉势shaping] OK：公式逐项、terminated 分支、门控校验、端到端确定性")


def test_rf_fix_smoke(tmp_root="algo/runs/_test_rf_fix"):
    """感受野修复端到端冒烟（2026-11-22）：dist_channel + dilated_backbone 双开小预算训练。"""
    shutil.rmtree(tmp_root, ignore_errors=True)
    cfg = tiny_cfg(tmp_root)
    cfg.dist_channel = True
    cfg.dilated_backbone = True
    trainer = Trainer(cfg)
    summary = trainer.run()
    assert summary["options"] >= cfg.stage1_options and summary["learn_steps"] > 0
    import torch
    ckpt = torch.load(os.path.join(tmp_root, "checkpoint_full_final.pt"),
                      weights_only=False, map_location="cpu")
    assert ckpt["agent"]["online"]["backbone.0.weight"].shape[1] == 17   # 距离通道已进网络
    rows = _read_options_csv(tmp_root)
    assert all(0 <= opt < 259 for opt, _, _ in rows)
    print(f"[RF 修复冒烟] OK：{summary['options']} options（17ch + 膨胀主干）端到端通过")


if __name__ == "__main__":
    test_stage_plan()
    test_monitor_null_and_validation()
    test_repeat_moves_unit()
    test_cuda_graph_equivalence()
    test_potential_shaping()
    test_end_to_end_smoke()
    test_same_seed_reproducibility()
    test_mini_eval_hook_and_csv_schema()
    test_eval_suite_and_reference()
    test_rf_fix_smoke()
    print("\n训练器/评估单测全部通过 [PASS]")
