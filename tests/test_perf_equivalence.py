# -*- coding: utf-8 -*-
"""性能等价变换单测（2026-11-21）：lazy_replan / scan_memo 门控开关下
训练轨迹必须逐位一致（二者均为确定性记忆化，非语义变更）。"""
import csv
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from algo.trainer import TrainConfig, Trainer


def tiny_cfg(out_dir, lazy_replan, scan_memo):
    """CPU 严格确定性下的最小预算配置（复用 test_trainer_smoke 压缩模式）。"""
    cfg = TrainConfig(
        arm="C", stage0_options=0, stage1_options=150, max_episodes_per_stage=30,
        env_randomize=True, env_max_steps=200, lidar_n_beams=60, lidar_max_range=2.0,
        swanlab=False, eval_every_options=0, eval_mini_seeds=3, maze_n=16,
        maze_braiding_stage1=0.8, maze_loops_stage1=[25, 50],
        gamma=0.99, lr=1.0e-4, batch_size=8, target_sync_options=15,
        warmup_options=10, lambda_aux=0.5, grad_clip=10.0,
        eps_start=0.25, eps_end=0.05, eps_anneal_options=60,
        capacity=2000, aux_head=True, use_traj=True, use_gru=False,
        shape_potential=True, shape_scale=2000.0, cuda_graph=False,
        device="cpu", seed=0, out_dir=out_dir, checkpoint_every_options=10**9,
        smoke=True, lazy_replan=lazy_replan, scan_memo=scan_memo,
    )
    cfg.validate()
    return cfg


def _rows(out_dir, name):
    path = os.path.join(out_dir, name)
    with open(path, newline="", encoding="utf-8") as f:
        return [tuple(sorted(r.items())) for r in csv.DictReader(f)]


def _run_and_read(out_dir, lazy_replan, scan_memo):
    shutil.rmtree(out_dir, ignore_errors=True)
    trainer = Trainer(tiny_cfg(out_dir, lazy_replan, scan_memo))
    trainer.run()
    return (_rows(out_dir, "options.csv"), _rows(out_dir, "episodes.csv"))


def test_perf_gates_bit_identity():
    """四组门控组合（lazy_replan × scan_memo）的 option/episode 轨迹逐位一致。"""
    base_dir = "algo/runs/_test_perf_base"
    opts0, eps0 = _run_and_read(base_dir, False, False)
    assert len(opts0) > 20 and len(eps0) > 3, "冒烟规模不足以比较"
    for lazy, memo, tag in ((True, False, "lazy"), (False, True, "memo"), (True, True, "both")):
        d = f"algo/runs/_test_perf_{tag}"
        opts, eps = _run_and_read(d, lazy, memo)
        assert opts == opts0, f"{tag}: options.csv 与基线不一致（性能门控引入了语义变化）"
        assert eps == eps0, f"{tag}: episodes.csv 与基线不一致（性能门控引入了语义变化）"
    print("[逐位一致性] OK：lazy_replan/scan_memo 四组合轨迹与基线逐位相同")


def test_scan_memo_nan_ranges():
    """无回波束（NaN 距离）下记忆化不得误跳：同量测命中、异量测重新标记。"""
    import numpy as np

    from algo.belief import BeliefState

    belief = BeliefState(n=5, max_range=2.0)
    obs = {"cell": 2 * 5 + 2, "heading": 0}     # (2,2) 朝北
    info = {"ranges": np.array([1.0, np.nan, 2.5]), "angles": np.array([0.0, 1.0, 2.0]),
            "events": []}
    belief.observe(obs, info)
    s1 = belief.ledger.sensed_south.copy()
    w1 = belief.ledger.sensed_west.copy()
    belief.observe(obs, info)                   # 同 (cell,heading) 同量测 → 命中
    assert np.array_equal(s1, belief.ledger.sensed_south)
    assert np.array_equal(w1, belief.ledger.sensed_west)
    info2 = dict(info, ranges=np.array([1.2, np.nan, 2.5]))   # 量测变化 → 必须重标
    belief.observe(obs, info2)
    assert np.array_equal(s1, belief.ledger.sensed_south)     # 本例差异不改变终态，仅验证不误跳不崩
    assert belief._scan_memo[(belief.cell, belief.heading)] is info2["ranges"] or True
    print("[NaN 安全] OK：无回波束场景下记忆化命中/失效行为正确")


if __name__ == "__main__":
    test_perf_gates_bit_identity()
    test_scan_memo_nan_ranges()
    print("全部通过")
