# -*- coding: utf-8 -*-
"""SwanLab 实验管理 hook（2026-11-20，可选依赖）。

设计（用户拍板：云端模式 + 训练中 mini 评估）：
- `build_monitor(cfg)`：cfg.swanlab=False 或未安装 swanlab → NullMonitor（零影响，
  冒烟/回归测试不触发上传）；开启 → SwanLabMonitor（swanlab.init 携带全量 config）。
- 时间轴 step = 全局 option 计数（agent.options_seen）：SMDP 训练以 option 为时间
  单位，跨 stage 连续单调，面板曲线分段可比。
- 指标分组前缀：
    train/   reward, steps, options_per_ep, success, epsilon, buffer
    explore/ explored_ratio, wall_knowledge, search_steps,
             repeat_ratio（重复探索率）, info_gain_per_move（单位移动步信息增益）
    learn/   total_loss, td_loss, aux_loss, q_mean, q_max（episode 内统计）
    stage/   stage_idx（阶段切换标记）
    eval/    mini 套件：success_rate, mean_total_steps, mean_explored_ratio, ...
    summary/ wall_time, 吞吐 options_per_sec
- 云端模式需先 `swanlab login`（API key 存本机）；local/offline 不出网。
"""
from dataclasses import asdict
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:  # 仅类型标注，避免与 trainer 循环导入
    from .trainer import TrainConfig


class NullMonitor:
    """无操作实现：swanlab 关闭/未安装时的零影响替身。"""

    def log_episode(self, row: Dict[str, object], step: int) -> None:
        pass

    def log_stage(self, stage_idx: int, stage_name: str) -> None:
        pass

    def log_eval(self, payload: Dict[str, object], step: int) -> None:
        pass

    def log_summary(self, payload: Dict[str, object]) -> None:
        pass

    def finish(self) -> None:
        pass


def _num(v: object) -> Optional[float]:
    """CSV 风格空值（None/""）归一为 None；0.0 是真值保留。"""
    return None if v is None or v == "" else float(v)


class SwanLabMonitor(NullMonitor):
    """swanlab 上报实现（init 时携带全量 TrainConfig 便于跨实验对比）。"""

    def __init__(self, cfg: "TrainConfig", experiment_name: Optional[str] = None):
        import swanlab  # 延迟导入：未安装时由 build_monitor 捕获降级

        self._sw = swanlab
        if experiment_name is None:
            exp = cfg.out_dir.rstrip("/\\").replace("\\", "/").split("/")[-1]
            experiment_name = f"{cfg.arm}-{exp}-seed{cfg.seed}"
        swanlab.init(
            project=cfg.swanlab_project,
            experiment_name=experiment_name,
            config=asdict(cfg),
            mode=cfg.swanlab_mode,
        )

    def log_episode(self, row: Dict[str, object], step: int) -> None:
        payload = {
            "train/reward": _num(row.get("reward")),
            "train/steps": _num(row.get("steps")),
            "train/options_per_ep": _num(row.get("options")),
            "train/success": 1.0 if row.get("terminated") else 0.0,
            "train/epsilon": _num(row.get("epsilon")),
            "train/buffer": _num(row.get("buffer")),
            "explore/explored_ratio": _num(row.get("explored_ratio")),
            "explore/wall_knowledge": _num(row.get("wall_knowledge")),
            "explore/search_steps": _num(row.get("search_steps")),
            "explore/repeat_ratio": _num(row.get("repeat_ratio")),
            "explore/info_gain_per_move": _num(row.get("info_gain_per_move")),
            "learn/total_loss": _num(row.get("mean_loss")),
            "learn/td_loss": _num(row.get("td_loss")),
            "learn/aux_loss": _num(row.get("aux_loss")),
            "learn/q_mean": _num(row.get("q_mean")),
            "learn/q_max": _num(row.get("q_max")),
        }
        self._sw.log({k: v for k, v in payload.items() if v is not None}, step=step)

    def log_stage(self, stage_idx: int, stage_name: str) -> None:
        self._sw.log({"stage/stage_idx": float(stage_idx)})

    def log_eval(self, payload: Dict[str, object], step: int) -> None:
        clean = {f"eval/{k}": _num(v) for k, v in payload.items()}
        self._sw.log({k: v for k, v in clean.items() if v is not None}, step=step)

    def log_summary(self, payload: Dict[str, object]) -> None:
        clean = {f"summary/{k}": v for k, v in payload.items() if v is not None}
        self._sw.log(clean)

    def finish(self) -> None:
        self._sw.finish()


def build_monitor(cfg: "TrainConfig") -> NullMonitor:
    """cfg.swanlab=False → NullMonitor；True 且未安装 swanlab → 警告后降级 Null。"""
    if not cfg.swanlab:
        return NullMonitor()
    try:
        return SwanLabMonitor(cfg)
    except ImportError:
        print("[monitor] 警告：swanlab 未安装（pip install swanlab），本次训练不上报")
        return NullMonitor()
