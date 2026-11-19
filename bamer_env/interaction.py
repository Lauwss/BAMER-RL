# -*- coding: utf-8 -*-
"""状态机双向交互协议（M1.1，PLAN.md 阶段一）。

题目规则（docs/机器人与人工智能-项目介绍.md §二）：
- 机器鼠**实时上报自身位置坐标**给状态机；
- 机器鼠**经过状态机所在位置**时，状态机下发对应人物（屈原/昭君）的古装图片。

事件由 Env 在 step() 中经 `InteractionManager` 产生，通过 info["events"]
透传给算法侧——这是机器鼠侧的合法信息通道（AGENT.md R1 信息封闭），
事件载荷只含机器鼠自身位姿与交互反馈，不含迷宫结构/状态机位置。

事件类型（与 PLAN.md M1.1 一致）：
    Reported      机器鼠每步上报自身格坐标（题目规则：实时上报）
    ImageFeed     进入状态机所在格，状态机下发人物图像（事件 + 图像资产引用）
    Arrived       已识别状态机被派送至对应终点格
    Collision     动作指向有墙边界（位置不变，不单方面终止）
    TaskComplete  全部派送达成（terminated）
    Timeout       达到 max_steps（truncated）

派送语义（占位，最终随用户奖励设计在 M2.6 契约冻结时确认）：
    进入某终点格时，凡"已识别（收到 ImageFeed）且未派送"并映射到该终点的
    状态机自动完成派送；识别前经过终点格不产生派送。
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .scene import Scene

Cell = Tuple[int, int]

# ---------- 事件类型常量 ----------
EVENT_REPORTED = "Reported"
EVENT_IMAGE_FEED = "ImageFeed"
EVENT_ARRIVED = "Arrived"
EVENT_COLLISION = "Collision"
EVENT_TASK_COMPLETE = "TaskComplete"
EVENT_TIMEOUT = "Timeout"

# 人物图像资产（取自 PPT 素材，与 render.py 同源；缺失时引用为 None）
MEDIA_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "docs", "_pptx_media", "ppt", "media"))
IMAGE_ASSETS = {
    "quyuan": os.path.join(MEDIA_DIR, "image2.jpeg"),
    "zhaojun": os.path.join(MEDIA_DIR, "image4.jpeg"),
}
PERSON_LABELS = {"quyuan": "屈原", "zhaojun": "昭君"}


def image_asset(name: str) -> Optional[str]:
    """状态机对应人物图像资产路径；素材缺失时返回 None（仅引用语义不受影响）。"""
    path = IMAGE_ASSETS.get(name)
    return path if path and os.path.exists(path) else None


@dataclass(frozen=True)
class Event:
    """单条交互事件。payload 只放 JSON 兼容的基本类型，便于后续数据 IO（M2.4）。"""
    type: str
    step: int
    payload: Dict[str, object] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[step {self.step:>4}] {self.type:<12} {self.payload}"


class InteractionManager:
    """状态机双向交互协议 + 任务簿记。

    Env 每 step 调用本管理器产出事件；识别/派送进度（identified/delivered）
    属于交互协议状态，供 terminated 判定与展示层使用。
    """

    def __init__(self, scene: Scene):
        self._scene = scene
        self.identified: set = set()   # 已收到 ImageFeed 的状态机名
        self.delivered: set = set()    # 已派送到对应终点的状态机名
        self.reset()

    def reset(self) -> None:
        self.identified = set()
        self.delivered = set()

    # ---------- 上报通道（机器鼠 → 状态机） ----------
    def report(self, step: int, pos: Cell) -> Event:
        """机器鼠每步实时上报自身格坐标（题目规则）。"""
        return Event(EVENT_REPORTED, step, {"x": int(pos[0]), "y": int(pos[1])})

    # ---------- Env 动作结果 → 事件 ----------
    def collision_event(self, step: int, pos: Cell, direction: str) -> Event:
        """动作指向有墙边界：位置不变 + Collision 事件（惩罚/终止语义由用户定）。"""
        return Event(EVENT_COLLISION, step,
                     {"x": int(pos[0]), "y": int(pos[1]), "direction": direction})

    def timeout_event(self, step: int, max_steps: int) -> Event:
        return Event(EVENT_TIMEOUT, step, {"max_steps": int(max_steps)})

    # ---------- 进入新格：图像下发 + 派送判定 ----------
    def enter_events(self, step: int, pos: Cell, prev_pos: Optional[Cell] = None) -> List[Event]:
        """机器鼠到达新格（或出生格）后的交互事件。

        - 进入状态机所在格（prev_pos 不同，含出生在该格）→ ImageFeed；
          同格内停留/原地动作不重复下发；离开后再进入会重新下发（"经过即反馈"）。
        - 进入终点格 → 已识别且未派送、映射到该终点的状态机完成派送 → Arrived；
          全部派送达成 → TaskComplete（只在该转变发生时发一次）。
        """
        out: List[Event] = []

        # 状态机下发人物图像（识别 = 第一次 ImageFeed）
        for name, cell in self._scene.state_machines.items():
            if pos == cell and pos != prev_pos:
                out.append(Event(EVENT_IMAGE_FEED, step, {
                    "machine": name,
                    "label": PERSON_LABELS.get(name, name),
                    "image": image_asset(name) or "",
                    "first": name not in self.identified,
                }))
                self.identified.add(name)

        # 派送判定（占位语义，见模块 docstring）
        delivered_now = False
        for dname, dcell in self._scene.destinations.items():
            if pos != dcell:
                continue
            for mname, mapped in self._scene.delivery_map.items():
                if mapped == dname and mname in self.identified and mname not in self.delivered:
                    self.delivered.add(mname)
                    delivered_now = True
                    out.append(Event(EVENT_ARRIVED, step, {
                        "machine": mname,
                        "destination": dname,
                        "x": int(pos[0]), "y": int(pos[1]),
                    }))
        if delivered_now and self.task_complete:
            out.append(Event(EVENT_TASK_COMPLETE, step, {"steps": int(step)}))
        return out

    # ---------- 任务状态 ----------
    @property
    def task_complete(self) -> bool:
        """全部状态机均已派送（terminated 判据，占位语义随 M2.6 确认）。"""
        return bool(self._scene.state_machines) and self.delivered == set(self._scene.state_machines)
