# -*- coding: utf-8 -*-
"""状态机双向交互协议（M1.1，PLAN.md 阶段一）。

题目规则（docs/机器人与人工智能-项目介绍.md §二）：
- 机器鼠**实时上报自身位置坐标**给状态机；
- 机器鼠**经过状态机所在位置**时，状态机下发对应人物（屈原/昭君）的古装图片。

事件由 Env 在 step() 中经 `InteractionManager` 产生，通过 info["events"]
透传给算法侧——这是机器鼠侧的合法信息通道（AGENT.md R1 信息封闭），
事件载荷只含机器鼠自身位姿与交互反馈，不含迷宫结构/状态机位置。

事件类型（contract_v1.md §3 定稿语义，M2.6 冻结候选）：
    Reported         机器鼠每步上报自身格坐标（题目规则：实时上报）
    ImageFeed        进入状态机所在格，状态机下发人物图像（事件 + 图像资产引用）
    Pickup           【2026-11-20 新增】进入状态机所在格且空手且未送达 → carrying 生效
    DestinationFeed  【2026-11-20 新增】进入终点所在格（发现终点；终点位置封闭，须自主发现）
    Arrived          进入"已携带且映射"的终点格 → 派送自动完成（无需显式放下动作）
    Collision        动作指向有墙边界（位置不变，不单方面终止）
    TaskComplete     全部派送达成（terminated，仅转变时一次）
    Timeout          达到 max_steps（truncated）

拾取语义（2026-11-20 用户拍板"显式 Pick + 进终点格自动派送"的 env 层落地）：
    拾取 = 到达状态机所在格这一事实本身（执行器只产出 D5 原语，env 不感知
    option 意图；决策层的显式 Pick 体现为"以该机已知格为目标到达"）。首达
    即识别 + 拾取同步发生（ImageFeed → Pickup 同一进入事件序列，符合"搜寻
    + 派送步数最优"）；识别时手持他机则仅识别不拾取，事后经 Pick option
    回取——这正是 259 选项中 2 个 Pick 动作的真实语义。已送达机器不再重复
    拾取；单机负载（一次只能背一台）。
派送语义（2026-11-20 用户拍板）：
    进入"已携带且映射"的终点格 → 自动派送（Arrived）；未携带（或携带的非
    映射机器）进终点格 = 仅发现该终点（DestinationFeed）。终点位置固定但对
    机器鼠封闭，须自主发现。
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .scene import Scene

Cell = Tuple[int, int]

# ---------- 事件类型常量 ----------
EVENT_REPORTED = "Reported"
EVENT_IMAGE_FEED = "ImageFeed"
EVENT_PICKUP = "Pickup"
EVENT_DESTINATION_FEED = "DestinationFeed"
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
# 终点图像资产（DestinationFeed 下发；与 render.py 终点图标同源）
DESTINATION_ASSETS = {
    "chuguo": os.path.join(MEDIA_DIR, "image3.jpeg"),
    "xihan": os.path.join(MEDIA_DIR, "image5.jpeg"),
}
PERSON_LABELS = {"quyuan": "屈原", "zhaojun": "昭君"}


def image_asset(name: str) -> Optional[str]:
    """状态机对应人物图像资产路径；素材缺失时返回 None（仅引用语义不受影响）。"""
    path = IMAGE_ASSETS.get(name)
    return path if path and os.path.exists(path) else None


def destination_asset(name: str) -> Optional[str]:
    """终点对应图像资产路径；素材缺失时返回 None（仅引用语义不受影响）。"""
    path = DESTINATION_ASSETS.get(name)
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
        self.identified: set = set()    # 已收到 ImageFeed 的状态机名
        self.delivered: set = set()     # 已派送到对应终点的状态机名
        self.carrying: Optional[str] = None   # 正在携带的状态机名（None = 空手）
        self.discovered: set = set()    # 已发现（进入过）的终点名
        self.reset()

    def reset(self) -> None:
        self.identified = set()
        self.delivered = set()
        self.carrying = None
        self.discovered = set()

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

    # ---------- 进入新格：图像下发 + 拾取 + 终点发现 + 派送判定 ----------
    def enter_events(self, step: int, pos: Cell, prev_pos: Optional[Cell] = None) -> List[Event]:
        """机器鼠到达新格（或出生格）后的交互事件。

        同格内停留/原地动作（pos == prev_pos）不触发任何进入事件；离开再进入
        会重新下发图像（"经过即反馈"）。同一进入的事件序列按序：

        1. ImageFeed：进入状态机所在格 → 状态机下发人物图像（first = 首次识别）；
        2. Pickup：空手且该机未送达 → carrying 生效（单机负载；首达即识别 +
           拾取同步发生——见模块 docstring 拾取语义）；
        3. DestinationFeed：进入终点所在格 → 终点图像下发（first = 首次发现；
           终点封闭，须自主发现——这是算法侧唯一的终点发现通道）；
        4. Arrived：当前携带机器映射到该终点 → 派送自动完成（carrying 清空，
           无需显式放下动作）；全部派送达成 → TaskComplete（仅转变时一次）。
        """
        out: List[Event] = []
        entered = pos != prev_pos

        # 1. 状态机下发人物图像（识别 = 第一次 ImageFeed）
        if entered:
            for name, cell in self._scene.state_machines.items():
                if pos == cell:
                    out.append(Event(EVENT_IMAGE_FEED, step, {
                        "machine": name,
                        "label": PERSON_LABELS.get(name, name),
                        "image": image_asset(name) or "",
                        "first": name not in self.identified,
                    }))
                    self.identified.add(name)

        # 2. 拾取判定：空手 + 该机未送达 → carrying 生效（Pickup 为 carrying 的合法信息通道）
        if entered and self.carrying is None:
            for name, cell in self._scene.state_machines.items():
                if pos == cell and name not in self.delivered:
                    self.carrying = name
                    out.append(Event(EVENT_PICKUP, step, {
                        "machine": name,
                        "x": int(pos[0]), "y": int(pos[1]),
                    }))
                    break    # 单机负载：至多拾取一台

        # 3. 终点发现（DestinationFeed）：发现终点 = 算法侧合法信息通道
        if entered:
            for dname, dcell in self._scene.destinations.items():
                if pos == dcell:
                    out.append(Event(EVENT_DESTINATION_FEED, step, {
                        "destination": dname,
                        "image": destination_asset(dname) or "",
                        "first": dname not in self.discovered,
                    }))
                    self.discovered.add(dname)

        # 4. 派送判定（2026-11-20 拍板语义）：已携带且映射 → 自动派送
        delivered_now = False
        if self.carrying is not None:
            mname = self.carrying
            dname = self._scene.delivery_map.get(mname)
            if dname is not None and self._scene.destinations.get(dname) == pos:
                self.delivered.add(mname)
                self.carrying = None
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
        """全部状态机均已派送（terminated 判据，全任务语义）。"""
        return bool(self._scene.state_machines) and self.delivered == set(self._scene.state_machines)

    @property
    def search_complete(self) -> bool:
        """全部 POI（2 台机器 + 2 个终点）均已定位（课程臂 A 阶段 0 的 terminated 判据）。"""
        return (bool(self._scene.state_machines) and self.identified == set(self._scene.state_machines)
                and bool(self._scene.destinations) and self.discovered == set(self._scene.destinations))
