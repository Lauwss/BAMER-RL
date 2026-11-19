# -*- coding: utf-8 -*-
"""摄像头与识别挂接（M2.5，PLAN.md 阶段二）。

触发语义（与 M1.1 交互协议一致）：机器鼠进入状态机所在格 → 状态机下发
ImageFeed 事件（含人物图像资产引用）；离开再进入会重新下发（"经过即反馈"）。

归属说明（D9 待用户拍板）：图像数据集构建与真实分类模型（深度学习分类）
训练不在当前范围内。本模块只提供——
    1. Classifier 协议：可插拔识别接口，真实分类器后续实现该协议即可挂接；
    2. GTPassthroughClassifier：GT 直通占位（管线打通/展示用，非识别成绩）；
    3. CameraUnit：ImageFeed 事件 → 取图 → 识别 → Recognition 记录 的挂接流水线。

R1 边界提示（M2.6 契约冻结需用户确认）：当前 ImageFeed 事件载荷同时携带
machine/label 字段——算法侧若直接读取等于免识别获知人物身份。识别语义上
算法侧只应消费 image（像素）。CameraUnit 的识别流水线默认只把图像像素交给
classifier；gt_machine 仅在 evaluate=True 的评估口径（环境侧/汇报）参与
正确率统计，不进入决策数据流。

分类器输出契约：classify 返回状态机身份标识（"quyuan"/"zhaojun"）——
即派送判定所需的身份语义；展示用中文标签由事件 payload["label"] 提供。
"""
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Protocol

import numpy as np


class Classifier(Protocol):
    """识别器协议（可插拔）：真实深度分类器实现本协议即可挂接 CameraUnit。

    image 为 RGB uint8 数组（H, W, 3）；gt_label 仅评估口径可选传入
    （GT 直通占位实现依赖它；真实识别器应忽略该参数）。
    返回状态机身份标识（"quyuan"/"zhaojun"）。
    """

    def classify(self, image: np.ndarray, *, gt_label: Optional[str] = None) -> str:
        ...


class GTPassthroughClassifier:
    """GT 直通占位（D9 待定）：不做真实识别，直接返回事件携带的 GT 身份。

    仅用于打通"识别 → 派送"管线与展示；其输出不得作为识别能力口径汇报。
    gt_label 缺省时报错——强制调用方显式意识到自己在走 GT 通道。
    """

    def classify(self, image: np.ndarray, *, gt_label: Optional[str] = None) -> str:
        if gt_label is None:
            raise ValueError("GT 直通占位需要显式提供 gt_label（评估口径专用）")
        return gt_label


@dataclass(frozen=True)
class CameraFrame:
    """一次图像下发的快照。gt_machine/gt_label 来自事件载荷，仅供评估与展示。"""
    step: int
    image_path: str
    image: np.ndarray
    gt_machine: str
    gt_label: str


@dataclass(frozen=True)
class Recognition:
    """一次识别结果。correct 仅在 evaluate=True（评估口径）时有值。"""
    step: int
    machine: str          # GT 身份
    predicted: str        # 分类器输出
    correct: Optional[bool]


def load_image(path: str) -> np.ndarray:
    """读取图像资产 → RGB uint8 数组（路径为空/缺失时报错，不静默吞掉）。"""
    from PIL import Image   # 展示层依赖，按需导入

    if not path or not np.asarray(path).size:
        raise FileNotFoundError("ImageFeed 未携带图像资产引用（payload.image 为空）")
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


class CameraUnit:
    """识别挂接流水线：监听 ImageFeed → 取图 → Classifier → Recognition。

    事件为鸭型对象（bamer_env.interaction.Event 满足）：type/step/payload。
    本模块不 import bamer_env，保持 bamer_mouse 单向依赖。
    """

    def __init__(self, classifier: Classifier):
        self._classifier = classifier
        self.frames: List[CameraFrame] = []       # 累积的图像下发快照
        self.recognitions: List[Recognition] = [] # 累积的识别记录

    def collect(self, events: Iterable[Any]) -> List[CameraFrame]:
        """从事件流中提取 ImageFeed 帧（含取图）。"""
        out: List[CameraFrame] = []
        for ev in events:
            if getattr(ev, "type", None) != "ImageFeed":
                continue
            payload = getattr(ev, "payload", {})
            out.append(CameraFrame(
                step=int(getattr(ev, "step", 0)),
                image_path=str(payload.get("image", "")),
                image=load_image(payload.get("image", "")),
                gt_machine=str(payload.get("machine", "")),
                gt_label=str(payload.get("label", "")),
            ))
        self.frames.extend(out)
        return out

    def recognize(self, frame: CameraFrame, *, evaluate: bool = False) -> Recognition:
        """对一帧图像执行识别；evaluate=True 时附带 GT 正确率（评估口径）。"""
        predicted = self._classifier.classify(frame.image, gt_label=frame.gt_machine)
        rec = Recognition(
            step=frame.step, machine=frame.gt_machine, predicted=predicted,
            correct=(predicted == frame.gt_machine) if evaluate else None,
        )
        self.recognitions.append(rec)
        return rec

    def on_events(self, events: Iterable[Any], *, evaluate: bool = False) -> List[Recognition]:
        """组合入口：collect + 逐帧 recognize。"""
        return [self.recognize(f, evaluate=evaluate) for f in self.collect(events)]
