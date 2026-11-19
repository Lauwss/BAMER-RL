# -*- coding: utf-8 -*-
"""摄像头/识别挂接单测（M2.5）：Classifier 协议、GT 直通占位、CameraUnit 流水线。"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from bamer_env.interaction import image_asset
from bamer_mouse import CameraUnit, GTPassthroughClassifier


class _Ev:
    """鸭型事件（bamer_env.interaction.Event 同构，避免本测试反向依赖场景层）。"""

    def __init__(self, type: str, step: int, payload: dict):
        self.type, self.step, self.payload = type, step, payload


def _asset_path(name: str) -> str:
    """真实人物图像资产；缺失时生成临时合成图兜底。"""
    path = image_asset(name)
    if path:
        return path
    tmp = tempfile.mkdtemp(prefix="bamer_cam_")
    path = os.path.join(tmp, f"{name}.png")
    from PIL import Image
    Image.new("RGB", (32, 32), (200, 60, 60) if name == "quyuan" else (60, 60, 200)).save(path)
    return path


def _image_feed_ev(step: int, machine: str) -> _Ev:
    return _Ev("ImageFeed", step, {
        "machine": machine,
        "label": {"quyuan": "屈原", "zhaojun": "昭君"}[machine],
        "image": _asset_path(machine),
        "first": True,
    })


def test_gt_passthrough():
    clf = GTPassthroughClassifier()
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    assert clf.classify(img, gt_label="quyuan") == "quyuan"
    assert clf.classify(img, gt_label="zhaojun") == "zhaojun"
    try:
        clf.classify(img)
        raise AssertionError("GT 直通缺 gt_label 未报错")
    except ValueError:
        pass
    print("[GT 直通] OK：返回显式 GT；缺省报错（强制意识到走 GT 通道）")


def test_camera_unit_pipeline():
    """流水线：ImageFeed → 取图 → 分类 → Recognition；非 ImageFeed 事件忽略。"""
    unit = CameraUnit(GTPassthroughClassifier())
    events = [
        _Ev("Reported", 1, {"x": 0, "y": 0}),
        _image_feed_ev(2, "quyuan"),
        _Ev("Collision", 3, {"x": 1, "y": 0, "direction": "N"}),
        _image_feed_ev(5, "zhaojun"),
    ]
    recs = unit.on_events(events, evaluate=True)
    assert len(recs) == 2 and len(unit.frames) == 2
    r1, r2 = recs
    assert (r1.step, r1.machine, r1.predicted, r1.correct) == (2, "quyuan", "quyuan", True)
    assert (r2.step, r2.machine, r2.predicted, r2.correct) == (5, "zhaojun", "zhaojun", True)
    frame = unit.frames[0]
    assert frame.image.ndim == 3 and frame.image.dtype == np.uint8
    assert frame.gt_label == "屈原" and os.path.exists(frame.image_path)
    # 评估口径关闭时 correct=None（GT 不进决策数据流）
    rec_plain = CameraUnit(GTPassthroughClassifier()).on_events([_image_feed_ev(7, "quyuan")])
    assert rec_plain[0].correct is None and rec_plain[0].predicted == "quyuan"
    print("[流水线] OK：ImageFeed 消费/取图/识别/评估口径 GT 隔离")


def test_pluggable_classifier():
    """可插拔分类器：真实识别器实现 classify(image, *, gt_label=None) 即可挂接。"""

    class FixedClassifier:
        """假想深度分类器占位：恒输出 zhaojun，并记录收到的图像。"""

        def __init__(self):
            self.seen = []

        def classify(self, image, *, gt_label=None):    # 真实实现将忽略 gt_label
            self.seen.append(image.shape)
            return "zhaojun"

    clf = FixedClassifier()
    unit = CameraUnit(clf)
    recs = unit.on_events([_image_feed_ev(3, "quyuan")], evaluate=True)
    assert clf.seen == [unit.frames[0].image.shape]
    assert recs[0].predicted == "zhaojun" and recs[0].correct is False   # 误识也被如实记录
    print("[可插拔] OK：自定义分类器挂接成功，误识结果如实记录")


def test_missing_asset_raises():
    """图像资产缺失时不静默吞掉：显式报错（含 payload.image 为空）。"""
    unit = CameraUnit(GTPassthroughClassifier())
    try:
        unit.collect([_Ev("ImageFeed", 1, {"machine": "quyuan", "image": ""})])
        raise AssertionError("空图像引用未报错")
    except FileNotFoundError:
        pass
    try:
        unit.collect([_Ev("ImageFeed", 1, {"machine": "quyuan",
                                           "image": "Z:/no/such/file.jpeg"})])
        raise AssertionError("缺失资产未报错")
    except FileNotFoundError:
        pass
    print("[资产缺失] OK：空引用/缺失路径显式报错")


if __name__ == "__main__":
    test_gt_passthrough()
    test_camera_unit_pipeline()
    test_pluggable_classifier()
    test_missing_asset_raises()
    print("\n摄像头/识别挂接单测全部通过 [PASS]")
