# -*- coding: utf-8 -*-
"""网络单测（§8 冒烟 3）：随机权重 + 手工构造信念场景 前向 + 掩码 + argmax——选出动作不违反掩码。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from algo.actions import IDX_DELIVER, IDX_PICK, N_OPTIONS, explore_index, legal_mask
from algo.belief import CH_CARRY_QUYUAN, CH_POI_CHUGUO, CH_POI_QUYUAN, BeliefState
from algo.network import FCQNet, dueling_q

torch.manual_seed(0)


def net16():
    return FCQNet(in_channels=16, aux_head=True)


def _belief_empty_scene(n=16):
    """手工构造信念：空手、全未识别、起点已知（生产张量恒 16×16 → Explore 256 分支）。"""
    b = BeliefState(n=n)
    b.visited = {(0, 0)}
    b.cell = (0, 0)
    b.heading = "N"
    # 手工账本：(0,0) 北界开放（模拟已探明）
    b.ledger._mark(b.ledger.sensed_south, 0, 1, 1)   # (0,1) 南界 = 无墙
    return b


def test_forward_shapes():
    """前向形状：advantage (259) / value (1) / aux_logits (4,16,16)；计算图可反传。"""
    net = net16()
    x = torch.zeros((2, 16, 16, 16), dtype=torch.uint8)
    out = net(x)
    assert out["advantage"].shape == (2, N_OPTIONS)
    assert out["value"].shape == (2, 1)
    assert out["aux_logits"].shape == (2, 4, 16, 16)
    q = dueling_q(out["advantage"], out["value"], torch.ones(2, N_OPTIONS, dtype=torch.bool))
    q.sum().backward()                                   # 共享主干两路反传结构可微
    assert torch.isfinite(q).all()
    print("[前向形状] OK：三分支输出/dueling 组合/梯度可反传")


def test_dueling_mask_semantics():
    """dueling 动态掩码语义：非法动作 −inf 不进均值；Q 公式在合法集上成立。"""
    torch.manual_seed(1)
    B = 3
    adv = torch.randn(B, N_OPTIONS)
    value = torch.randn(B, 1)
    mask = torch.zeros(B, N_OPTIONS, dtype=torch.bool)
    mask[0, 5] = True; mask[0, 200] = True; mask[0, 258] = True      # 3 个合法
    mask[1, :] = True                                                # 全合法
    mask[2, 7] = True                                                # 单合法
    q = dueling_q(adv, value, mask)
    # 非法位置 = −inf
    assert torch.isinf(q[0, 0]) and q[0, 0] < 0
    # 合法位置 Q = V + A − mean_legal(A)
    for b, legal in [(0, [5, 200, 258]), (1, list(range(N_OPTIONS))), (2, [7])]:
        mean = adv[b, legal].mean()
        for a in legal:
            assert torch.isclose(q[b, a], value[b, 0] + adv[b, a] - mean, atol=1e-5)
    # argmax 永不违反掩码
    assert int(q[0].argmax()) in (5, 200, 258)
    assert int(q[2].argmax()) == 7
    print("[掩码语义] OK：−inf 屏蔽 / 均值只取合法 / argmax 不越掩码")


def test_scenes_argmax_respects_mask():
    """手工信念场景（§8 冒烟 3）：'空手全未识别' 与 '背屈原且终点已定位' 的掩码+argmax 冒烟。"""
    net = net16()
    net.eval()

    def act_legal(belief):
        T = torch.as_tensor(belief.tensor()).unsqueeze(0)
        mask = legal_mask(belief)
        m = torch.as_tensor(mask).unsqueeze(0)
        with torch.no_grad():
            out = net(T)
            q = dueling_q(out["advantage"], out["value"], m)
        a = int(q.argmax())
        assert mask[a], f"argmax 违反掩码: {a}"
        return a, mask

    # 场景 1：空手全未识别 → 只有 Explore 合法
    b1 = _belief_empty_scene()
    a1, m1 = act_legal(b1)
    assert a1 < 256 and not m1[IDX_DELIVER]
    assert not m1[IDX_PICK["quyuan"]] and not m1[IDX_PICK["zhaojun"]]
    assert m1.sum() >= 1                                  # 回溯/边界目标恒存在

    # 场景 2：背屈原 + 楚国已定位 → Deliver 合法、Pick 非法、carrying 广播通道全 1
    b2 = _belief_empty_scene()
    b2.carrying = "quyuan"
    b2.machine_known = {"quyuan": (0, 0)}
    b2.dest_known = {"chuguo": (1, 1)}
    b2.visited = {(0, 0), (1, 1)}
    b2.ledger._mark(b2.ledger.sensed_south, 1, 1, 1)
    a2, m2 = act_legal(b2)
    assert bool(m2[IDX_DELIVER]) is True                  # 持机且终点已定位
    assert not m2[IDX_PICK["quyuan"]] and not m2[IDX_PICK["zhaojun"]]   # 空手掩码
    T2 = b2.tensor()
    assert T2[CH_CARRY_QUYUAN].all()                      # carrying 全图广播
    assert T2[CH_POI_QUYUAN, 0, 0] == 1 and T2[CH_POI_CHUGUO, 1, 1] == 1
    print("[场景冒烟] OK：空手全未识别 → 仅 Explore；背屈原+终点定位 → Deliver 合法且 argmax 合规")


def test_masked_argmax_fuzz():
    """随机掩码模糊测试：200 次随机前向 argmax 全部落在合法集内。"""
    net = net16()
    rng = np.random.default_rng(0)
    x = torch.zeros((1, 16, 16, 16), dtype=torch.uint8)
    with torch.no_grad():
        out = net(x)
    for _ in range(200):
        mask = torch.as_tensor(rng.random(N_OPTIONS) < 0.3).unsqueeze(0)
        if not mask.any():
            mask[0, 0] = True
        q = dueling_q(out["advantage"], out["value"], mask)
        a = int(q.argmax())
        assert bool(mask[0, a])
    print("[模糊测试] OK：200 组随机掩码 argmax 全部合法")


def test_aux_head_and_switches():
    """辅助头形状 + unknown 掩码 BCE 冒烟；use_gru 显式未实现；14 通道消融前向。"""
    net = net16()
    x = torch.zeros((2, 16, 16, 16), dtype=torch.uint8)
    out = net(x)
    logits = out["aux_logits"]                            # (B,4,16,16)
    unknown = (x[:, :4] == 0)                             # 输入墙位=0 → 未知槽位
    gt = torch.zeros((2, 4, 16, 16))
    gt[:, 2] = 1.0                                        # 手工 GT：南墙全有
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[unknown], gt[unknown], reduction="mean")
    assert torch.isfinite(bce)
    try:
        FCQNet(use_gru=True)
        raise AssertionError("use_gru=True 未按预期报错")
    except NotImplementedError:
        pass
    net14 = FCQNet(in_channels=14, aux_head=False)
    out14 = net14(torch.zeros((1, 14, 16, 16), dtype=torch.uint8))
    assert out14["advantage"].shape == (1, N_OPTIONS) and "aux_logits" not in out14
    print("[辅助头/开关] OK：BCE 冒烟 / use_gru 预留报错 / 14 通道消融前向")


def test_receptive_field_dilated():
    """膨胀主干 RF 行为测试（2026-11-22）：距中心 Chebyshev 5 的输入扰动。

    默认主干 RF 7（半径 3）→ 扰动在窗外，中心格优势逐位不变（架构保证，与权重无关）；
    dilated=True → RF 15（半径 7）→ 扰动可达中心格，优势改变。同种子初始化仅差
    dilation 接线，权重相同。
    """
    torch.manual_seed(3)
    net_a = FCQNet(in_channels=16, aux_head=False)
    torch.manual_seed(3)
    net_b = FCQNet(in_channels=16, aux_head=False, dilated=True)
    x = torch.randint(0, 3, (1, 16, 16, 16), dtype=torch.uint8)
    x2 = x.clone()
    x2[0, 0, 8, 13] = (int(x[0, 0, 8, 13]) + 1) % 3       # 距中心 (8,8) Chebyshev 5
    center = 8 * 16 + 8                                    # explore_index((8,8)) = y*16+x
    with torch.no_grad():
        a1 = net_a(x)["advantage"][0, center]
        a2 = net_a(x2)["advantage"][0, center]
        b1 = net_b(x)["advantage"][0, center]
        b2 = net_b(x2)["advantage"][0, center]
    assert torch.equal(a1, a2), "默认 RF 7 不应受 Chebyshev 5 处扰动影响"
    assert not torch.equal(b1, b2), "膨胀 RF 15 应受 Chebyshev 5 处扰动影响"
    print("[膨胀感受野] OK：默认中心逐位不变 / 膨胀中心响应（RF 7→15）")


def test_dist_channel_forward_and_scale():
    """17 通道前向 + 逐通道归一语义：距离通道 /64、类别通道 /2；默认路径逐位回归。

    逐位回归：16ch 网络的 explore 分支输出 == 旧公式手工复算
    （x.float()/2.0 过同权重主干 → 1×1 head）——chan_scale 重构（x·0.5）与
    旧 x/2.0 在 IEEE 754 下逐元素恒等。
    """
    torch.manual_seed(4)
    net = FCQNet(in_channels=17, aux_head=True, dist_channel=True)
    x = torch.randint(0, 255, (2, 17, 16, 16), dtype=torch.uint8)
    out = net(x)
    assert out["advantage"].shape == (2, N_OPTIONS)
    scale = net._chan_scale
    assert torch.all(scale[:, :16] == 0.5) and scale[0, 16, 0, 0] == 1.0 / 64
    torch.manual_seed(5)
    net16 = FCQNet(in_channels=16, aux_head=False)
    x16 = torch.randint(0, 3, (1, 16, 16, 16), dtype=torch.uint8)
    with torch.no_grad():
        got = net16(x16)["advantage"][:, :256]            # explore 分支前 256 维
        feat = net16.backbone(x16.float() / 2.0)
        ref = net16.explore_head(feat).flatten(1)
    assert torch.equal(got, ref), "默认归一路径与旧 x/2.0 公式输出不一致"
    print("[距离通道前向] OK：17ch 形状 / 归一 /64 / 默认路径逐位一致")


if __name__ == "__main__":
    test_forward_shapes()
    test_dueling_mask_semantics()
    test_scenes_argmax_respects_mask()
    test_masked_argmax_fuzz()
    test_aux_head_and_switches()
    test_receptive_field_dilated()
    test_dist_channel_forward_and_scale()
    print("\n网络单测全部通过 [PASS]")
