# -*- coding: utf-8 -*-
"""信念层单测（§8 冒烟 1）：受控场景派生张量与手算对拍 + R1 自检（不含 GT 的输入流）。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from algo.belief import (
    CH_ALIGN, CH_CARRY_QUYUAN, CH_DIST, CH_POSE_N, CH_POSE_PREV_BOUNDARY, CH_POI_CHUGUO,
    CH_POI_QUYUAN, CH_TRAIL_PREV_OPTION, CH_WALL_E, CH_WALL_N, CH_WALL_S, CH_WALL_W,
    BeliefState, DIST_CAP, GTAssetClassifier, HEADINGS, SensingLedger, W_FREE, W_UNKNOWN, W_WALL,
)
from bamer_env import BamerEnv, Maze
from bamer_mouse.lidar import LidarSpec


def empty_maze(n: int) -> Maze:
    return Maze([[False] * n for _ in range(n)], [[False] * n for _ in range(n)])


def scan_env(n: int = 3, **kwargs):
    """接雷达的受控环境（测试扮演环境侧角色；belief 只见 obs/info）。"""
    params = dict(
        maze=empty_maze(n),
        state_machines={"quyuan": (0, 2), "zhaojun": (2, 2)},
        destinations={"chuguo": (4, 2), "xihan": (4, 3)} if n >= 5 else {"chuguo": (2, 0), "xihan": (2, 1)},
        max_steps=200,
        lidar=LidarSpec(),
    )
    params.update(kwargs)
    return BamerEnv(**params)


def test_ledger_handcheck_reset_scan():
    """探测账本手算对拍：3×3 空迷宫 (0,0) 朝北 reset 扫描（量程 2.0，180 束）。

    手算：北向射线穿越 N 边界→(0,1)→(1?) 不，北向沿 x=0 列：south[0][1]、south[0][2]
    探明无墙；南/西边界即外框→有墙；东向射线探明 west[1][0]、west[2][0]；
    斜向射线探明 west[1][1] 等；北外行/东外列槽位未被穿越→unknown。
    """
    env = scan_env(3)
    obs, info = env.reset(seed=0)
    belief = BeliefState(n=3)
    belief.observe(obs, info)
    led = belief.ledger
    # 南/西外边框：正南/正西射线 0.5 处命中外框 → 有墙(2)
    assert led.wall_value(0, 0, "S") == W_WALL and led.wall_value(0, 0, "W") == W_WALL
    # 正北射线穿越：(0,0)-(0,1)、(0,1)-(0,2) 边界无墙
    assert led.wall_value(0, 0, "N") == W_FREE
    assert led.wall_value(0, 1, "S") == W_FREE and led.wall_value(0, 1, "N") == W_FREE
    # 正东射线穿越：west[1][0]、west[2][0]
    assert led.wall_value(0, 0, "E") == W_FREE and led.wall_value(1, 0, "W") == W_FREE
    assert led.wall_value(1, 0, "E") == W_FREE
    # 斜向射线（44°/46°族）探明 west[1][1]（从 (0,1) 东行）
    assert led.wall_value(0, 1, "E") == W_FREE
    # 未被任何射线穿越的槽位 = unknown（悲观：不可通行）
    assert led.wall_value(2, 2, "N") == W_UNKNOWN or led.sensed_south[2, 2] == W_UNKNOWN
    print("[账本手算] OK：reset 扫描的边界感知记录与解析射线穿越一致")


def test_ledger_monotone_wall_sticky():
    """账本只增不减：无墙观测可升级为有墙（悲观），有墙观测不可降级。"""
    led = SensingLedger(4)
    led._mark(led.sensed_south, 1, 1, W_FREE)
    assert led.sensed_south[1, 1] == W_FREE
    led._mark(led.sensed_south, 1, 1, W_WALL)
    assert led.sensed_south[1, 1] == W_WALL      # 有墙覆盖无墙
    led._mark(led.sensed_south, 1, 1, W_FREE)
    assert led.sensed_south[1, 1] == W_WALL      # 有墙不可降级
    # Collision 揭示墙位（合法事件通道）
    led.mark_wall(2, 2, "N")
    assert led.wall_value(2, 2, "N") == W_WALL
    led.mark_wall(3, 3, "E")                     # 东外边框越界保护
    assert led.wall_value(3, 3, "E") == W_WALL
    print("[账本单调] OK：只增不减 / 有墙优先 / 碰撞揭示")


def test_tensor_channels_handcheck():
    """16 通道派生手算对拍：位姿 one-hot / POI / carrying 广播 / 轨迹 / 墙位派生。"""
    env = scan_env(5, state_machines={"quyuan": (0, 2), "zhaojun": (2, 2)},
                   destinations={"chuguo": (4, 2), "xihan": (4, 3)})
    obs, info = env.reset(seed=0)
    belief = BeliefState(n=5)
    belief.observe(obs, info)

    # 边界 0：全未识别 / 空手 → POI 与 carrying 通道全零（全零=未知约定）
    T = belief.tensor()
    assert T.shape == (16, 5, 5)
    assert T[CH_POI_QUYUAN].sum() == 0 and T[CH_POI_CHUGUO].sum() == 0
    assert T[CH_CARRY_QUYUAN].sum() == 0
    # 位姿 one-hot：cell=y*5+x=(0,0) 朝北 → CH_POSE_N 在 (0,0)
    assert T[CH_POSE_N, 0, 0] == 1 and T[CH_POSE_N].sum() == 1
    # 首边界：轨迹通道全零
    assert T[CH_TRAIL_PREV_OPTION].sum() == 0 and T[CH_POSE_PREV_BOUNDARY].sum() == 0

    # 走到屈原格 (0,2)：ImageFeed + Pickup → 识别位标记 + carrying 全图广播
    for act in ["F", "F"]:
        obs, _, _, _, info = env.step(act)
        belief.observe(obs, info)
    assert belief.carrying == "quyuan" and belief.machine_known["quyuan"] == (0, 2)
    T = belief.tensor()
    assert T[CH_POI_QUYUAN, 0, 2] == 1                      # 已识别未送达
    assert T[CH_CARRY_QUYUAN].sum() == 25 and T[CH_CARRY_QUYUAN].all()
    assert T[CH_POSE_N, 0, 2] == 1

    # commit_option → 轨迹通道
    belief.commit_option([(0, 0), (0, 1), (0, 2)], (0, 0))
    T = belief.tensor()
    assert T[CH_TRAIL_PREV_OPTION, 0, 0] == 1 and T[CH_TRAIL_PREV_OPTION, 0, 2] == 1
    assert T[CH_TRAIL_PREV_OPTION].sum() == 3
    assert T[CH_POSE_PREV_BOUNDARY, 0, 0] == 1

    # use_traj=False → 14 通道消融
    T14 = belief.tensor(use_traj=False)
    assert T14.shape == (14, 5, 5)
    assert np.array_equal(T14, T[:14])
    print("[张量派生] OK：位姿/POI/carrying 广播/轨迹/墙位通道与手算一致，14 通道裁剪正确")


def test_tensor_delivered_and_destination():
    """送达后机器 POI=2（在该机已知格）；终点发现后 dest 通道=1（发现格）。"""
    env = scan_env(5, state_machines={"quyuan": (0, 2), "zhaojun": (2, 2)},
                   destinations={"chuguo": (4, 2), "xihan": (4, 3)})
    obs, info = env.reset(seed=0)
    belief = BeliefState(n=5)
    belief.observe(obs, info)
    # 屈原(0,2) → 楚国(4,2)
    for act in ["F", "F", "TR", "F", "F", "F", "F"]:
        obs, _, _, _, info = env.step(act)
        belief.observe(obs, info)
    assert belief.carrying is None and "quyuan" in belief.machine_delivered
    T = belief.tensor()
    assert T[CH_POI_QUYUAN, 0, 2] == 2                       # 已送达
    assert T[CH_POI_CHUGUO, 4, 2] == 1                       # 终点已定位
    assert T[CH_CARRY_QUYUAN].sum() == 0                     # 空手广播归零
    print("[送达/终点] OK：POI=2 送达态 / 终点发现格标记 / 空手广播归零")


def test_r1_selfcheck_gt_free_stream():
    """R1 自检（§8 冒烟 1）：纯合成输入流（不含任何 GT 对象）派生确定性张量。"""
    # 构造不含 GT 的输入流：量测由测试侧内联 wall_query 生成（模拟环境职责），
    # belief 只见 ranges/angles/obs/events 字典
    from bamer_mouse.lidar import Lidar, LidarSpec
    from bamer_mouse.pointcloud import scan_to_world

    def wall_query(x, y, direction):
        return x == 0 or y == 0 or x == 4 or y == 4    # 空房间四周有框（内联谓词，非 Maze）

    belief = BeliefState(n=5)
    lidar = Lidar(LidarSpec())
    tensors = []
    for cell, heading in [((1, 1), "N"), ((2, 1), "E"), ((2, 2), "S")]:
        from bamer_mouse.kinematics import pose_at
        scan = lidar.scan(wall_query, pose_at(cell, heading))
        obs = {"cell": cell[1] * 5 + cell[0], "heading": HEADINGS.index(heading)}
        info = {"steps": 0, "events": [], "ranges": scan.ranges, "angles": scan.angles}
        belief.observe(obs, info)
        tensors.append(belief.tensor().copy())
    # 纯度：同一输入流重放（重置后）→ 张量逐位一致（派生是合法输入的纯函数）
    belief2 = BeliefState(n=5)
    lidar2 = Lidar(LidarSpec())
    replayed = []
    for cell, heading in [((1, 1), "N"), ((2, 1), "E"), ((2, 2), "S")]:
        scan = lidar2.scan(wall_query, pose_at(cell, heading))
        obs = {"cell": cell[1] * 5 + cell[0], "heading": HEADINGS.index(heading)}
        info = {"steps": 0, "events": [], "ranges": scan.ranges, "angles": scan.angles}
        belief2.observe(obs, info)
        replayed.append(belief2.tensor().copy())
    for a, b in zip(tensors, replayed):
        assert np.array_equal(a, b)
    # 派生值与内联谓词一致：房间中央四周均无墙（已探明）
    T = tensors[-1]
    assert T[CH_WALL_S, 2, 2] == W_FREE and T[CH_WALL_E, 2, 2] == W_FREE
    # 结构性 R1：belief 模块的 import 不触碰场景内部（允许 bamer_env.interaction 资产常量）
    src_path = os.path.join(os.path.dirname(__file__), "..", "algo", "belief.py")
    with open(src_path, encoding="utf-8") as f:
        src = f.read()
    import_lines = [ln.strip() for ln in src.splitlines()
                    if ln.strip().startswith(("import ", "from "))]
    assert not any("bamer_env.scene" in ln or "maze" in ln.lower() for ln in import_lines)
    assert not any("Maze" in ln for ln in import_lines)
    print("[R1 自检] OK：不含 GT 的输入流跑通 / 派生纯函数 / import 无场景内部引用")


def test_gt_asset_classifier():
    """GT 直通占位分类器：仅消费图像资产引用（合法字段）反查身份。"""
    cls = GTAssetClassifier()
    from bamer_env.interaction import DESTINATION_ASSETS, IMAGE_ASSETS
    assert cls.classify(IMAGE_ASSETS["quyuan"]) == "quyuan"
    assert cls.classify(IMAGE_ASSETS["zhaojun"]) == "zhaojun"
    assert cls.classify(DESTINATION_ASSETS["chuguo"]) == "chuguo"
    assert cls.classify(DESTINATION_ASSETS["xihan"]) == "xihan"
    try:
        cls.classify("")
        raise AssertionError("空资产引用未报错")
    except ValueError:
        pass
    print("[GT 分类器] OK：资产引用→身份（机器/终点），缺引用报错")


def test_distance_field_handcheck():
    """距离场手算对拍（2026-11-22 距离通道）：走廊/折行支路/围死区域。

    图与可达语义与 actions.reachable_cells 同源（开放边界=已探明无墙、unknown
    不可通行的悲观约定）。槽位记账约定：sensed_west[x,y] = (x,y) 西界（即
    (x-1,y)-(x,y) 边界）、sensed_south[x,y] = (x,y) 南界。
    """
    b = BeliefState(n=5, use_dist=True)
    b.cell = (0, 0)
    led = b.ledger
    for x in range(1, 5):                     # 东向走廊 (0,0)-(1,0)-(2,0)-(3,0)-(4,0)
        led._mark(led.sensed_west, x, 0, W_FREE)
    led._mark(led.sensed_south, 2, 1, W_FREE)  # (2,0)-(2,1) 边界开放（(2,1) 南界）
    led._mark(led.sensed_west, 3, 1, W_FREE)   # (2,1)-(3,1) 边界开放
    d = b.distance_field()
    assert d[0, 0] == 0                        # 当前格 = 0
    for x, want in [(1, 1), (2, 2), (3, 3), (4, 4)]:
        assert d[x, 0] == want, f"走廊距离错: ({x},0)={d[x, 0]}≠{want}"
    assert d[2, 1] == 3 and d[3, 1] == 4       # 支路经 (2,0) 折行
    assert d[2, 2] == DIST_CAP                 # 围死（四界全 unknown）→ 饱和
    assert d[4, 4] == DIST_CAP                 # 不可达 → 饱和（与"极远"同值，恒被掩码排除）
    print("[距离场手算] OK：走廊/支路折行/围死饱和（悲观可达语义）")


def test_tensor_dist_channel():
    """17 通道派生：前 16 通道与旧 16ch 张量逐位一致；末通道 = 距离场；use_traj=False 组合。"""
    def build(use_dist):
        b = BeliefState(n=5, use_dist=use_dist)
        b.cell = (0, 0)
        b.ledger._mark(b.ledger.sensed_west, 1, 0, W_FREE)
        b.ledger._mark(b.ledger.sensed_south, 0, 1, W_FREE)
        return b

    b, b16 = build(True), build(False)
    T17, T16 = b.tensor(), b16.tensor()
    assert T17.shape == (17, 5, 5) and T16.shape == (16, 5, 5)
    assert np.array_equal(T17[:16], T16)       # 前缀逐位一致（默认路径不受影响）
    assert np.array_equal(T17[CH_DIST], b.distance_field())
    assert T17[CH_DIST, 0, 0] == 0 and T17[CH_DIST, 1, 0] == 1 and T17[CH_DIST, 2, 2] == DIST_CAP
    T15 = b.tensor(use_traj=False)
    assert T15.shape == (15, 5, 5)             # 14 基础 + 距离（无轨迹）
    assert np.array_equal(T15[:14], T16[:14])
    assert np.array_equal(T15[14], T17[CH_DIST])
    print("[距离通道张量] OK：前缀逐位一致 / 末通道=距离场 / 15ch 组合")


def test_alignment_field_handcheck():
    """对齐度场手算对拍（2026-11-22 方向连续性特征）：正前/回头/垂直/对角 + 未定义情形。

    参考向量 v = cell − prev_boundary_cell（上个 option 净位移）。编码
    round((cos+1)·127)+1：+1→255、0→128、−1→1；0 = 未定义（全零=未知约定）。
    """
    b = BeliefState(n=10, use_dir=True)
    b.cell = (2, 5)
    b.prev_boundary_cell = (2, 2)              # v = (0, +3)，朝北
    f = b.alignment_field()
    assert f[2, 9] == 255                      # 正前 cos=+1（沿轴任意距离同值）
    assert f[2, 0] == 1                        # 完全回头 cos=−1
    assert f[6, 5] == 128                      # 垂直 cos=0（正东）
    assert f[5, 9] == 230                      # 3-4-5 对角 cos=12/15=0.8 → round(228.6)+1
    assert f[2, 5] == 0                        # 当前格自身：无方向语义 → 未定义
    print("[对齐场手算] OK：255/1/128/对角 230/当前格 0")


def test_alignment_field_undefined():
    """未定义情形 → 全零通道（episode 首边界 / 零位移 option）。"""
    b = BeliefState(n=6, use_dir=True)
    assert not b.alignment_field().any()       # 首边界：prev_boundary_cell=None
    b.cell = (3, 3)
    b.prev_boundary_cell = (3, 3)              # 零位移（到位型 Pick 等）
    assert not b.alignment_field().any()
    print("[对齐场未定义] OK：首边界/零位移 全零 = 未知约定")


def test_tensor_dir_channel():
    """通道布局组合：仅 dir 17ch / dist+dir 18ch / use_traj=False 15ch；旧前缀逐位不变。"""
    b = BeliefState(n=5, use_dir=True)
    b.cell = (2, 2)
    b.prev_boundary_cell = (2, 0)              # v = (0, +2) 朝北
    b0 = BeliefState(n=5)                      # 同位姿对照（含 prev 字段，保证前缀可比）
    b0.cell = (2, 2)
    b0.prev_boundary_cell = (2, 0)
    T17, T16 = b.tensor(), b0.tensor()
    assert T17.shape == (17, 5, 5) and T16.shape == (16, 5, 5)
    assert np.array_equal(T17[:16], T16)       # 前缀逐位一致（默认路径不受影响）
    assert T17[-1, 2, 3] == 255 and T17[-1, 2, 2] == 0    # 末通道 = 对齐场
    b18 = BeliefState(n=5, use_dist=True, use_dir=True)
    b18.cell = (2, 2)
    b18.prev_boundary_cell = (2, 0)
    T18 = b18.tensor()
    assert T18.shape == (18, 5, 5)
    b0d = BeliefState(n=5, use_dist=True)
    b0d.cell = (2, 2)
    b0d.prev_boundary_cell = (2, 0)
    assert np.array_equal(T18[:17], b0d.tensor())   # dist@16 与是否带 dir 无关（布局稳定）
    assert T18[CH_DIST, 2, 2] == 0 and T18[CH_DIST, 4, 4] == DIST_CAP
    assert T18[CH_ALIGN, 2, 3] == 255          # align@17 恒居末位
    T15 = b.tensor(use_traj=False)
    assert T15.shape == (15, 5, 5)             # 14 基础 + 对齐场（无轨迹）
    assert np.array_equal(T15[:14], T16[:14])
    assert np.array_equal(T15[14], T17[-1])
    print("[对齐通道张量] OK：17/18/15ch 组合 + 前缀逐位一致 + dist 布局稳定")


if __name__ == "__main__":
    test_ledger_handcheck_reset_scan()
    test_ledger_monotone_wall_sticky()
    test_tensor_channels_handcheck()
    test_tensor_delivered_and_destination()
    test_r1_selfcheck_gt_free_stream()
    test_gt_asset_classifier()
    test_distance_field_handcheck()
    test_tensor_dist_channel()
    test_alignment_field_handcheck()
    test_alignment_field_undefined()
    test_tensor_dir_channel()
    print("\n信念层单测全部通过 [PASS]")
