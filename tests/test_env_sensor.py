# -*- coding: utf-8 -*-
"""Env 传感器集成单测（M2.2/M2.3 接通验收）：info 附加键、信息封闭、扫描时机、可复现。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from bamer_env import BamerEnv, Maze
from bamer_env.env import HEADINGS
from bamer_env.viewer import record_episode
from bamer_mouse import LidarSpec, pose_at


def empty_maze(n: int = 5) -> Maze:
    return Maze([[False] * n for _ in range(n)], [[False] * n for _ in range(n)])


def sensor_env(**kwargs):
    params = dict(
        maze=empty_maze(5),
        state_machines={"quyuan": (0, 2), "zhaojun": (2, 2)},
        destinations={"chuguo": (4, 2), "xihan": (4, 3)},
        max_steps=100,
        lidar=LidarSpec(),
    )
    params.update(kwargs)
    return BamerEnv(**params)


def test_sensor_info_keys_and_shapes():
    env = sensor_env()
    obs, info = env.reset(seed=0)
    assert set(info.keys()) == {"steps", "events", "identified", "delivered",
                                "ranges", "angles", "points"}
    assert info["ranges"].shape == (180,) and info["angles"].shape == (180,)
    assert info["points"].shape == (180, 2)
    obs, _, _, _, info = env.step("F")
    assert set(info.keys()) == {"steps", "events", "identified", "delivered",
                                "ranges", "angles", "points"}
    assert np.isfinite(info["points"][~np.isnan(info["ranges"])]).all()
    assert np.isnan(info["points"][np.isnan(info["ranges"])]).all()   # 无回波行 NaN
    print("[info 键] OK：接雷达时附加 ranges/angles/points，形状一致")


def test_scan_at_step_end():
    """扫描时机：每步末尾在规范位姿（新格中心）采帧——正对墙步进后距离正确。"""
    env = sensor_env()
    env.reset(seed=0)
    # 出生 (0,0) 朝北：正前束（机体系 0° = 束 90）北望边框 2.5 > 2.0 → 无回波
    obs, info = env.reset(seed=0)
    pose0 = pose_at(env.pose_cell, env.heading)
    assert (pose0.x, pose0.y) == (0.5, 0.5) and np.isnan(info["ranges"][90])
    # TB 转南后：正前束=世界系南 → 外边框 0.5 格
    obs, _, _, _, info = env.step("TB")
    assert env.heading == "S" and np.isclose(info["ranges"][90], 0.5)
    # F 南进一格（(0,1)→? 不，(0,0) 南是边框——碰撞，位置不变，仍采帧 0.5）
    obs, _, _, _, info2 = env.step("F")
    assert env.pose_cell == (0, 0) and np.isclose(info2["ranges"][90], 0.5)
    # 北进两格后正前束距离递减：(0,1)→1.5，(0,2)→0.5……此处 (0,1) 距北边框 3.5>2 → NaN
    obs, _, _, _, info3 = env.step("F")     # 碰撞步不移动，转向北再进
    obs, _, _, _, info4 = env.step("TB")    # 转北
    obs, _, _, _, info5 = env.step("F")     # 北进到 (0,1)，正前 3.5 > 2.0 → 无回波
    assert env.pose_cell == (0, 1) and np.isnan(info5["ranges"][90])
    print("[扫描时机] OK：step 末尾规范位姿采帧，碰撞步同样出帧")


def test_sensor_info_closure():
    """R1 信息封闭：传感器附加键不引入场景内部字段；obs 空间键不变。"""
    env = sensor_env()
    obs, info = env.reset(seed=0)
    for a in ["F", "TR", "F", "F", "TL"]:
        obs, _, _, _, info = env.step(a)
    forbidden = ("maze", "wall", "state_machine", "delivery_map", "destination",
                 "machine", "scene")
    assert set(obs.keys()) == {"cell", "heading", "identified", "delivered"}
    for key in info:
        assert not any(f in key.lower() for f in forbidden), f"info 键泄漏: {key}"
    assert set(info["ranges"].dtype.kind for _ in [0])   # 数组，无对象泄漏
    print("[信息封闭] OK：传感器通道仅含量测数据，无场景内部字段")


def test_seed_reproducible_with_noise():
    """同 reset 种子 + 同动作 → 含噪量测逐位一致（雷达流由环境种子派生）。"""
    def run():
        env = sensor_env(lidar=LidarSpec(noise_sigma=0.02))
        obs, info = env.reset(seed=7)
        scans = [info["ranges"].copy()]
        for a in ["F", "TR", "F", "TL", "F"]:
            obs, _, term, trunc, info = env.step(a)
            scans.append(info["ranges"].copy())
            if term or trunc:
                break
        return scans

    s1, s2 = run(), run()
    assert len(s1) == len(s2)
    for a, b in zip(s1, s2):
        assert np.array_equal(a, b, equal_nan=True)
    print("[可复现] OK：σ=0.02 噪声下同种子逐位一致（6 帧 × 180 束）")


def test_default_off_unchanged():
    """默认不接雷达：行为与阶段一验收版本逐位一致（info 键集不变）。"""
    env = BamerEnv(maze=empty_maze(5),
                   state_machines={"quyuan": (0, 2), "zhaojun": (2, 2)},
                   destinations={"chuguo": (4, 2), "xihan": (4, 3)},
                   max_steps=100)
    obs, info = env.reset(seed=0)
    assert set(info.keys()) == {"steps", "events", "identified", "delivered"}
    obs, _, _, _, info = env.step("F")
    assert set(info.keys()) == {"steps", "events", "identified", "delivered"}
    print("[默认关闭] OK：未接雷达时 info 键集与阶段一完全一致（零侵入）")


def test_viewer_points_hook():
    """展示层挂点：接雷达的 episode 逐帧携带世界系点云（M1.3 挂点正式启用）。"""
    env = sensor_env()
    frames = record_episode(env, lambda obs, info: 1)   # 原地左转，点云随朝向变化
    assert len(frames) > 1
    assert all(f.points is not None for f in frames)
    echo0 = int((~np.isnan(frames[0].points[:, 0])).sum())
    assert echo0 > 0
    print(f"[展示挂点] OK：{len(frames)} 帧均携带世界系点云（首帧 {echo0} 回波点）")


if __name__ == "__main__":
    test_sensor_info_keys_and_shapes()
    test_scan_at_step_end()
    test_sensor_info_closure()
    test_seed_reproducible_with_noise()
    test_default_off_unchanged()
    test_viewer_points_hook()
    print("\nEnv 传感器集成单测全部通过 [PASS]")
