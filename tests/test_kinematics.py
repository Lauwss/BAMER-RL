# -*- coding: utf-8 -*-
"""运动学单测（M2.1 验收）：原语序列终态与解析解一致；逻辑格同步正确；碰撞语义；插值。"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bamer_env import BamerEnv, Maze
from bamer_mouse import (
    Pose,
    apply_primitive,
    cell_of,
    deg_to_heading,
    heading_to_deg,
    interpolate,
    normalize_deg,
    pose_at,
    try_move,
)


def empty_maze(n: int = 5) -> Maze:
    """无内墙迷宫（外边框由 Maze 构造自动封闭）。"""
    return Maze([[False] * n for _ in range(n)], [[False] * n for _ in range(n)])


def task_env(**kwargs):
    """5×5 空迷宫任务场景（与 tests/test_env.py 同一布点，便于交叉验证）。"""
    params = dict(
        maze=empty_maze(5),
        state_machines={"quyuan": (0, 2), "zhaojun": (2, 2)},
        destinations={"chuguo": (4, 2), "xihan": (4, 3)},
        max_steps=100,
    )
    params.update(kwargs)
    return BamerEnv(**params)


def test_heading_conversion():
    assert (deg_to_heading(0), deg_to_heading(90), deg_to_heading(180),
            deg_to_heading(270)) == ("E", "N", "W", "S")
    assert deg_to_heading(360) == "E" and deg_to_heading(-90) == "S"
    assert heading_to_deg("N") == 90.0
    for h in ("N", "E", "S", "W"):
        assert deg_to_heading(heading_to_deg(h)) == h
    assert normalize_deg(-90) == 270.0 and normalize_deg(450) == 90.0
    print("[朝向换算] OK：deg↔heading 往返一致，角度归一化正确")


def test_primitive_sequence_analytic():
    """动作序列终态与解析解逐点一致（空迷宫，无碰撞）。"""
    maze = empty_maze(5)
    pose = pose_at((0, 0), "N")           # (0.5, 0.5, 90°)
    expected = [
        ((0.5, 1.5, 90.0), (0, 1)),       # F
        ((0.5, 1.5, 0.0), (0, 1)),        # TR → 东
        ((1.5, 1.5, 0.0), (1, 1)),        # F
        ((1.5, 1.5, 90.0), (1, 1)),       # TL → 北
        ((1.5, 2.5, 90.0), (1, 2)),       # F
        ((1.5, 2.5, 270.0), (1, 2)),      # TB → 南
        ((1.5, 1.5, 270.0), (1, 1)),      # F
    ]
    for action, (exp_xy_theta, exp_cell) in zip(["F", "TR", "F", "TL", "F", "TB", "F"], expected):
        pose = try_move(pose, action, maze.has_wall).pose
        assert (pose.x, pose.y, pose.theta_deg) == exp_xy_theta, f"{action} → {pose}"
        assert cell_of(pose) == exp_cell, f"{action} 逻辑格 {cell_of(pose)} ≠ {exp_cell}"
    print("[序列解析解] OK：7 步原语终态位姿/逻辑格与解析解逐点一致")


def test_consistency_with_env():
    """与 BamerEnv 交叉验证：同动作序列下 kinematics 逻辑格/朝向与 Env 逐步一致。"""
    maze = empty_maze(5)
    env = task_env()
    env.reset(seed=0)
    pose = pose_at(env.pose_cell, env.heading)
    for action in ["F", "TR", "F", "F", "TL", "F", "TB", "F", "TR", "F"]:
        out = try_move(pose, action, maze.has_wall)
        pose = out.pose
        env.step(action)
        assert cell_of(pose) == env.pose_cell, \
            f"{action}: kinematics {cell_of(pose)} ≠ env {env.pose_cell}"
        assert pose.heading == env.heading, f"{action}: 朝向不一致"
    print("[Env 交叉验证] OK：10 步动作逻辑格与朝向和 BamerEnv 逐步一致")


def test_turn_semantics():
    """转向原语：TL=+90°(左转 N→W)、TR=-90°(右转 N→E)、TB=180°(N→S)，与 Env 一致。"""
    maze = empty_maze(5)
    for h0, h_tl, h_tr, h_tb in [("N", "W", "E", "S"), ("E", "N", "S", "W"),
                                 ("S", "E", "W", "N"), ("W", "S", "N", "E")]:
        pose = pose_at((2, 2), h0)
        assert try_move(pose, "TL", maze.has_wall).pose.heading == h_tl
        assert try_move(pose, "TR", maze.has_wall).pose.heading == h_tr
        assert try_move(pose, "TB", maze.has_wall).pose.heading == h_tb
    # 旋转不改变位置
    pose = try_move(pose_at((2, 2), "N"), "TB", maze.has_wall).pose
    assert (pose.x, pose.y) == (2.5, 2.5)
    print("[转向语义] OK：TL/TR/TB 与 Env 朝向机一致，旋转不改位置")


def test_collision_wall():
    """碰撞：F 指向有墙边界 → 位置不变 + moved=False + 撞击方向；无墙方向正常前进。"""
    maze = empty_maze(5)
    maze.wall_south[1][2] = True        # (1,1) 与 (1,2) 之间加墙（北侧）
    pose = pose_at((1, 1), "N")
    out = try_move(pose, "F", maze.has_wall)
    assert not out.moved and out.direction == "N"
    assert pose == out.pose             # 位置不变
    out = try_move(pose, "TR", maze.has_wall)      # 转东，无墙
    assert out.moved and out.direction is None
    out = try_move(out.pose, "F", maze.has_wall)
    assert out.moved and cell_of(out.pose) == (2, 1)
    # 外边框碰撞
    out = try_move(pose_at((0, 0), "W"), "F", maze.has_wall)
    assert not out.moved and out.direction == "W"
    print("[碰撞] OK：位置不变 + moved=False + 方向；边框与内墙均拦截")


def test_apply_primitive_pure():
    """纯几何演化：旋转任意角可用；F 沿朝向严格推进 1.0 格。"""
    pose = Pose(2.5, 2.5, 45.0)
    p = apply_primitive(pose, "F")
    assert math.isclose(p.x, 2.5 + math.cos(math.radians(45)), abs_tol=1e-12)
    assert math.isclose(p.y, 2.5 + math.sin(math.radians(45)), abs_tol=1e-12)
    p = apply_primitive(pose, "TL")
    assert p.theta_deg == 135.0 and (p.x, p.y) == (2.5, 2.5)
    for bad in ("X", "FF", ""):
        try:
            apply_primitive(pose, bad)
            raise AssertionError(f"未知原语 {bad!r} 未报错")
        except ValueError:
            pass
    print("[纯几何] OK：F 沿朝向推进 1.0 格；非法原语报错")


def test_interpolate():
    """匀速插值：端点=解析解，中点线性；F 平移 / 转向原地旋转；t 越界报错。"""
    pose0 = pose_at((0, 0), "N")
    # F：位置线性推进
    p = interpolate(pose0, "F", 0.5)
    assert (p.x, p.y, p.theta_deg) == (0.5, 1.0, 90.0)
    assert interpolate(pose0, "F", 1.0) == apply_primitive(pose0, "F")
    assert interpolate(pose0, "F", 0.0) == pose0
    # TL：原地匀速旋转（90° → t=0.5 时 135°）
    p = interpolate(pose0, "TL", 0.5)
    assert (p.x, p.y) == (0.5, 0.5) and p.theta_deg == 135.0
    assert interpolate(pose0, "TB", 1.0).theta_deg == 270.0
    for t in (-0.1, 1.1):
        try:
            interpolate(pose0, "F", t)
            raise AssertionError(f"t={t} 未报错")
        except ValueError:
            pass
    print("[插值] OK：端点/中点正确，F 平移与转向旋转分离")


def test_cell_of_floor_semantics():
    """逻辑格 = floor(位姿)：格中心位姿映射正确，边界位姿按 floor 落格。"""
    assert pose_at((3, 2), "E").cell == (3, 2)
    assert cell_of(Pose(1.0, 0.5, 0.0)) == (1, 0)      # 恰在格边界上归右格
    assert cell_of(Pose(0.999, 2.5, 0.0)) == (0, 2)
    print("[逻辑格] OK：floor 语义与格中心映射正确")


if __name__ == "__main__":
    test_heading_conversion()
    test_primitive_sequence_analytic()
    test_consistency_with_env()
    test_turn_semantics()
    test_collision_wall()
    test_apply_primitive_pure()
    test_interpolate()
    test_cell_of_floor_semantics()
    print("\n运动学单测全部通过 [PASS]")
