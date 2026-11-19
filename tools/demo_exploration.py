# -*- coding: utf-8 -*-
"""M1.3 验收演示：默认场景完整探索录像（GIF，验收要求"输出 1 段完整探索录像"）。

用法：
    python tools/demo_exploration.py                     # 默认参数
    python tools/demo_exploration.py --steps 800 --seed 3

策略为展示层演示用占位随机漫步（只用机器鼠合法自有信息：自身位姿 + 碰撞事件
记忆），不是算法基线——正式策略由阶段三 RL 产出（AGENT.md 3.3）。
点云未接通（阶段二 M2.2/M2.3），按 PLAN.md 允许先无点云录像。
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bamer_env import BamerEnv
from bamer_env.env import HEADINGS
from bamer_env.interaction import (
    EVENT_ARRIVED,
    EVENT_COLLISION,
    EVENT_IMAGE_FEED,
    EVENT_TASK_COMPLETE,
    EVENT_TIMEOUT,
)
from bamer_env.viewer import EpisodeAnimator, record_episode

ROOT = os.path.join(os.path.dirname(__file__), "..")


class DemoWalkPolicy:
    """有偏随机漫步占位策略：已撞过的方向不再直行（碰撞记忆 = 自身事件反馈，合规）。"""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.blocked = set()   # {(cell_idx, heading_idx)} 已撞墙方向

    def __call__(self, obs, info):
        cell, heading = obs["cell"], obs["heading"]
        for ev in info.get("events", ()):
            if ev.type == EVENT_COLLISION:
                self.blocked.add((cell, HEADINGS.index(ev.payload["direction"])))
        if (cell, heading) not in self.blocked and self.rng.random() < 0.8:
            return 0           # F
        r = self.rng.random()
        return 1 if r < 0.45 else (2 if r < 0.9 else 3)   # TL / TR / TB


def main():
    ap = argparse.ArgumentParser(description="M1.3 探索动画演示")
    ap.add_argument("--steps", type=int, default=1500, help="episode 步数上限")
    ap.add_argument("--seed", type=int, default=7, help="策略随机种子（环境场景为默认 PPT 场景）")
    ap.add_argument("--stride", type=int, default=5, help="GIF 抽帧间隔（每 stride 步取 1 帧）")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "_demo_exploration.gif"))
    args = ap.parse_args()

    env = BamerEnv(max_steps=args.steps)
    frames = record_episode(env, DemoWalkPolicy(seed=args.seed))
    animator = EpisodeAnimator(env.scene, frames, max_steps=env.max_steps,
                               title="机器鼠迷宫探索（M1.3 演示）")
    animator.save_gif(args.out, fps=args.fps, stride=args.stride, dpi=95)

    # 事件时间线（剔除每步必发的 Reported）
    print(f"探索录像已保存: {args.out}（{len(frames)} 步，抽帧 stride={args.stride}, fps={args.fps}）")
    print("\n事件时间线：")
    for f in frames:
        for ev in f.events:
            if ev.type != "Reported":
                print(f"  {ev}")
    final = frames[-1]
    print("\n汇总：")
    print(f"  总步数: {final.step} / {env.max_steps}")
    print(f"  事件统计: ImageFeed={sum(1 for f in frames for e in f.events if e.type == EVENT_IMAGE_FEED)}"
          f"  Arrived={sum(1 for f in frames for e in f.events if e.type == EVENT_ARRIVED)}"
          f"  Collision={sum(1 for f in frames for e in f.events if e.type == EVENT_COLLISION)}")
    done = any(e.type == EVENT_TASK_COMPLETE for f in frames for e in f.events)
    print(f"  任务达成: {'是（TaskComplete）' if done else '否（占位随机策略无寻的，正式策略见阶段三 RL）'}")
    print(f"  文件大小: {os.path.getsize(args.out) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
