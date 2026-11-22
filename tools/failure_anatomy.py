# -*- coding: utf-8 -*-
"""失败样例解剖：held-out 逐 seed 失败分类 + 场景/轨迹特征 + 代表样例渲染（2026-11-22）。

问题：fixgen 与 random 的 held-out 成功率都在 ~2/3 处封顶，失败样例是哪些、
为什么失败？本工具在固定 seed 套件上贪心评估并给每条 episode 记录：

  失败桶（互斥，按序判定）：
    success        全任务完成（terminated）
    deadlock       连续 8 次 option 不可达中止（合法集异常/规划失败）
    delivery_fail  4 POI 全识别但未 terminated（携带/送达未完成）
    blindspot      有未识别 POI，且该格到 episode 结束全部墙槽已探明、从未
                   到访（"识别须进格" × "全信息格不在 frontier" 的激励盲区：
                   探索与 ε 均不会进入零信息格）
    budget_open    未识别 POI 的格仍有未知墙槽（前沿未闭合）——纯搜索预算/
                   效率不足
    anomaly        其余（防御桶，应为空）

  每 seed 记录：步数/option 数、search_steps、识别/送达进度、末端 frontier /
  可达 / 到访 / 墙知识率、最后新格步、最后前沿收缩步、全知参照步数、各 POI
  的 GT 死胡同属性与末端信息状态；失败样例渲染轨迹 PNG。

用法：
    python tools/failure_anatomy.py --run-dir algo/runs/armA_frontier_dist_dilate \
        [--checkpoint checkpoint_stage1_final.pt] [--n-seeds 100] [--max-steps 1500]
        [--seed-base 777] [--render 8] [--out-dir output/failure_anatomy]

跨 run 对比：对每个 run 各跑一次，用 --compare output/failure_anatomy 汇总
（失败 seed 重叠矩阵 + 桶分布对照）。
"""
import argparse
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

CLASS_ORDER = ("success", "blindspot", "delivery_fail", "deadlock", "budget_open", "anomaly")

CSV_FIELDS = [
    "seed", "outcome", "total_steps", "options", "search_steps", "ref_steps",
    "n_machine_known", "n_dest_known", "n_delivered", "carrying_end",
    "explored_ratio", "wall_knowledge_ratio", "frontier_end", "reachable_end",
    "visited_end", "steps_last_new_cell", "steps_last_frontier_shrink",
    "unidentified", "poi_info_dead_end", "poi_flags", "abort_total",
]


def gt_dead_end(maze, cell) -> bool:
    """GT 迷宫该格开放边界数 =1 → 死胡同（相对全图拓扑）。"""
    return sum(1 for d in ("N", "E", "S", "W") if not maze.has_wall(cell[0], cell[1], d)) == 1


def poi_status(belief, maze, name, cell, identified, frontier_end, reach_end):
    """POI 末端状态串：是否死胡同 / 信息死格（全槽已知未到访）/ 仍在前沿。"""
    slots_known = all(belief.ledger.wall_value(cell[0], cell[1], d) != 0
                      for d in ("N", "E", "S", "W"))
    visited = cell in belief.visited
    dead = gt_dead_end(maze, cell)
    state = "id" if identified else ("vis" if visited else
                                     ("front" if cell in frontier_end else
                                      ("infodead" if slots_known else "unknown")))
    return f"{name}@{cell[0]},{cell[1]}:{state}{'|DE' if dead else ''}"


def run_episode(agent, env, belief, executor, seed, use_traj, frontier_only,
                fallback_unvisited=False):
    """带插桩的贪心 episode；返回 (指标 dict, 轨迹, 分类所需末端快照)。"""
    from algo.actions import frontier_cells, legal_mask, reachable_cells

    obs, info = env.reset(seed=seed)
    belief.reset(env.scene.maze.n)
    belief.observe(obs, info)
    steps = 0
    options = 0
    abort_total = 0
    abort_guard = 0
    deadlocked = False
    search_steps = None
    traj = [belief.cell]
    visited_prev = 1
    frontier_prev = len(frontier_cells(belief))
    steps_last_new_cell = 0
    steps_last_frontier_shrink = 0
    ident_step: dict = {}

    while True:
        s = belief.tensor(use_traj=use_traj)
        mask = legal_mask(belief, frontier_only=frontier_only,
                          fallback_unvisited=fallback_unvisited)
        action = agent.act(s, mask, greedy=True)
        result = executor.run(env, action)
        options += 1
        steps += result.steps
        traj.extend(result.path[1:])
        belief.commit_option(result.path, result.start_cell)
        for name in ("quyuan", "zhaojun"):
            if name in belief.machine_known and name not in ident_step:
                ident_step[name] = steps
        for name in ("chuguo", "xihan"):
            if name in belief.dest_known and name not in ident_step:
                ident_step[name] = steps
        if search_steps is None and belief.poi_complete():
            search_steps = steps
        if len(belief.visited) > visited_prev:
            visited_prev = len(belief.visited)
            steps_last_new_cell = steps
        fron = frontier_cells(belief)
        if len(fron) < frontier_prev:
            frontier_prev = len(fron)
            steps_last_frontier_shrink = steps
        abort_total += int(result.steps == 0)
        if result.steps == 0:
            abort_guard += 1
            if abort_guard >= 8:
                deadlocked = True
                break
        else:
            abort_guard = 0
        if result.terminated or result.truncated:
            break

    return {
        "deadlocked": deadlocked,
        "seed": seed, "total_steps": steps, "options": options,
        "search_steps": search_steps,
        "n_machine_known": len(belief.machine_known),
        "n_dest_known": len(belief.dest_known),
        "n_delivered": len(belief.machine_delivered),
        "carrying_end": belief.carrying or "",
        "terminated": bool(result.terminated), "truncated": bool(result.truncated),
        "explored_ratio": len(belief.visited) / (belief.n * belief.n),
        "wall_knowledge_ratio": 1.0 - belief.ledger.unknown_count() / (2 * belief.n * belief.n),
        "frontier_end": frontier_prev, "reachable_end": len(reachable_cells(belief, belief.cell)),
        "visited_end": len(belief.visited),
        "steps_last_new_cell": steps_last_new_cell,
        "steps_last_frontier_shrink": steps_last_frontier_shrink,
        "abort_total": abort_total,
        "ident_step": ident_step,
        "end_cell": belief.cell,
        "frontier_set": fron,
    }, traj


def classify(metrics, belief, maze, frontier_end, scene):
    """失败桶判定（互斥，按序）。同时产出 unidentified / poi 字段。"""
    machines = dict(scene.state_machines)
    dests = dict(scene.destinations)
    rows, info_dead = [], 0
    for name, cell in list(machines.items()) + list(dests.items()):
        identified = (name in belief.machine_known) or (name in belief.dest_known)
        status = poi_status(belief, maze, name, cell, identified,
                            frontier_end, set())
        rows.append(status)
        if not identified and cell not in belief.visited and \
                all(belief.ledger.wall_value(cell[0], cell[1], d) != 0
                    for d in ("N", "E", "S", "W")):
            info_dead += 1
    fields = {
        "unidentified": ";".join(rows),
        "poi_info_dead_end": info_dead,
        "poi_flags": "",
    }
    if metrics["terminated"]:
        fields["outcome"] = "success"
    elif metrics["deadlocked"]:
        fields["outcome"] = "deadlock"
    elif metrics["search_steps"] is not None:
        fields["outcome"] = "delivery_fail"
    elif info_dead > 0:
        fields["outcome"] = "blindspot"
    elif frontier_end:
        fields["outcome"] = "budget_open"
    else:
        fields["outcome"] = "anomaly"
    return fields


def render(out_png, maze, scene, belief, traj, frontier_end, metrics, fields, max_steps):
    """单样例轨迹渲染：墙线 + 到访灰底 + 轨迹蓝线 + POI/起终点/前沿标注。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = maze.n
    fig, ax = plt.subplots(figsize=(7, 7))
    for x in range(n):
        for y in range(n):
            if maze.has_wall(x, y, "S"):
                ax.plot([x, x + 1], [y, y], "k-", lw=1)
            if maze.has_wall(x, y, "W"):
                ax.plot([x, x], [y, y + 1], "k-", lw=1)
    for x in range(n):
        if maze.has_wall(x, n - 1, "N"):
            ax.plot([x, x + 1], [n, n], "k-", lw=1)
    for y in range(n):
        if maze.has_wall(n - 1, y, "E"):
            ax.plot([n, n], [y, y + 1], "k-", lw=1)
    for (x, y) in belief.visited:
        ax.add_patch(plt.Rectangle((x, y), 1, 1, color="0.88", zorder=0))
    for (x, y) in frontier_end:
        ax.add_patch(plt.Rectangle((x, y), 1, 1, color="#ffcccc", zorder=1))
    tx = [c[0] + 0.5 for c in traj]
    ty = [c[1] + 0.5 for c in traj]
    ax.plot(tx, ty, "b-", lw=1.2, alpha=0.75, zorder=3)
    for name, cell in dict(scene.state_machines).items():
        known = name in belief.machine_known
        ax.plot(cell[0] + 0.5, cell[1] + 0.5, "s",
                color="limegreen" if known else "red", ms=11, zorder=4)
        ax.annotate(name[0].upper(), (cell[0] + 0.5, cell[1] + 0.5),
                    color="k", fontsize=8, ha="center", va="center", zorder=5,
                    fontweight="bold")
    for name, cell in dict(scene.destinations).items():
        known = name in belief.dest_known
        ax.plot(cell[0] + 0.5, cell[1] + 0.5, "D",
                color="limegreen" if known else "red", ms=9, zorder=4)
        ax.annotate(name[0].upper(), (cell[0] + 0.5, cell[1] + 0.5),
                    color="k", fontsize=7, ha="center", va="center", zorder=5,
                    fontweight="bold")
    sx, sy = scene.start
    ax.plot(sx + 0.5, sy + 0.5, "o", color="black", ms=6, zorder=4)
    ex, ey = metrics["end_cell"]
    ax.plot(ex + 0.5, ey + 0.5, "X", color="purple", ms=8, zorder=4)
    ax.set_xlim(0, n); ax.set_ylim(0, n)
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"seed {metrics['seed']}  {fields['outcome']}  "
                 f"steps {metrics['total_steps']}/{max_steps}  ref {metrics['ref_steps']}  "
                 f"mach {metrics['n_machine_known']}/2 dest {metrics['n_dest_known']}/2 "
                 f"deliv {metrics['n_delivered']}  fron {metrics['frontier_end']}",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def eval_run(run_dir, checkpoint, n_seeds, max_steps, seed_base, render_n, out_dir,
             fallback_unvisited=False):
    """单 run 全套评估；返回 CSV 路径。"""
    import torch

    from algo.actions import legal_mask  # noqa: F401  （保证依赖已就绪）
    from algo.agent import SMDPDQNAgent
    from algo.belief import BeliefState
    from algo.executor import Executor
    from algo.eval import omniscient_reference
    from algo.trainer import TrainConfig, build_fixed_maze

    cfg = TrainConfig.from_yaml(os.path.join(run_dir, "config.json"))
    cfg.env_max_steps = int(max_steps)
    run_name = os.path.basename(os.path.normpath(run_dir))
    ckpt_name = checkpoint or "checkpoint_stage1_final.pt"
    ckpt_path = os.path.join(run_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    _threads = os.environ.get("BAMER_TORCH_THREADS", "")
    if _threads and int(_threads) > 0:
        torch.set_num_threads(int(_threads))
    torch.manual_seed(int(cfg.seed))
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    agent = SMDPDQNAgent(in_channels=14 + 2 * int(cfg.use_traj) + int(cfg.dist_channel)
                         + int(cfg.dir_channel),
                         n=cfg.maze_n, aux_head=cfg.aux_head,
                         dist_channel=cfg.dist_channel, dilated=cfg.dilated_backbone,
                         dir_channel=cfg.dir_channel, device="cpu", seed=cfg.seed)
    agent.load_state_dict(ckpt["agent"])

    seeds = [int(s) for s in np.random.default_rng(seed_base).integers(0, 2**31 - 1, n_seeds)]
    belief = BeliefState(n=cfg.maze_n, max_range=cfg.lidar_max_range,
                         scan_memo=cfg.scan_memo, use_dist=cfg.dist_channel,
                         use_dir=cfg.dir_channel)
    executor = Executor(belief, max_option_steps=cfg.env_max_steps, lazy_replan=cfg.lazy_replan)

    from bamer_env.env import BamerEnv
    from bamer_mouse.lidar import LidarSpec

    def factory():
        return BamerEnv(
            randomize=cfg.env_randomize, max_steps=cfg.env_max_steps,
            maze=build_fixed_maze(cfg) if cfg.maze_fixed is not None else None,
            maze_braiding=cfg.maze_braiding_stage1,
            maze_loops_range=tuple(cfg.maze_loops_stage1) if cfg.maze_loops_stage1 is not None else None,
            lidar=LidarSpec(n_beams=cfg.lidar_n_beams, max_range=cfg.lidar_max_range,
                            noise_sigma=0.0),
        )

    os.makedirs(out_dir, exist_ok=True)
    run_out = os.path.join(out_dir, f"{run_name}_ms{max_steps}"
                           + ("_funv" if fallback_unvisited else ""))
    os.makedirs(run_out, exist_ok=True)
    csv_path = os.path.join(run_out, "episodes.csv")

    rows = []
    for i, seed in enumerate(seeds):
        env = factory()
        metrics, traj = run_episode(agent, env, belief, executor, seed,
                                    cfg.use_traj, cfg.frontier_mask,
                                    fallback_unvisited=fallback_unvisited)
        scene = env.scene
        maze = scene.maze
        frontier_end = set(metrics.pop("frontier_set"))
        fields = classify(metrics, belief, maze, frontier_end, scene)
        ref = omniscient_reference(scene)
        metrics["ref_steps"] = ref if ref is not None else -1
        row = {k: metrics[k] for k in CSV_FIELDS if k in metrics}
        row.update(fields)
        row["ref_steps"] = ref if ref is not None else -1
        rows.append(row)
        env.close()

        # 代表样例渲染：每桶首个失败样例 + blindspot 额外 2 个
        want = render_n > 0 and fields["outcome"] in ("blindspot", "delivery_fail",
                                                      "deadlock", "budget_open", "anomaly")
        if want:
            per_class_cap = 2 if fields["outcome"] == "blindspot" else 1
            already = sum(1 for r in rows
                          if r["outcome"] == fields["outcome"] and r.get("_rendered"))
            if already < per_class_cap:
                render(os.path.join(run_out, f"seed{seed}_{fields['outcome']}.png"),
                       maze, scene, belief, traj, frontier_end, metrics, fields,
                       cfg.env_max_steps)
                row["_rendered"] = True
        if (i + 1) % 20 == 0:
            print(f"  [{run_name}] {i + 1}/{len(seeds)} seeds done", flush=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # 控制台汇总
    n = len(rows)
    print(f"\n[失败解剖] {run_name}  ckpt={ckpt_name}  max_steps={max_steps}  "
          f"n={n}  seed_base={seed_base}")
    counts = {c: sum(1 for r in rows if r["outcome"] == c) for c in CLASS_ORDER}
    print("  桶分布: " + "  ".join(f"{c} {counts[c]}" for c in CLASS_ORDER))
    print(f"  成功率 {counts['success'] / n:.1%}")
    ok = [r for r in rows if r["outcome"] == "success"]
    if ok:
        print(f"  成功步数均值 {np.mean([r['total_steps'] for r in ok]):.0f}  "
              f"参照均值 {np.mean([r['ref_steps'] for r in ok if r['ref_steps'] > 0]):.0f}")
    bs = [r for r in rows if r["outcome"] == "blindspot"]
    if bs:
        print(f"  blindspot: 步数均值 {np.mean([r['total_steps'] for r in bs]):.0f}  "
              f"探索率均值 {np.mean([r['explored_ratio'] for r in bs]):.1%}  "
              f"墙知识均值 {np.mean([r['wall_knowledge_ratio'] for r in bs]):.1%}  "
              f"末端前沿 {np.mean([r['frontier_end'] for r in bs]):.1f}  "
              f"末段游走(步-末新格) {np.mean([r['total_steps'] - r['steps_last_new_cell'] for r in bs]):.0f}")
    bo = [r for r in rows if r["outcome"] == "budget_open"]
    if bo:
        print(f"  budget_open: 步数均值 {np.mean([r['total_steps'] for r in bo]):.0f}  "
              f"探索率均值 {np.mean([r['explored_ratio'] for r in bo]):.1%}  "
              f"墙知识均值 {np.mean([r['wall_knowledge_ratio'] for r in bo]):.1%}  "
              f"末端前沿 {np.mean([r['frontier_end'] for r in bo]):.1f}  "
              f"末段仍在收缩(步-末收缩) {np.mean([r['total_steps'] - r['steps_last_frontier_shrink'] for r in bo]):.0f}")
    print(f"  CSV → {csv_path}")
    with open(os.path.join(run_out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"run_dir": run_dir, "checkpoint": ckpt_name, "n_seeds": n_seeds,
                   "max_steps": max_steps, "seed_base": seed_base,
                   "counts": counts,
                   "maze_fixed": cfg.maze_fixed,
                   "env_randomize": cfg.env_randomize,
                   "env_max_steps_cfg": cfg.env_max_steps}, f, indent=2)
    return csv_path


def compare(out_dir):
    """跨 run 失败 seed 重叠矩阵 + 桶分布对照。"""
    import glob

    paths = sorted(glob.glob(os.path.join(out_dir, "*", "episodes.csv")))
    if len(paths) < 2:
        print("compare 需要 ≥2 个 run 的 episodes.csv")
        return
    data = {}
    for p in paths:
        run = os.path.basename(os.path.dirname(p))
        with open(p, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        data[run] = {int(r["seed"]): r["outcome"] for r in rows}
    runs = list(data)
    seed_sets = {r: {s for s, o in data[r].items() if o != "success"} for r in runs}
    print("\n[跨 run 对比] 失败桶分布")
    all_classes = CLASS_ORDER
    print("  " + "  ".join(f"{r}" for r in runs))
    for c in all_classes:
        print(f"  {c:<14} " + "  ".join(
            f"{sum(1 for o in data[r].values() if o == c):>3}   " for r in runs))
    print("\n  失败 seed 重叠（|A∩B|/|A∪B|, 对角=|失败集|）")
    for i, a in enumerate(runs):
        line = f"  {a[:28]:<28}"
        for b in runs[:i + 1]:
            sa, sb = seed_sets[a], seed_sets[b]
            inter, union = len(sa & sb), len(sa | sb)
            line += f" {inter}/{union}  "
        print(line)
    core = set.intersection(*[seed_sets[r] for r in runs]) if runs else set()
    union_all = set.union(*[seed_sets[r] for r in runs]) if runs else set()
    print(f"  全体失败并集 {len(union_all)}  硬核交集（所有 run 都失败） {len(core)}: "
          f"{sorted(core)[:20]}")
    # 硬核 seed 的桶一致性
    for s in sorted(core)[:10]:
        print(f"    seed {s}: " + "  ".join(f"{r}={data[r][s]}" for r in runs))


def main():
    ap = argparse.ArgumentParser(description="失败样例解剖：分类 + 特征 + 渲染")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--seed-base", type=int, default=777)
    ap.add_argument("--render", type=int, default=0)
    ap.add_argument("--fallback-unvisited", action="store_true",
                    help="frontier 空回退改为未到访可达格（盲区修复验证）")
    ap.add_argument("--out-dir", default="output/failure_anatomy")
    ap.add_argument("--compare", default=None, help="只做跨 run 对比，指向 out-dir")
    args = ap.parse_args()

    if args.compare:
        compare(args.compare)
        return
    eval_run(args.run_dir, args.checkpoint, args.n_seeds, args.max_steps,
             args.seed_base, args.render, args.out_dir,
             fallback_unvisited=args.fallback_unvisited)


if __name__ == "__main__":
    main()
