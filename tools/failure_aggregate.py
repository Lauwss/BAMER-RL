# -*- coding: utf-8 -*-
"""失败解剖汇总：跨 run 盲区样例的 POI 归因统计（2026-11-22）。

读 failure_anatomy 的各 run episodes.csv，统计：
  - 桶分布
  - 盲区样例的未识别 POI 归因（名字 / 末端状态 / GT 死胡同）
  - 卡死时已送达数与末段零进展游走步数
  - 对照组：成功样例中 POI 落 GT 死胡同的占比（死胡同本身不判死刑）

用法：python tools/failure_aggregate.py --out-dir output/failure_anatomy
"""
import argparse
import csv
import os
from collections import Counter

CSV_FIELDS_STAT = ("unidentified", "outcome", "total_steps", "steps_last_new_cell",
                   "n_delivered", "n_machine_known", "n_dest_known")


def parse_poi(unidentified: str):
    """'quyuan@3,4:infodead|DE;...' → [(name, cell, state, is_dead_end)]。"""
    out = []
    for item in filter(None, unidentified.split(";")):
        name_loc, state_part = item.split(":")
        name, loc = name_loc.split("@")
        parts = state_part.split("|")
        yield name, loc, parts[0], len(parts) > 1 and parts[1] == "DE"


def main():
    ap = argparse.ArgumentParser(description="失败解剖汇总")
    ap.add_argument("--out-dir", default="output/failure_anatomy")
    args = ap.parse_args()

    buckets = Counter()
    per_run = {}
    fail_poi = Counter()
    de_fail = de_success_poi = 0
    n_fail_poi = n_success_poi = 0
    deliv_at_stuck = Counter()
    wasted = []
    unentered = {"success": [], "blindspot": []}

    for run in sorted(os.listdir(args.out_dir)):
        path = os.path.join(args.out_dir, run, "episodes.csv")
        if not os.path.exists(path):
            continue
        for row in csv.DictReader(open(path, encoding="utf-8")):
            buckets[row["outcome"]] += 1
            per_run.setdefault(run, Counter())[row["outcome"]] += 1
            for name, loc, state, de in parse_poi(row["unidentified"]):
                if row["outcome"] == "blindspot" and state != "id":
                    fail_poi[(name, state, de)] += 1
                    n_fail_poi += 1
                    de_fail += int(de)
                elif row["outcome"] == "success":
                    n_success_poi += 1
                    de_success_poi += int(de)
            if row["outcome"] == "blindspot":
                deliv_at_stuck[row["n_delivered"]] += 1
                wasted.append(int(row["total_steps"]) - int(row["steps_last_new_cell"]))
            if row["outcome"] in unentered:
                unentered[row["outcome"]].append(
                    int(row["reachable_end"]) - int(row["visited_end"]))

    n_bs = buckets["blindspot"]
    print("逐 run 桶分布（成功率 / blindspot / 其余失败）:")
    for run, cnt in per_run.items():
        n = sum(cnt.values())
        rest = sum(v for k, v in cnt.items() if k not in ("success", "blindspot"))
        print(f"  {run:<44} n={n}  成功 {cnt['success']} ({cnt['success'] / n:.0%})  "
              f"blindspot {cnt['blindspot']}  其余 {rest}")
    print(f"\n合计桶分布: {dict(buckets)}")
    print(f"\n盲区样例 {n_bs} 条，未识别 POI {n_fail_poi} 个：")
    for (name, state, de), v in sorted(fail_poi.items(), key=lambda x: -x[1]):
        print(f"  {name:<8} 末端={state:<8} GT死胡同={de}  n={v}")
    if n_fail_poi:
        print(f"  GT 死胡同占比 {de_fail / n_fail_poi:.0%}")
    if n_success_poi:
        print(f"对照组（成功样例）POI 落死胡同占比 {de_success_poi / n_success_poi:.0%}")
    if deliv_at_stuck:
        print(f"卡死时已送达数分布: {dict(sorted(deliv_at_stuck.items()))}")
    if wasted:
        ws = sorted(wasted)
        print(f"末段零进展游走: 均值 {sum(wasted) / len(wasted):.0f} 步 "
              f"中位 {ws[len(ws) // 2]} 最小 {ws[0]} 最大 {ws[-1]}")
    for cls, vals in unentered.items():
        if vals:
            vs = sorted(vals)
            print(f"{cls}: 未进入的可达格 = 均值 {sum(vals) / len(vals):.1f} "
                  f"中位 {vs[len(vs) // 2]} 最大 {vs[-1]} "
                  f"(wall=100% 时即被系统性绕开的信息死格)")


if __name__ == "__main__":
    main()
