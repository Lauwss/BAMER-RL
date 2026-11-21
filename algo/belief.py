# -*- coding: utf-8 -*-
"""信念层：探测账本 + 轨迹账本 + 16 通道信念张量派生（docs/rl_algorithm_design.md §2 定稿实现）。

R1 自洽（信息封闭）：本模块是机器鼠算法侧的记忆与观测合成层，输入仅限
contract_v1.md 允许的算法侧合法通道——
    1. obs 占位 Dict：cell / heading（机器鼠自身位姿）；
    2. info["ranges"] / info["angles"]：雷达量测（探测账本唯一环境来源）；
    3. info["events"]：状态机交互事件（识别经分类器走像素、carrying 走
       Pickup/Arrived 事件通道、碰撞揭示墙位）；
    4. 分类器输出：图像资产引用 → 身份（GT 直通占位，M2.5 接口；换真实
       分类器时仅替换该 callable，本模块零改动）。
本模块不 import bamer_env.scene / Maze，不读取任何场景内部字段（迷宫墙线、
状态机位置、delivery_map 均不可见）；单测以"不含 GT 的输入流"验证（§8 冒烟 1）。

四条编码约定（§2.1，全通道贯穿）：
    1. 账本派生：无任何被维护的空间地图，信念张量在每个决策边界由
       f(探测账本, 轨迹账本, 事件簿记, 位姿) 一次性重建，无写入路径；
    2. 全零 = 未知：POI 未发现、空手、墙位未探测 → 对应通道/槽位为 0；
    3. 类别量用小整数：墙位 0/1/2、机器 0/1/2、终点 0/1；
    4. 全局量广播：carrying 做成全图常量通道。

张量布局：[C, x, y]（与 tools/bench_belief_derive.py 同构；maze.py 的
wall_south[x][y] asarray 后即 [x, y]）。所有通道共用同一布局，位姿 4 通道与
墙位 4 通道方向对齐，使"正前方有无墙"成为静态权重共享的卷积运算。

探测账本语义（§2.2）：规范形式感知记录 sensed_south[x][y] / sensed_west[x][y]
（与 maze.py 南/西墙约定同构），值 0=未观测 1=观测到无墙 2=观测到有墙；
只增不减，冲突时"有墙"优先（悲观约定，与噪声关闭下的确定性观测自洽）。
**墙值不存储**——账本只存"看过了什么"，派生墙值 = sensed ? (wall ? 2 : 1) : 0
在 tensor() 中每次重算（bench 实测 ~108 µs/次，可忽略）。

规划语义（悲观约定）：unknown 一律按不可通行处理，执行器只在"已知空闲格"
上规划（见 algo/executor.py）；雷达量程 2.0 格保证 frontier 处恒有已揭示的
相邻可进格。
"""
import math
import os
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from bamer_mouse.kinematics import HEADING_DEG

# 朝向离散化（与 bamer_env.env.HEADINGS 同序：0=N 1=E 2=S 3=W）
HEADINGS = ("N", "E", "S", "W")
_DIRS = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}

# ---------- 通道布局（rl_algorithm_design.md §2.1 定稿表） ----------
CH_WALL_N, CH_WALL_E, CH_WALL_S, CH_WALL_W = 0, 1, 2, 3   # 与 HEADINGS 同序
CH_POSE_N, CH_POSE_E, CH_POSE_S, CH_POSE_W = 4, 5, 6, 7
CH_POI_QUYUAN, CH_POI_ZHAOJUN = 8, 9
CH_POI_CHUGUO, CH_POI_XIHAN = 10, 11
CH_CARRY_QUYUAN, CH_CARRY_ZHAOJUN = 12, 13
CH_TRAIL_PREV_OPTION = 14
CH_POSE_PREV_BOUNDARY = 15
N_CHANNELS = 16
BASE_CHANNELS = 14              # use_traj=False 消融：裁剪轨迹双通道

POI_CHANNELS = {"quyuan": CH_POI_QUYUAN, "zhaojun": CH_POI_ZHAOJUN}
DEST_CHANNELS = {"chuguo": CH_POI_CHUGUO, "xihan": CH_POI_XIHAN}
CARRY_CHANNELS = {"quyuan": CH_CARRY_QUYUAN, "zhaojun": CH_CARRY_ZHAOJUN}
MACHINE_NAMES = ("quyuan", "zhaojun")
DEST_NAMES = ("chuguo", "xihan")

# 墙位三态（类别小整数约定；与 pointcloud.OCC_* 同值不同源——彼为局部栅格，此为全图信念）
W_UNKNOWN, W_FREE, W_WALL = 0, 1, 2

Cell = Tuple[int, int]

# 算法侧合法识别通道的分类器协议：classify(图像资产引用) → 身份标识
# （"quyuan"/"zhaojun"/"chuguo"/"xihan"）；与 bamer_mouse.camera.Classifier
# 同为可插拔协议（GT 占位 = GTAssetClassifier，真实识别模型实现同签名即可挂接）。
class IdentityClassifier:
    def classify(self, image_ref: str) -> str:
        raise NotImplementedError


class GTAssetClassifier:
    """GT 直通占位分类器（与 bamer_mouse.camera.GTPassthroughClassifier 同语义位）。

    只消费算法侧合法字段 payload["image"]（图像资产引用 = 像素的引用），按资产
    文件反查身份——等价于理想像素分类器的占位实现；不读取 payload["machine"] /
    ["label"] / ["destination"] 等 GT 身份字段（R1：识别标记的数据来源是分类器
    输出而非事件字段，换真实分类器时本类整体替换，belief 零改动）。
    资产表与 bamer_env.interaction / render.py 同源（image2/4=机器，image3/5=终点）。
    """

    def __init__(self):
        # 延迟导入 bamer_env（资产表常量；不触碰 Scene/Maze 等场景内部）
        from bamer_env.interaction import DESTINATION_ASSETS, IMAGE_ASSETS
        self._path_to_name = {os.path.normpath(p): name
                              for name, p in {**IMAGE_ASSETS, **DESTINATION_ASSETS}.items()}

    def classify(self, image_ref: str) -> str:
        key = os.path.normpath(str(image_ref)) if image_ref else ""
        if key not in self._path_to_name:
            raise ValueError(f"分类器无法识别图像资产引用: {image_ref!r}"
                             "（素材缺失或非注册资产——GT 占位要求显式资产路径）")
        return self._path_to_name[key]


class SensingLedger:
    """探测账本：规范形式感知记录（§2.2）。

    sensed_south[x][y]：格 (x,y) 与 (x,y-1) 之间的边界感知结果（y=0 为南外边框）；
    sensed_west [x][y]：格 (x,y) 与 (x-1,y) 之间的边界感知结果（x=0 为西外边框）。
    值 0=未观测 1=观测到无墙 2=观测到有墙；只增不减，"有墙"优先（悲观约定）。
    """

    def __init__(self, n: int):
        self.n = int(n)
        self.sensed_south = np.zeros((n, n), dtype=np.uint8)   # [x][y]
        self.sensed_west = np.zeros((n, n), dtype=np.uint8)

    def reset(self, n: Optional[int] = None) -> None:
        if n is not None:
            self.n = int(n)
        self.sensed_south = np.zeros((self.n, self.n), dtype=np.uint8)
        self.sensed_west = np.zeros((self.n, self.n), dtype=np.uint8)

    # ---------- 槽位定位（两邻格 → 规范形式槽位） ----------
    def _slot_between(self, a: Cell, b: Cell) -> Optional[Tuple[np.ndarray, int, int]]:
        """相邻格 a、b 之间的边界对应的规范槽位（数组引用 + 下标）。

        不相邻或越界返回 None：南/西外边框有规范槽位（south[x][0] / west[0][y]），
        北/东外边框无规范槽位（派生时恒为有墙，无需记录）。
        """
        ax, ay = a
        bx, by = b
        n = self.n
        if bx == ax + 1 and by == ay:      # b 在 a 东侧 → 边界 = west[bx][by]
            return (self.sensed_west, bx, by) if bx < n else None
        if bx == ax - 1 and by == ay:      # b 在 a 西侧 → 边界 = west[ax][ay]
            return (self.sensed_west, ax, ay) if ax < n else None
        if by == ay + 1 and bx == ax:      # b 在 a 北侧 → 边界 = south[bx][by]
            return (self.sensed_south, bx, by) if by < n else None
        if by == ay - 1 and bx == ax:      # b 在 a 南侧 → 边界 = south[ax][ay]
            return (self.sensed_south, ax, ay) if ay < n else None
        return None

    def _mark(self, arr: np.ndarray, x: int, y: int, value: int) -> None:
        """单调写入：有墙(2) 不可降级；无墙(1) 可被后续有墙(2) 覆盖（悲观约定）。"""
        if value == W_WALL or arr[x, y] != W_WALL:
            arr[x, y] = max(arr[x, y], value)

    # ---------- 量测更新 ----------
    def update_scan(self, ranges: np.ndarray, angles: np.ndarray,
                    pose_x: float, pose_y: float, heading: str, max_range: float) -> None:
        """一帧全扫描 → 边界感知记录（纯量测复算）。

        DDA 射线穿越的格边界 = 无墙已探明（1）；回波边界（命中邻格）= 有墙已探明（2）。
        与 lidar.raycast 同一套边界推进/角点 tie 约定，噪声关闭时穿越路径与真实
        射线路径逐格一致（tests/test_pointcloud.py 已实证等价性）。

        2026-11-20 向量化：束间锁步 DDA（每轮每束恰跨一条边界，穿越侧直接给出
        规范槽位），标记收集后批量写入。与逐束标量版终态逐位一致——_mark 单调
        且"有墙恒胜无墙"，故先全部 free 后全部 wall 的批量写入与逐束交错写入
        同序无关；穿出网格的束即刻终止（外边框恒有墙，正常束不会走到这一步；
        标量版在噪声越界场景存在负索引回绕隐患，此处顺带消除）。
        """
        ranges = np.asarray(ranges, dtype=float)
        angles = np.asarray(angles, dtype=float)
        n_beams = ranges.shape[0]
        if n_beams == 0:
            return
        a = np.radians(HEADING_DEG[heading] + angles)
        dx = np.cos(a)
        dy = np.sin(a)
        n = self.n
        ix = np.full(n_beams, int(math.floor(pose_x)), dtype=np.int64)
        iy = np.full(n_beams, int(math.floor(pose_y)), dtype=np.int64)
        eps = 1e-9
        pos_dx, neg_dx = dx > eps, dx < -eps
        pos_dy, neg_dy = dy > eps, dy < -eps
        # 距离带与 walk_cells 一致：t < dist-eps 穿越；dist-eps <= t <= dist+eps
        # 记回波邻格（不跨入）；t > dist+eps 无更多边界。无回波束 dist = 量程。
        dist = np.where(np.isfinite(ranges), ranges, float(max_range))
        echo = np.isfinite(ranges)
        finished = np.zeros(n_beams, dtype=bool)
        free_x, free_y, free_s = [], [], []      # 穿越槽位（True=south 槽 / False=west 槽）
        wall_x, wall_y, wall_s = [], [], []      # 回波边界槽位

        for _ in range(4 * int(math.ceil(dist.max())) + 8):
            act = ~finished
            if not act.any():
                break
            with np.errstate(divide="ignore", invalid="ignore"):
                tx = np.where(pos_dx, (ix + 1 - pose_x) / dx,
                              np.where(neg_dx, (ix - pose_x) / dx, np.inf))
                ty = np.where(pos_dy, (iy + 1 - pose_y) / dy,
                              np.where(neg_dy, (iy - pose_y) / dy, np.inf))
            vert = tx <= ty                      # 角点 tie：先竖直（确定性约定）
            t = np.minimum(tx, ty)
            # 越带终止：t > dist+eps（无邻格记录）
            finished |= act & (t > dist + eps)
            # 回波边界：dist-eps <= t <= dist+eps → 记 wall 槽（echo 束才标记）
            fin1 = act & ~finished & (t >= dist - eps)
            if fin1.any():
                # 槽位取内/外侧格须与 _slot_between 逐条一致：
                # E→west[ix+1,iy]（外侧,护栏 bx<n）、W→west[ix,iy]（内侧）、
                # N→south[ix,iy+1]（外侧,护栏 by<n）、S→south[ix,iy]（内侧）
                bx = np.where(vert, np.where(pos_dx, ix + 1, ix), ix)
                by = np.where(vert, iy, np.where(pos_dy, iy + 1, iy))
                is_south = ~vert
                valid = (is_south & (by < n)) | (~is_south & (bx < n))
                sel = fin1 & echo & valid
                wall_x.append(bx[sel]); wall_y.append(by[sel]); wall_s.append(is_south[sel])
                finished |= fin1
            # 正常穿越：t < dist-eps → 记 free 槽并跨入邻格
            cross = act & ~finished
            if cross.any():
                bx = np.where(vert, np.where(pos_dx, ix + 1, ix), ix)
                by = np.where(vert, iy, np.where(pos_dy, iy + 1, iy))
                is_south = ~vert
                valid = (is_south & (by < n)) | (~is_south & (bx < n))
                sel = cross & valid
                free_x.append(bx[sel]); free_y.append(by[sel]); free_s.append(is_south[sel])
                # 跨入邻格；穿出网格的束即刻终止（外边框恒有墙，见 docstring）
                stepx = np.where(vert, np.where(pos_dx, 1, -1), 0)
                stepy = np.where(~vert, np.where(pos_dy, 1, -1), 0)
                exits = cross & (((stepx != 0) & ((ix + stepx < 0) | (ix + stepx >= n)))
                                 | ((stepy != 0) & ((iy + stepy < 0) | (iy + stepy >= n))))
                finished |= exits
                adv = cross & ~exits
                ix += stepx * adv
                iy += stepy * adv

        # 批量标记（顺序无关性见 docstring）
        if free_x:
            self._mark_slots(np.concatenate(free_x), np.concatenate(free_y),
                             np.concatenate(free_s), W_FREE)
        if wall_x:
            self._mark_slots(np.concatenate(wall_x), np.concatenate(wall_y),
                             np.concatenate(wall_s), W_WALL)

    def _mark_slots(self, xs: np.ndarray, ys: np.ndarray,
                    is_south: np.ndarray, value: int) -> None:
        """批量版 _mark：语义一致（wall 恒置 2；free 不可降级已有 wall）。"""
        for arr, sel in ((self.sensed_south, is_south), (self.sensed_west, ~is_south)):
            x, y = xs[sel], ys[sel]
            if x.size == 0:
                continue
            if value == W_WALL:
                arr[x, y] = W_WALL
            else:
                vals = arr[x, y]
                upd = vals != W_WALL
                arr[x[upd], y[upd]] = np.maximum(vals[upd], value)

    def mark_wall(self, x: int, y: int, direction: str) -> None:
        """碰撞/显式墙揭示：按方向定位规范槽位标记有墙（Collision 事件的合法消费）。"""
        if direction == "S":
            self._mark(self.sensed_south, x, y, W_WALL)
        elif direction == "W":
            self._mark(self.sensed_west, x, y, W_WALL)
        elif direction == "N":
            if y + 1 < self.n:
                self._mark(self.sensed_south, x, y + 1, W_WALL)
        elif direction == "E":
            if x + 1 < self.n:
                self._mark(self.sensed_west, x + 1, y, W_WALL)
        else:
            raise ValueError(f"未知方向: {direction!r}")

    # ---------- 查询 ----------
    def wall_value(self, x: int, y: int, direction: str) -> int:
        """派生墙值（§2.2）：0=未探测 1=已探明无墙 2=有墙。北/东外边框恒 2（封闭场地先验）。"""
        n = self.n
        if direction == "S":
            return int(self.sensed_south[x, y])
        if direction == "W":
            return int(self.sensed_west[x, y])
        if direction == "N":
            return int(self.sensed_south[x, y + 1]) if y + 1 < n else W_WALL
        if direction == "E":
            return int(self.sensed_west[x + 1, y]) if x + 1 < n else W_WALL
        raise ValueError(f"未知方向: {direction!r}")

    def unknown_count(self) -> int:
        """未观测槽位数（臂 B 熵奖励的精确 bit 计数）。"""
        return int((self.sensed_south == W_UNKNOWN).sum() + (self.sensed_west == W_UNKNOWN).sum())


class BeliefState:
    """机器鼠算法侧的信念状态 = 账本 + 事件簿记 + 轨迹账本，张量按需派生。

    用法（trainer/executor 的每步数据流）：
        belief.reset(n)
        belief.observe(obs, info)          # 每个低层步后（含 reset 第 0 帧）
        T = belief.tensor()                # 决策边界派生 (16,16,16) uint8
        belief.commit_option(path, start)  # option 结束时登记轨迹账本
    """

    def __init__(self, n: int = 16, classifier: Optional[IdentityClassifier] = None,
                 max_range: float = 2.0, scan_memo: bool = True):
        self.n = int(n)
        self.classifier = classifier if classifier is not None else GTAssetClassifier()
        self.max_range = float(max_range)
        self.ledger = SensingLedger(self.n)
        # 事件簿记（全部来自合法通道；识别身份经分类器，见模块 docstring）
        self.machine_known: Dict[str, Cell] = {}    # 已识别机器 → 识别位置（当步自身格）
        self.machine_delivered: Set[str] = set()
        self.dest_known: Dict[str, Cell] = {}       # 已定位终点 → 发现位置
        self.carrying: Optional[str] = None
        # 位姿与轨迹账本
        self.cell: Cell = (0, 0)
        self.heading: str = "N"
        self.visited: Set[Cell] = set()             # 累计到访格（轨迹账本·全历史）
        self.prev_option_path: List[Cell] = []      # 上一个 option 实际走过的格序列
        self.prev_boundary_cell: Optional[Cell] = None  # 上一个决策边界位置
        # 扫描记忆化（2026-11-21 性能等价变换）：规范位姿（格中心,朝向）确定场景下
        # 射线投射结果逐位确定 → 同 (cell, heading) 且量测一致时账本标记幂等，可跳过。
        # ranges 逐帧比对（equal_nan）保证噪声/异帧不误跳； episode 内 dict ≤256×4 项。
        self.scan_memo = bool(scan_memo)
        self._scan_memo: Dict[Tuple[Cell, str], np.ndarray] = {}

    # ---------- 生命周期 ----------
    def reset(self, n: Optional[int] = None) -> None:
        if n is not None:
            self.n = int(n)
        self.ledger.reset(self.n)
        self.machine_known = {}
        self.machine_delivered = set()
        self.dest_known = {}
        self.carrying = None
        self.cell = (0, 0)
        self.heading = "N"
        self.visited = set()
        self.prev_option_path = []
        self.prev_boundary_cell = None
        self._scan_memo.clear()

    # ---------- 每步观测更新（R1 合法通道唯一入口） ----------
    def observe(self, obs: dict, info: dict) -> None:
        """消费一步的 obs + info（量测 + 事件），更新账本与簿记。

        obs：占位 Dict {"cell", "heading"}（cell = y*n+x）；info：{"events"} +
        接雷达时 {"ranges", "angles"}。不含任何场景内部字段（R1）。
        """
        n = self.n
        idx = int(obs["cell"])
        self.cell = (idx % n, idx // n)             # env 约定 cell = y*n+x
        self.heading = HEADINGS[int(obs["heading"])]
        self.visited.add(self.cell)

        ranges = info.get("ranges")
        if ranges is not None:
            cx, cy = self.cell[0] + 0.5, self.cell[1] + 0.5   # 规范位姿 = 格中心
            hit = None
            if self.scan_memo:
                hit = self._scan_memo.get((self.cell, self.heading))
            if hit is not None and np.array_equal(hit, ranges, equal_nan=True):
                pass    # 同位姿同量测：账本标记幂等，跳过重复 update_scan（精确等价）
            else:
                self.ledger.update_scan(ranges, info["angles"], cx, cy, self.heading,
                                        self.max_range)
                if self.scan_memo:
                    self._scan_memo[(self.cell, self.heading)] = ranges

        for ev in info.get("events", ()):
            self._on_event(ev)

    def _on_event(self, ev) -> None:
        etype = getattr(ev, "type", None)
        payload = getattr(ev, "payload", {})
        if etype == "ImageFeed":
            # 识别 = 分类器(像素引用) → 身份；位置 = 当步自身格（自身信息）
            name = self.classifier.classify(str(payload.get("image", "")))
            self.machine_known.setdefault(name, self.cell)
        elif etype == "Pickup":
            # carrying 的合法信息通道（contract §3）；与分类器识别结果交叉校验
            name = str(payload["machine"])
            if name in self.machine_known:
                self.carrying = name
            else:   # 防御：事件先于识别到达时按识别位补记（首达即拾取时序）
                self.machine_known.setdefault(name, self.cell)
                self.carrying = name
        elif etype == "DestinationFeed":
            name = self.classifier.classify(str(payload.get("image", "")))
            self.dest_known.setdefault(name, self.cell)
        elif etype == "Arrived":
            name = str(payload["machine"]) if payload.get("machine") else self.carrying
            if name is not None:
                self.machine_delivered.add(name)
            self.carrying = None
        elif etype == "Collision":
            self.ledger.mark_wall(int(payload["x"]), int(payload["y"]), str(payload["direction"]))
        # Reported / TaskComplete / Timeout：与账本无关（自身位姿已由 obs 覆盖 / 终止信号）

    # ---------- option 边界轨迹登记 ----------
    def commit_option(self, path: List[Cell], start_cell: Cell) -> None:
        """option 结束时登记轨迹账本（下一个边界的 trail_prev_option / pose_prev_boundary）。"""
        self.prev_option_path = list(path)
        self.prev_boundary_cell = tuple(start_cell)

    # ---------- 派生查询（规划用，悲观约定） ----------
    def open_boundary(self, cell: Cell, direction: str) -> bool:
        """边界已探明且无墙（F 可行的唯一判据；unknown = 不可通行）。"""
        return self.ledger.wall_value(cell[0], cell[1], direction) == W_FREE

    def known_free(self, cell: Cell) -> bool:
        """格已知空闲：到访过，或任一入射边界已探明无墙（开放边界 ⇒ 两侧可达）。"""
        if cell in self.visited:
            return True
        return any(self.open_boundary(cell, d) for d in HEADINGS)

    def poi_complete(self) -> bool:
        """全部 POI（2 机器 + 2 终点）已定位（评估诊断量）。"""
        return len(self.machine_known) >= len(MACHINE_NAMES) and \
            len(self.dest_known) >= len(DEST_NAMES)

    # ---------- 16 通道信念张量派生 ----------
    def tensor(self, use_traj: bool = True) -> np.ndarray:
        """由账本一次性重建信念张量（§2.1 定稿；无写入路径，每次全量派生）。

        返回 (16,16,16) uint8（use_traj=False 时 (14,16,16)）；布局 [C, x, y]。
        """
        n = self.n
        T = np.zeros((N_CHANNELS, n, n), dtype=np.uint8)
        south, west = self.ledger.sensed_south, self.ledger.sensed_west

        # --- 墙位 4 通道（0=unknown 1=free 2=wall；派生值 = sensed ? (wall?2:1) : 0） ---
        # 规范槽位值本身即 0/1/2 感知结果；N/E 侧读邻格槽位，外边框恒 2（封闭场地先验，
        # 与 tools/bench_belief_derive.py 参考实现逐元素一致）
        T[CH_WALL_N, :, :-1] = south[:, 1:]
        T[CH_WALL_N, :, -1] = W_WALL
        T[CH_WALL_E, :-1, :] = west[1:, :]
        T[CH_WALL_E, -1, :] = W_WALL
        T[CH_WALL_S] = south
        T[CH_WALL_W] = west

        # --- 位姿 4 通道（位置 ⊗ 朝向 one-hot，方向对齐墙位通道） ---
        x, y = self.cell
        T[CH_POSE_N + HEADINGS.index(self.heading), x, y] = 1

        # --- 状态机知识（0=未识别 1=已识别未送达 2=已送达；在该机已知格标记） ---
        for name, cell in self.machine_known.items():
            ch = POI_CHANNELS.get(name)
            if ch is not None:
                T[ch, cell[0], cell[1]] = 2 if name in self.machine_delivered else 1

        # --- 终点知识（0=未定位 1=已定位；发现且识别后在该格标记） ---
        for name, cell in self.dest_known.items():
            ch = DEST_CHANNELS.get(name)
            if ch is not None:
                T[ch, cell[0], cell[1]] = 1

        # --- carrying（全局量广播通道；空手 = 双通道全 0） ---
        if self.carrying is not None:
            ch = CARRY_CHANNELS.get(self.carrying)
            if ch is not None:
                T[ch] = 1

        # --- 轨迹双通道（新近性上下文；episode 首边界全零 = 未知约定） ---
        if use_traj:
            for cx, cy in self.prev_option_path:
                T[CH_TRAIL_PREV_OPTION, cx, cy] = 1
            if self.prev_boundary_cell is not None:
                T[CH_POSE_PREV_BOUNDARY, self.prev_boundary_cell[0], self.prev_boundary_cell[1]] = 1
            return T
        return T[:BASE_CHANNELS].copy()

    # ---------- 辅助任务 GT 通道（训练监督专用，非决策输入） ----------
    def unknown_slot_mask(self) -> np.ndarray:
        """规范槽位未知掩码 (2,n,n) bool：[0]=south [1]=west（辅助 CE 只在 unknown 位置计算）。"""
        return np.stack([self.ledger.sensed_south == W_UNKNOWN,
                         self.ledger.sensed_west == W_UNKNOWN])
