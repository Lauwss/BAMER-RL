# -*- coding: utf-8 -*-
"""从 PPT slide1 的虚线连接符（开口标记）提取墙上开口，生成 PPT 场景迷宫。

原理：PPT 在官方 2022 迷宫底图上叠加了 13 条虚线段（cxnSp 连接符形状），
按迷宫图纸惯例，虚线 = 该处无墙（开口）。本脚本：
  1. 解析 slide1.xml，取迷宫图片的 xfrm+srcRect 与全部连接符坐标；
  2. EMU → 底图像素 → 迷宫格子坐标 精确映射；
  3. 把每个开口对应的墙线从 default_maze.json 中移除；
  4. 输出 default_maze_ppt.json 并验证全连通。
"""
import os
import re
import sys
import zipfile

import cv2
import numpy as np
import xml.etree.ElementTree as ET

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT))

from extract_maze import find_maze_bbox, IMG_PATH  # noqa: E402
from bamer_env.maze import Maze  # noqa: E402

NS = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}

N = 16

with zipfile.ZipFile(os.path.join(ROOT, "docs", "机器人与人工智能.pptx")) as z:
    xml = z.read("ppt/slides/slide1.xml").decode("utf-8")

root = ET.fromstring(xml)
tree = root.find("p:cSld/p:spTree", NS)

# ---- 迷宫底图形状与 srcRect 裁剪 ----
maze_xfrm = None
maze_src = None
for sp in tree.iter(f'{{{NS["p"]}}}pic'):
    nv = sp.find(".//p:cNvPr", NS)
    blip = sp.find(".//a:blip", NS)
    rid = blip.get(f'{{http://schemas.openxmlformats.org/officeDocument/2006/relationships}}embed')
    # rId1 = image1.png（迷宫底图）
    if rid == "rId1":
        xfrm = sp.find(".//a:xfrm", NS)
        off, ext = xfrm.find("a:off", NS), xfrm.find("a:ext", NS)
        maze_xfrm = tuple(int(off.get(k)) for k in ("x", "y")) + tuple(int(ext.get(k)) for k in ("cx", "cy"))
        src = sp.find(".//a:srcRect", NS)
        maze_src = {k: int(src.get(k, 0)) / 100000 for k in ("l", "t", "r", "b")} if src is not None else {"l": 0, "t": 0, "r": 0, "b": 0}
mx, my, mcx, mcy = maze_xfrm
print(f"迷宫形状 EMU: {maze_xfrm}, srcRect: {maze_src}")

# ---- 底图检测的迷宫包围盒（像素）----
img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
left, top, right, bottom = find_maze_bbox((img < 128).astype(np.uint8))
cw = (right - left) / N
ch = (bottom - top) / N
print(f"底图迷宫包围盒: ({left},{top})-({right},{bottom})  格子 {cw:.2f}x{ch:.2f}px")

src_w, src_h = img.shape[1], img.shape[0]
u_span = 1 - maze_src["l"] - maze_src["r"]
v_span = 1 - maze_src["t"] - maze_src["b"]


def emu_to_grid_x(emu_x):
    """形状内 EMU x → 底图像素 x → 格坐标 x。"""
    u = (emu_x - mx) / mcx
    px = src_w * (maze_src["l"] + u * u_span)
    return (px - left) / cw


def emu_to_grid_y(emu_y):
    v = (emu_y - my) / mcy
    py = src_h * (maze_src["t"] + v * v_span)
    return (bottom - py) / ch


# ---- 解析全部连接符（虚线开口）----
openings = []
for sp in tree.iter(f'{{{NS["p"]}}}cxnSp'):
    nv = sp.find(".//p:cNvPr", NS)
    xfrm = sp.find(".//a:xfrm", NS)
    off, ext = xfrm.find("a:off", NS), xfrm.find("a:ext", NS)
    x, y = int(off.get("x")), int(off.get("y"))
    w, h = int(ext.get("cx")), int(ext.get("cy"))

    if w >= h:  # 水平虚线 → y 边界上的开口
        g0, g1 = sorted((emu_to_grid_x(x), emu_to_grid_x(x + w)))
        gy = emu_to_grid_y(y + h / 2)
        Y = int(round(gy))
        lo, hi = int(np.floor(g0 + 0.15)), int(np.ceil(g1 - 0.15))
        cells = list(range(lo, hi))
        openings.append(("H", Y, cells, (g0, g1, gy)))
    else:  # 垂直虚线 → x 边界上的开口
        g0, g1 = sorted((emu_to_grid_y(y), emu_to_grid_y(y + h)))
        gx = emu_to_grid_x(x + w / 2)
        X = int(round(gx))
        lo, hi = int(np.floor(g0 + 0.15)), int(np.ceil(g1 - 0.15))
        cells = list(range(lo, hi))
        openings.append(("V", X, cells, (g0, g1, gx)))

print(f"\n共解析出 {len(openings)} 个开口:")
for kind, b, cells, raw in openings:
    print(f"  {kind} 边界={b} 格={cells}  原始={tuple(round(v,2) for v in raw)}")

# ---- 应用开口 ----
maze = Maze.default()
n_open = 0
for kind, b, cells, _ in openings:
    if kind == "H":
        if b <= 0:
            print(f"  跳过南外边框开口 y={b}")
            continue
        for x in cells:
            if maze.wall_south[x][b]:
                maze.wall_south[x][b] = False
                n_open += 1
    else:
        if b <= 0:
            print(f"  跳过西外边框开口 x={b}")
            continue
        for y in cells:
            if maze.wall_west[b][y]:
                maze.wall_west[b][y] = False
                n_open += 1
print(f"\n共移除墙线 {n_open} 段")

maze.name = "ppt-scene-2022tohoku-opened"
out = os.path.join(ROOT, "bamer_env", "default_maze_ppt.json")
maze.save(out)
print(f"已保存 {out}")

ok = maze.is_fully_connected()
reach = maze.reachable_from((0, 0))
unreach = [(x, y) for x in range(N) for y in range(N) if (x, y) not in reach]
print(f"全连通: {ok}  不可达格子: {len(unreach)} {unreach[:20] if unreach else ''}")
