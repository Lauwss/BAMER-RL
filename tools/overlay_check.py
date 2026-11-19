# -*- coding: utf-8 -*-
"""把提取的墙线以红色叠加回原图，生成对比图用于人工验收。"""
import json
import os

import cv2

ROOT = os.path.join(os.path.dirname(__file__), "..")
IMG = os.path.join(ROOT, "docs", "_pptx_media", "ppt", "media", "image1.png")
MAZE_JSON = os.path.join(ROOT, "bamer_env", "default_maze.json")
OUT = os.path.join(ROOT, "docs", "_overlay_check.png")

N = 16
left, top, right, bottom = 125, 427, 785, 1078
cw = (right - left) / N
ch = (bottom - top) / N

d = json.load(open(MAZE_JSON, encoding="utf-8"))
ws, ww = d["wall_south"], d["wall_west"]

img = cv2.imread(IMG)
ov = img.copy()
n_s = n_w = 0
for x in range(N):
    for y in range(N):
        if ws[x][y]:
            p1 = (int(round(left + x * cw)), int(round(bottom - y * ch)))
            p2 = (int(round(left + (x + 1) * cw)), int(round(bottom - y * ch)))
            cv2.line(ov, p1, p2, (0, 0, 255), 2)
            n_s += 1
        if ww[x][y]:
            p1 = (int(round(left + x * cw)), int(round(bottom - y * ch)))
            p2 = (int(round(left + x * cw)), int(round(bottom - (y + 1) * ch)))
            cv2.line(ov, p1, p2, (0, 0, 255), 2)
            n_w += 1

crop = ov[top - 25: bottom + 30, left - 25: right + 30]
cv2.imwrite(OUT, crop)
print(f"saved {OUT}  south_walls={n_s}  west_walls={n_w}")
