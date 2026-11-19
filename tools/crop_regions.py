# -*- coding: utf-8 -*-
"""裁剪原图关键区域放大查看，核对封闭区是否原图固有。"""
import os

import cv2

ROOT = os.path.join(os.path.dirname(__file__), "..")
IMG = os.path.join(ROOT, "docs", "_pptx_media", "ppt", "media", "image1.png")
left, top, right, bottom = 125, 427, 785, 1078
N = 16
cw = (right - left) / N
ch = (bottom - top) / N

img = cv2.imread(IMG)


def cell_to_px(x, y):
    """格左下角 -> 像素。"""
    return int(left + x * cw), int(bottom - y * ch)


# 待核对区域: (名称, x0,y0, x1,y1) 格范围（左下、右上，含边界外一圈）
regions = [
    ("A_topleft", 0, 11, 4, 15),
    ("B_midleft", 1, 7, 4, 9),
    ("C_center", 3, 4, 7, 8),
    ("D_bottommid", 8, 0, 11, 3),
    ("E_midright", 8, 4, 14, 8),
    ("F_right", 11, 11, 14, 15),
    ("G_topright", 13, 13, 15, 15),
]

crops = []
for name, x0, y0, x1, y1 in regions:
    px0, py1 = cell_to_px(x0, y0)
    px1, py0 = cell_to_px(x1 + 1, y1 + 1)
    crop = img[py0:py1, px0:px1]
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    crops.append((name, crop))

# 纵向拼接，标签留白
max_w = max(c.shape[1] for _, c in crops)
total_h = sum(c.shape[0] + 30 for _, c in crops)
canvas = 255 * np.ones((total_h, max_w, 3), dtype="uint8") if (np := __import__("numpy")) else None
y_off = 0
for name, c in crops:
    canvas[y_off:y_off + 30, :200] = 255
    cv2.putText(canvas, name, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    canvas[y_off + 30:y_off + 30 + c.shape[0], :c.shape[1]] = c
    y_off += 30 + c.shape[0]

out = os.path.join(ROOT, "docs", "_regions_check.png")
cv2.imwrite(out, canvas)
print(f"saved {out}  size={canvas.shape}")
