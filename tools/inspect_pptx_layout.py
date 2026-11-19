# -*- coding: utf-8 -*-
"""解析 PPTX slide1 的形状坐标，精确计算屈原/昭君/终点图标所在的格子。

EMU: 914400 EMU = 1 英寸；格子坐标由迷宫底图（图片 7）的包围盒线性映射。
"""
import os
import re
import zipfile

import xml.etree.ElementTree as ET

ROOT = os.path.join(os.path.dirname(__file__), "..")
PPTX = os.path.join(ROOT, "docs", "机器人与人工智能.pptx")

NS = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

with zipfile.ZipFile(PPTX) as z:
    xml = z.read("ppt/slides/slide1.xml").decode("utf-8")
    rels = z.read("ppt/slides/_rels/slide1.xml.rels").decode("utf-8")
    media_map = {}
    for m in re.finditer(r'Id="(rId\d+)"[^>]*Target="([^"]+)"', rels):
        media_map[m.group(1)] = m.group(2)

root = ET.fromstring(xml)
tree = root.find("p:cSld/p:spTree", NS)

shapes = []
for sp in tree:
    tag = sp.tag.split("}")[1]
    nv = sp.find(".//p:cNvPr", NS)
    if nv is None:
        continue
    sid, name = nv.get("id"), nv.get("name")
    xfrm = sp.find(".//a:xfrm", NS)
    if xfrm is None:
        continue
    off = xfrm.find("a:off", NS)
    ext = xfrm.find("a:ext", NS)
    if off is None or ext is None:
        continue
    x, y = int(off.get("x")), int(off.get("y"))
    cx, cy = int(ext.get("cx")), int(ext.get("cy"))
    # 图片形状找到引用的媒体文件
    blip = sp.find(".//a:blip", NS)
    rid = blip.get(f'{{{NS["r"]}}}embed') if blip is not None else None
    media = media_map.get(rid, "")
    shapes.append((int(sid), name, tag, x, y, cx, cy, media))

for s in shapes:
    print(s)

# 迷宫底图 = image1.png
maze = [s for s in shapes if "image1" in s[7]][0]
_, _, _, mx, my, mcx, mcy, _ = maze
print(f"\n迷宫底图 EMU: x={mx} y={my} w={mcx} h={mcy}")

# 底图内容实际有白边（日文标题等），需要用内容 bbox 校正：
# 先假设底图内容铺满形状；后面用渲染对比再校正
def to_grid(px, py):
    gx = (px - mx) / mcx * 16
    gy = 16 - (py - my) / mcy * 16
    return gx, gy

print("\n各图标中心格子坐标（初步，假设底图铺满形状）:")
for sid, name, tag, x, y, cx, cy, media in shapes:
    if tag == "pic" and "image1" not in media and media:
        gx, gy = to_grid(x + cx / 2, y + cy / 2)
        print(f"  {name} [{os.path.basename(media)}] center=({gx:.2f},{gy:.2f}) -> cell ({int(gx)},{int(gy)})")
