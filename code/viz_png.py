#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_png.py -- 极简 Pillow 绘图helper（用于第四章 5 个算法程序的可视化）

为什么单独写：第四章的 5 个算法图需要「多面板 + 网络图 + 热力矩阵」，
chartlib 只做柱/折/散点，所以这里用 Pillow 直接画，输出 PNG 供课件与网页共用。

对外接口：
    Canvas(w, h)          画布
      .title(t)           顶部标题
      .text(x, y, s, ...) 文本（自动用微软雅黑渲染中文）
      .rect(...)          圆角矩形
      .arrow(...)         箭头
      .node(...)          圆形节点
      .line(...)          线段
      .save(path)         保存
      颜色常量 RED / GOLD / INK / MUTED / PAPER / LINE
"""

import os
from PIL import Image, ImageDraw, ImageFont

RED = (0x71, 0x14, 0x26)
RED_L = (0xE8, 0xC9, 0xCE)
GOLD = (0xC9, 0xA2, 0x27)
GOLD_L = (0xF3, 0xE6, 0xBC)
INK = (0x1E, 0x26, 0x30)
MUTED = (0x66, 0x71, 0x7D)
PAPER = (0xFA, 0xF7, 0xF6)
LINE = (0xC9, 0xCF, 0xD6)

_FONT_CANDIDATES = [
    r'C:\Windows\Fonts\msyh.ttc',
    r'C:\Windows\Fonts\msyhbd.ttc',
    r'C:\Windows\Fonts\simhei.ttf',
    r'C:\Windows\Fonts\simsun.ttc',
]


def _font(size, bold=False):
    idx = 1 if bold else 0
    order = [_FONT_CANDIDATES[1], _FONT_CANDIDATES[0]] if bold else _FONT_CANDIDATES
    for p in order[idx:] + order[:idx]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


class Canvas:
    def __init__(self, w=1200, h=760, bg=PAPER, scale=1):
        self.w, self.h = int(w * scale), int(h * scale)
        self.s = scale
        self.img = Image.new('RGB', (self.w, self.h), bg)
        self.d = ImageDraw.Draw(self.img)

    def _p(self, v):
        return int(round(v * self.s))

    def title(self, t, size=30):
        self.d.text((self._p(30), self._p(20)), t, font=_font(int(size * self.s), True), fill=RED)
        self.d.line([(self._p(30), self._p(64)), (self.w - self._p(30), self._p(64))],
                    fill=RED, width=max(1, int(2 * self.s)))

    def text(self, x, y, s, size=16, color=INK, bold=False, anchor='la'):
        self.d.text((self._p(x), self._p(y)), s,
                    font=_font(int(size * self.s), bold), fill=color, anchor=anchor)

    def rect(self, x, y, w, h, fill=None, outline=LINE, width=2, r=10, dash=False):
        box = [self._p(x), self._p(y), self._p(x + w), self._p(y + h)]
        if dash:
            self._dashed_rect(box, outline or LINE, width)
        else:
            self.d.rounded_rectangle(box, radius=self._p(r), fill=fill,
                                     outline=outline, width=max(1, int(width * self.s)))

    def _dashed_rect(self, box, color, width, seg=10):
        x0, y0, x1, y1 = box
        step = self._p(seg)
        for x in range(x0, x1, step * 2):
            self.d.line([(x, y0), (min(x + step, x1), y0)], fill=color, width=width)
            self.d.line([(x, y1), (min(x + step, x1), y1)], fill=color, width=width)
        for y in range(y0, y1, step * 2):
            self.d.line([(x0, y), (x0, min(y + step, y1))], fill=color, width=width)
            self.d.line([(x1, y), (x1, min(y + step, y1))], fill=color, width=width)

    def line(self, x0, y0, x1, y1, color=LINE, width=2, dash=False):
        p0 = (self._p(x0), self._p(y0))
        p1 = (self._p(x1), self._p(y1))
        w = max(1, int(width * self.s))
        if dash:
            import math
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            dist = math.hypot(dx, dy) or 1
            seg = self._p(9)
            n = int(dist // (seg * 2)) + 1
            for i in range(n):
                a = i * seg * 2 / dist
                b = min((i * seg * 2 + seg) / dist, 1)
                self.d.line([(p0[0] + dx * a, p0[1] + dy * a),
                             (p0[0] + dx * b, p0[1] + dy * b)], fill=color, width=w)
        else:
            self.d.line([p0, p1], fill=color, width=w)

    def arrow(self, x0, y0, x1, y1, color=MUTED, width=2):
        self.line(x0, y0, x1, y1, color, width)
        import math
        ang = math.atan2(self._p(y1) - self._p(y0), self._p(x1) - self._p(x0))
        L = self._p(11)
        for da in (0.42, -0.42):
            self.d.line([(self._p(x1), self._p(y1)),
                         (self._p(x1) - L * math.cos(ang - da),
                          self._p(y1) - L * math.sin(ang - da))],
                        fill=color, width=max(1, int(width * self.s)))

    def node(self, x, y, r=22, fill=RED_L, outline=RED, width=2):
        box = [self._p(x - r), self._p(y - r), self._p(x + r), self._p(y + r)]
        self.d.ellipse(box, fill=fill, outline=outline, width=max(1, int(width * self.s)))

    def cell(self, x, y, w, h, color):
        self.d.rectangle([self._p(x), self._p(y), self._p(x + w), self._p(y + h)], fill=color)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self.img.save(path, 'PNG')
        return path


def lerp(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))
