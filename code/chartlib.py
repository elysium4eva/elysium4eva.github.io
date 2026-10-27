#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chartlib.py —— 纯标准库 SVG 图表工具（柱状 / 条形 / 折线 / 散点）

为什么自己写而不装 matplotlib：
    课程环境只保证 numpy + pandas。用 SVG 直接生成矢量图，
    不依赖任何第三方库、不依赖字体、在网页里清晰可缩放，
    而且生成的 SVG 可以当作文本 diff，便于版本管理。

用法：
    from chartlib import BarChart, HBarChart, LineChart, ScatterChart

    ch = BarChart("标题", "横轴名", "纵轴名")
    ch.add_series("样本", ["A", "B", "C"], [12, 30, 18])
    ch.save("out.svg")

所有坐标单位是像素；默认画布 680×400，适合网页卡片宽度。
"""

import math
import os


# ---------------------------------------------------------------- 基础工具
def _esc(s):
    """转义 SVG 文本。"""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _fmt(v):
    """数字格式化：整数不带小数，小数保留两位。"""
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v)):,}"
    if abs(v) >= 100:
        return f"{v:,.1f}"
    return f"{v:,.2f}".rstrip("0").rstrip(".")


def nice_ticks(vmin, vmax, n=5):
    """生成好看的刻度区间。"""
    if vmax == vmin:
        vmax = vmin + 1
    rng = vmax - vmin
    raw = rng / max(n, 2)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if step >= raw:
            break
    lo = math.floor(vmin / step) * step
    hi = math.ceil(vmax / step) * step
    ticks = []
    t = lo
    while t <= hi + step * 1e-6:
        ticks.append(round(t, 10))
        t += step
    return lo, hi, ticks


PALETTE = ["#8C1A2E", "#1F6FB2", "#C9A227", "#2E7D4F", "#6B4C9A", "#B45309"]


class _Canvas:
    ROTATE = 0

    def __init__(self, title, xlabel="", ylabel="", width=680, height=400,
                 ml=66, mr=26, mt=48, mb=58):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.W, self.H = width, height
        self.ml, self.mr, self.mt, self.mb = ml, mr, mt, mb
        self.parts = []
        self.legend = []
        self.notes = []
        self._si = 0
        self.plot_w = width - ml - mr
        self.plot_h = height - mt - mb

    def add_note(self, text):
        self.notes.append(text)
        return self

    def _reserve_notes(self):
        """有注释时，在画布底部预留空间，避免与 x 轴刻度重叠。"""
        # 注释改画在绘图区上方，不再需要底部留白
        return

    def _draw_notes(self):
        """把注释画在画布最底部，超长截断，只画一次。"""
        if not self.notes:
            return
        for i, n in enumerate(self.notes):
            ny = self.mt - 8 - (len(self.notes) - 1 - i) * 13
            txt = n if len(n) <= 70 else n[:68] + "\u2026"
            self.parts.append(
                f'<text x="{self.ml}" y="{ny:.0f}" font-family="system-ui,sans-serif" '
                f'font-size="9.5" fill="#6b7883">{_esc(txt)}</text>')

    def color(self):
        c = PALETTE[self._si % len(PALETTE)]
        self._si += 1
        return c

    def y_of(self, v, lo, hi):
        if hi == lo:
            return self.mt + self.plot_h
        return self.mt + self.plot_h * (1 - (v - lo) / (hi - lo))

    def frame(self, yticks, lo, hi, ylabel_suffix=""):
        self._reserve_notes()
        p = self.parts
        p.append(f'<rect x="0" y="0" width="{self.W}" height="{self.H}" fill="#ffffff"/>')
        if self.title:
            p.append(f'<text x="{self.W/2:.0f}" y="26" text-anchor="middle" '
                     f'font-family="system-ui,-apple-system,Segoe UI,sans-serif" '
                     f'font-size="14" font-weight="600" fill="#1e2630">{_esc(self.title)}</text>')
        # y 网格与刻度
        for t in yticks:
            y = self.y_of(t, lo, hi)
            p.append(f'<line x1="{self.ml}" y1="{y:.1f}" x2="{self.ml + self.plot_w}" '
                     f'y2="{y:.1f}" stroke="#e3e8ec" stroke-width="1"/>')
            p.append(f'<text x="{self.ml - 8}" y="{y + 3.5:.1f}" text-anchor="end" '
                     f'font-family="ui-monospace,SFMono-Regular,monospace" font-size="10" '
                     f'fill="#6b7883">{_esc(_fmt(t))}{_esc(ylabel_suffix)}</text>')
        # 轴线
        p.append(f'<line x1="{self.ml}" y1="{self.mt}" x2="{self.ml}" '
                 f'y2="{self.mt + self.plot_h}" stroke="#9aa7b2" stroke-width="1.2"/>')
        p.append(f'<line x1="{self.ml}" y1="{self.mt + self.plot_h}" '
                 f'x2="{self.ml + self.plot_w}" y2="{self.mt + self.plot_h}" '
                 f'stroke="#9aa7b2" stroke-width="1.2"/>')
        if self.xlabel:
            p.append(f'<text x="{self.ml + self.plot_w/2:.0f}" y="{self.H - 14}" '
                     f'text-anchor="middle" font-family="system-ui,sans-serif" font-size="11" '
                     f'fill="#5a6672">{_esc(self.xlabel)}</text>')
        if self.ylabel:
            cx, cy = 16, self.mt + self.plot_h / 2
            p.append(f'<text x="{cx}" y="{cy:.0f}" text-anchor="middle" '
                     f'font-family="system-ui,sans-serif" font-size="11" fill="#5a6672" '
                     f'transform="rotate(-90 {cx} {cy:.0f})">{_esc(self.ylabel)}</text>')
        self._draw_notes()

    def legend_box(self):
        if len(self.legend) <= 1:
            return
        x = self.ml + 8
        y = self.mt + 6 + 14 * len(self.notes)
        for i, (name, col) in enumerate(self.legend):
            yy = y + i * 15
            self.parts.append(f'<rect x="{x}" y="{yy - 7}" width="10" height="10" fill="{col}"/>')
            self.parts.append(f'<text x="{x + 15}" y="{yy + 1}" font-family="system-ui,sans-serif" '
                              f'font-size="10" fill="#33414c">{_esc(name)}</text>')

    def save(self, path):
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.W}" '
                f'height="{self.H}" viewBox="0 0 {self.W} {self.H}">')
        with open(path, "w", encoding="utf-8") as f:
            f.write(head + "\n" + "\n".join(self.parts) + "\n</svg>\n")
        return path


class BarChart(_Canvas):
    """竖向柱状图：适合类别数不多、标签较短的场景。"""

    def __init__(self, title, xlabel="", ylabel="", **kw):
        super().__init__(title, xlabel, ylabel, **kw)
        self.series = []

    def add_series(self, name, labels, values, color=None):
        self.series.append((name, list(labels), [float(v) for v in values], color or self.color()))
        return self

    def save(self, path):
        if not self.series:
            raise ValueError("没有数据")
        labels = self.series[0][1]
        allv = [v for s in self.series for v in s[2]]
        lo, hi, ticks = nice_ticks(0, max(allv) if allv else 1)
        self.frame(ticks, lo, hi)
        n = len(labels)
        k = len(self.series)
        slot = self.plot_w / max(n, 1)
        bw = slot * 0.72 / k
        for si, (name, _, vals, col) in enumerate(self.series):
            self.legend.append((name, col))
            for i, v in enumerate(vals):
                x = self.ml + slot * i + slot * 0.14 + bw * si
                y = self.y_of(v, lo, hi)
                h = self.mt + self.plot_h - y
                self.parts.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{max(h,0):.1f}" '
                    f'fill="{col}" opacity="0.88"/>')
                if k == 1:
                    self.parts.append(
                        f'<text x="{x + bw/2:.1f}" y="{y - 4:.1f}" text-anchor="middle" '
                        f'font-family="ui-monospace,monospace" font-size="9.5" '
                        f'fill="#33414c">{_esc(_fmt(v))}</text>')
        for i, lb in enumerate(labels):
            cx = self.ml + slot * (i + 0.5)
            self.parts.append(
                f'<text x="{cx:.1f}" y="{self.mt + self.plot_h + 15}" text-anchor="middle" '
                f'font-family="system-ui,sans-serif" font-size="9.5" fill="#5a6672">'
                f'{_esc(str(lb)[:8])}</text>')
        self.legend_box()
        return super().save(path)


class HBarChart(_Canvas):
    """横向条形图：适合标签长、类别多的场景。"""

    def __init__(self, title, xlabel="", ylabel="", **kw):
        super().__init__(title, xlabel, ylabel, ml=150, mb=48, **kw)
        self.labels = []
        self.values = []
        self.colors = []
        self._c = self.color()

    def add(self, labels, values, color=None):
        self.labels = list(labels)
        self.values = [float(v) for v in values]
        self._c = color or PALETTE[0]
        return self

    def save(self, path):
        n = len(self.labels)
        if n == 0:
            raise ValueError("没有数据")
        self._reserve_notes()
        lo, hi, ticks = nice_ticks(0, max(self.values))
        # 横向图：x 轴是数值，y 轴是类别
        p = self.parts
        p.append(f'<rect x="0" y="0" width="{self.W}" height="{self.H}" fill="#ffffff"/>')
        p.append(f'<text x="{self.W/2:.0f}" y="26" text-anchor="middle" '
                 f'font-family="system-ui,sans-serif" font-size="14" font-weight="600" '
                 f'fill="#1e2630">{_esc(self.title)}</text>')
        for t in ticks:
            x = self.ml + self.plot_w * (t - lo) / (hi - lo)
            p.append(f'<line x1="{x:.1f}" y1="{self.mt}" x2="{x:.1f}" '
                     f'y2="{self.mt + self.plot_h}" stroke="#e3e8ec" stroke-width="1"/>')
            p.append(f'<text x="{x:.1f}" y="{self.mt + self.plot_h + 15}" text-anchor="middle" '
                     f'font-family="ui-monospace,monospace" font-size="9.5" fill="#6b7883">'
                     f'{_esc(_fmt(t))}</text>')
        p.append(f'<line x1="{self.ml}" y1="{self.mt}" x2="{self.ml}" '
                 f'y2="{self.mt + self.plot_h}" stroke="#9aa7b2" stroke-width="1.2"/>')
        slot = self.plot_h / n
        bh = slot * 0.66
        for i, (lb, v) in enumerate(zip(self.labels, self.values)):
            y = self.mt + slot * i + (slot - bh) / 2
            w = self.plot_w * (v - lo) / (hi - lo)
            p.append(f'<rect x="{self.ml}" y="{y:.1f}" width="{max(w,0):.1f}" height="{bh:.1f}" '
                     f'fill="{self._c}" opacity="0.88"/>')
            p.append(f'<text x="{self.ml - 8}" y="{y + bh/2 + 3.5:.1f}" text-anchor="end" '
                     f'font-family="system-ui,sans-serif" font-size="9.5" fill="#33414c">'
                     f'{_esc(str(lb)[:16])}</text>')
            p.append(f'<text x="{self.ml + w + 6:.1f}" y="{y + bh/2 + 3.5:.1f}" '
                     f'font-family="ui-monospace,monospace" font-size="9.5" fill="#33414c">'
                     f'{_esc(_fmt(v))}</text>')
        if self.xlabel:
            p.append(f'<text x="{self.ml + self.plot_w/2:.0f}" y="{self.mt + self.plot_h + 34}" '
                     f'text-anchor="middle" font-family="system-ui,sans-serif" font-size="11" '
                     f'fill="#5a6672">{_esc(self.xlabel)}</text>')
        self._draw_notes()
        return super().save(path)


class LineChart(_Canvas):
    """折线图：支持多条线，以及一条「区间带」（用于预测区间）。"""

    def __init__(self, title, xlabel="", ylabel="", **kw):
        super().__init__(title, xlabel, ylabel, **kw)
        self.lines = []
        self.band = None
        self.xticks = None

    def add_band(self, xs, lows, highs, color="#C9A227", label="80% 预测区间"):
        self.band = (list(xs), [float(v) for v in lows], [float(v) for v in highs], color, label)
        return self

    def add_line(self, name, xs, ys, color=None, markers=True):
        self.lines.append((name, [float(x) for x in xs], [float(y) for y in ys],
                           color or self.color(), markers))
        return self

    def set_xticks(self, ticks):
        self.xticks = list(ticks)
        return self

    def save(self, path):
        vals = [v for l in self.lines for v in l[2]]
        if self.band:
            vals += self.band[1] + self.band[2]
        if not vals:
            raise ValueError("没有数据")
        lo, hi, ticks = nice_ticks(min(vals), max(vals))
        self.frame(ticks, lo, hi)
        xs_all = [x for l in self.lines for x in l[1]] + (self.band[0] if self.band else [])
        xmin, xmax = min(xs_all), max(xs_all)

        def X(x):
            if xmax == xmin:
                return self.ml + self.plot_w / 2
            return self.ml + self.plot_w * (x - xmin) / (xmax - xmin)

        if self.band:
            bs, bl, bh, bc, blabel = self.band
            top = " ".join(f"{X(x):.1f},{self.y_of(h, lo, hi):.1f}" for x, h in zip(bs, bh))
            bot = " ".join(f"{X(x):.1f},{self.y_of(l, lo, hi):.1f}"
                           for x, l in zip(reversed(bs), reversed(bl)))
            self.parts.append(f'<polygon points="{top} {bot}" fill="{bc}" opacity="0.20"/>')
            self.legend.append((blabel, bc))
        for name, xs, ys, col, mk in self.lines:
            pts = " ".join(f"{X(x):.1f},{self.y_of(y, lo, hi):.1f}" for x, y in zip(xs, ys))
            self.parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" '
                              f'stroke-width="2.2" stroke-linejoin="round"/>')
            if mk:
                for x, y in zip(xs, ys):
                    self.parts.append(f'<circle cx="{X(x):.1f}" cy="{self.y_of(y, lo, hi):.1f}" '
                                      f'r="3" fill="#fff" stroke="{col}" stroke-width="1.8"/>')
            self.legend.append((name, col))
        for x in (self.xticks or xs_all):
            self.parts.append(f'<text x="{X(x):.1f}" y="{self.mt + self.plot_h + 15}" '
                              f'text-anchor="middle" font-family="ui-monospace,monospace" '
                              f'font-size="9.5" fill="#5a6672">{_esc(_fmt(x))}</text>')
        self.legend_box()
        return super().save(path)


class ScatterChart(_Canvas):
    """散点图：支持分组、点标签、质心。"""

    def __init__(self, title, xlabel="", ylabel="", **kw):
        super().__init__(title, xlabel, ylabel, **kw)
        self.groups = []
        self.notes = []

    def add_group(self, name, xs, ys, color=None, marker="circle", labels=None):
        self.groups.append((name, [float(x) for x in xs], [float(y) for y in ys],
                            color or self.color(), marker, labels))
        return self

    def add_centroids(self, xs, ys, color="#1e2630"):
        self.groups.append(("质心", [float(x) for x in xs], [float(y) for y in ys],
                            color, "cross", None))
        return self

    def add_note(self, text):
        self.notes.append(text)
        return self

    def save(self, path):
        xs_all = [x for g in self.groups for x in g[1]]
        ys_all = [y for g in self.groups for y in g[2]]
        if not xs_all:
            raise ValueError("没有数据")
        xlo, xhi, xt = nice_ticks(min(xs_all), max(xs_all))
        ylo, yhi, yt = nice_ticks(min(ys_all), max(ys_all))
        self.frame(yt, ylo, yhi)

        def X(x):
            return self.ml + self.plot_w * (x - xlo) / (xhi - xlo)

        # x 轴刻度
        for t in xt:
            x = X(t)
            self.parts.append(f'<line x1="{x:.1f}" y1="{self.mt + self.plot_h}" '
                              f'x2="{x:.1f}" y2="{self.mt + self.plot_h + 4}" '
                              f'stroke="#9aa7b2"/>')
            self.parts.append(f'<text x="{x:.1f}" y="{self.mt + self.plot_h + 16}" '
                              f'text-anchor="middle" font-family="ui-monospace,monospace" '
                              f'font-size="9.5" fill="#6b7883">{_esc(_fmt(t))}</text>')
        for name, xs, ys, col, mk, labels in self.groups:
            if name != "质心":
                self.legend.append((name, col))
            for i, (x, y) in enumerate(zip(xs, ys)):
                cx, cy = X(x), self.y_of(y, ylo, yhi)
                if mk == "square":
                    self.parts.append(f'<rect x="{cx-3:.1f}" y="{cy-3:.1f}" width="6" height="6" '
                                      f'fill="none" stroke="{col}" stroke-width="1.8"/>')
                elif mk == "triangle":
                    self.parts.append(f'<polygon points="{cx:.1f},{cy-4:.1f} {cx-3.6:.1f},{cy+3:.1f} '
                                      f'{cx+3.6:.1f},{cy+3:.1f}" fill="none" stroke="{col}" '
                                      f'stroke-width="1.8"/>')
                elif mk == "cross":
                    self.parts.append(f'<line x1="{cx-4:.1f}" y1="{cy-4:.1f}" x2="{cx+4:.1f}" '
                                      f'y2="{cy+4:.1f}" stroke="{col}" stroke-width="2"/>')
                    self.parts.append(f'<line x1="{cx-4:.1f}" y1="{cy+4:.1f}" x2="{cx+4:.1f}" '
                                      f'y2="{cy-4:.1f}" stroke="{col}" stroke-width="2"/>')
                else:
                    self.parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.2" fill="none" '
                                      f'stroke="{col}" stroke-width="1.8"/>')
                if labels:
                    self.parts.append(f'<text x="{cx + 6:.1f}" y="{cy - 5:.1f}" '
                                      f'font-family="system-ui,sans-serif" font-size="9" '
                                      f'fill="#33414c">{_esc(labels[i])}</text>')
        self.legend_box()

        return super().save(path)
