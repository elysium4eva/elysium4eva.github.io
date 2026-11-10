#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mkt_attribution.py -- 多触点归因：五种口径，五个结论

用途：
    用讲义里的 10 条转化路径，分别用首次、末次、线性、马尔可夫链、Shapley
    五种方法分配功劳，展示「同一批数据，五种结论」。

输入：
    内置的 10 条路径（触点用 V=短视频、S=搜索、K=达人 表示）
输出：
    五种方法的分配合计与排名，以及排名变化的解读
依赖：
    numpy（可选 chartlib 出 SVG 图）
对应课件：
    模块五第四章 §4 效果评估（归因四口径 + 手工演算）
"""

from itertools import permutations
from collections import defaultdict

# 10 条转化路径（与讲义手工演算案例一致）
PATHS = [
    ['V', 'S'], ['V', 'K'], ['S'], ['K', 'V', 'S'], ['V'],
    ['S', 'K'], ['K'], ['V', 'S', 'K'], ['S', 'S'], ['V', 'K', 'S'],
]
CHANNELS = ['V', 'S', 'K']
NAME = {'V': '短视频', 'S': '搜索', 'K': '达人'}


def first_touch(paths):
    c = defaultdict(float)
    for p in paths:
        c[p[0]] += 1.0
    return dict(c)


def last_touch(paths):
    c = defaultdict(float)
    for p in paths:
        c[p[-1]] += 1.0
    return dict(c)


def linear(paths):
    c = defaultdict(float)
    for p in paths:
        share = 1.0 / len(p)
        for ch in p:
            c[ch] += share
    return dict(c)


def markov(paths):
    """移除效应：删掉某渠道后，整体「到达转化」的概率下降比例。

    简化版：把路径集合看成经验分布，P(转化) = 路径数/总路径。
    移除渠道 ch 后，凡是经过 ch 的路径视为作废（无法转化），
    计算剩余路径占比，即为移除效应。
    """
    total = len(paths)
    res = {}
    for ch in CHANNELS:
        survive = sum(1 for p in paths if ch not in p)
        drop = (total - survive) / total
        res[ch] = drop
    # 归一化成「功劳占比」
    s = sum(res.values()) or 1.0
    return {k: v / s * total for k, v in res.items()}


def shapley(paths):
    """把「渠道集合」当作玩家，v(S) = 只包含 S 中渠道的转化数。"""
    def v(subset):
        subset = set(subset)
        return sum(1 for p in paths if set(p).issubset(subset))

    res = {}
    for ch in CHANNELS:
        others = [c for c in CHANNELS if c != ch]
        phi = 0.0
        for r in range(len(others) + 1):
            for comb in permutations(others, r):
                S = list(comb)
                w = (len(S) * (len(others) - len(S))) / (len(others) + 1) if len(others) > 0 else 1
                # 用标准 Shapley 权重
                import math
                w = (math.factorial(len(S)) * math.factorial(len(CHANNELS) - len(S) - 1)
                     / math.factorial(len(CHANNELS)))
                phi += w * (v(S + [ch]) - v(S))
        res[ch] = phi
    s = sum(res.values()) or 1.0
    return {k: v / s * len(paths) for k, v in res.items()}


def demo():
    print('=== 多触点归因：五种口径，五个结论 ===')
    print('转化路径 %d 条：' % len(PATHS))
    for i, p in enumerate(PATHS, 1):
        print('  P%-2d %s' % (i, ' -> '.join(NAME[c] for c in p)))
    print()

    methods = [
        ('首次归因', first_touch(PATHS)),
        ('末次归因', last_touch(PATHS)),
        ('线性归因', linear(PATHS)),
        ('马尔可夫', markov(PATHS)),
        ('Shapley', shapley(PATHS)),
    ]

    print('%-10s %10s %10s %10s   %s' % ('方法', '短视频', '搜索', '达人', '排名第一'))
    print('-' * 62)
    ranks = {}
    for name, alloc in methods:
        vals = [alloc.get(c, 0.0) for c in CHANNELS]
        best = CHANNELS[vals.index(max(vals))]
        ranks[name] = NAME[best]
        print('%-10s %10.2f %10.2f %10.2f   %s' % (
            name, vals[0], vals[1], vals[2], NAME[best]))
    print()

    print('--- 排名变化 ---')
    for name, top in ranks.items():
        print('  %-10s 认为最强的是：%s' % (name, top))
    print()

    print('结论：')
    print('  1) 首次归因说短视频（拉新），末次归因说搜索（收割）—— 同一批数据，相反结论。')
    print('  2) 线性归因偏向「到处都出现」的渠道（搜索在 8 条路径里出现）。')
    print('  3) 马尔可夫与 Shapley 更接近「边际贡献」，但仍受路径数据本身的限制。')
    print('  4) 所以报 ROI 必须同时报口径；差异大的渠道要用增量实验校准。')
    return methods, ranks


def save_visual(path='visuals/mkt_attribution.svg'):
    """五法对比柱状图。"""
    from chartlib import BarChart
    methods, _ = demo()
    ch = BarChart('五种归因口径：同一个渠道，不同名次', '方法', '分到的转化次数')
    for c in CHANNELS:
        ch.add_series(NAME[c], [m[0] for m in methods],
                      [round(m[1].get(c, 0.0), 2) for m in methods])
    ch.add_note('首次说短视频最强，末次说搜索最强——口径决定结论，不是数据决定结论')
    return ch.save(path)


if __name__ == '__main__':
    demo()
