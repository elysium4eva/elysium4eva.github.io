#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ch3_influence_greedy.py -- 影响力最大化：贪心算法（极简版）

用途：
    在一个只有 8 个节点的小社交网络上，演示「影响力最大化」为什么必须用贪心：
    每次都选「边际收益最大」的那个节点，而不是「朋友最多」的那个。

输入：
    内置 8 节点有向图（边表示「谁会影响谁」）+ 每条边的传播概率
输出：
    贪心选出的种子集、按度数选的种子集、随机种子集，三者的期望触达人数
依赖：
    numpy、viz_png（Pillow）
对应课件：
    模块五第三章 第三章 §4 算法 4-1（影响力最大化与贪心）

说明：
    传播模型用 IC（独立级联）：种子先激活，每步以边上概率尝试激活邻居，
    直到没有新节点被激活。每个种子集重复模拟 200 次取平均。
"""

import numpy as np

NODES = list('ABCDEFGH')
EDGES = [
    ('A', 'B', 0.6), ('A', 'C', 0.5), ('A', 'D', 0.3),
    ('B', 'C', 0.5), ('B', 'E', 0.4),
    ('C', 'D', 0.5), ('C', 'F', 0.4),
    ('D', 'G', 0.4),
    ('E', 'F', 0.5), ('E', 'H', 0.3),
    ('F', 'G', 0.5), ('F', 'H', 0.4),
    ('G', 'H', 0.5),
]
K = 3            # 选几个种子
SIMS = 200
SEED = 7


def adjacency():
    adj = {n: [] for n in NODES}
    for u, v, p in EDGES:
        adj[u].append((v, p))
    return adj


def ic_spread(seeds, adj, rng):
    """IC 模型的一次模拟，返回被激活的节点总数。"""
    active = set(seeds)
    frontier = list(seeds)
    while frontier:
        nxt = []
        for u in frontier:
            for v, p in adj[u]:
                if v not in active and rng.random() < p:
                    active.add(v)
                    nxt.append(v)
        frontier = nxt
    return len(active)


def expected_spread(seeds, adj, sims=SIMS, seed=SEED):
    rng = np.random.default_rng(seed)
    return float(np.mean([ic_spread(seeds, adj, rng) for _ in range(sims)]))


def greedy(adj, k=K, sims=120):
    """贪心：每轮选边际收益最大的节点。"""
    chosen, curve = [], []
    base = 0.0
    for _ in range(k):
        best, best_gain = None, -1.0
        for v in NODES:
            if v in chosen:
                continue
            s = expected_spread(chosen + [v], adj, sims=sims)
            if s - base > best_gain:
                best, best_gain = v, s - base
        chosen.append(best)
        base = expected_spread(chosen, adj, sims=sims)
        curve.append((best, best_gain, base))
        print('  第 %d 轮：选 %s，边际增益 %+.2f，累计触达 %.2f' % (
            len(chosen), best, best_gain, base))
    return chosen, curve


def degree_based(adj):
    deg = {n: len(adj[n]) for n in NODES}
    return sorted(NODES, key=lambda n: -deg[n])[:K]


def demo():
    adj = adjacency()
    print('=== 影响力最大化：贪心 vs 按度数 vs 随机 ===')
    print('节点 %d 个，有向边 %d 条，选 %d 个种子\\n' % (len(NODES), len(EDGES), K))

    print('--- 贪心（每轮选边际收益最大）---')
    g, curve = greedy(adj)

    d = degree_based(adj)
    rng = np.random.default_rng(3)
    r = list(rng.choice(NODES, size=K, replace=False))

    print('\\n--- 三种选法的期望触达（含种子自身）---')
    rows = []
    for name, seeds in [('贪心', g), ('按度数', d), ('随机', r)]:
        s = expected_spread(seeds, adj)
        rows.append((name, seeds, s))
        print('  %-6s 种子 %-16s 期望触达 %5.2f / %d 个节点' % (
            name, ''.join(seeds), s, len(NODES)))

    print('\\n结论：')
    print('  1) 按度数选（朋友最多）并不等于影响力最大——度数只看直接邻居，')
    print('     而影响力是「连锁反应」，要看二阶、三阶能碰到谁。')
    print('  2) 贪心的核心是「边际收益」：同一批候选，加入已有种子后增益会下降。')
    print('  3) 所以贪心要一轮一轮选，不能一次选完——这是次模性的直接应用。')
    print('  4) 代价是慢：每轮要对每个候选都模拟一次，节点多时要近似（如 CELF）。')
    return rows, curve, g


def save_png(path='visuals/ch4_influence_greedy.png'):
    from viz_png import Canvas, RED, RED_L, GOLD, GOLD_L, INK, MUTED, LINE
    rows, curve, g = demo()
    import math

    cv = Canvas(1200, 700)
    cv.title('影响力最大化：贪心为什么比「朋友最多」更好')

    # 左：网络图
    pos = {'A': (150, 250), 'B': (300, 150), 'C': (300, 330), 'D': (300, 470),
           'E': (470, 150), 'F': (470, 330), 'G': (470, 470), 'H': (640, 300)}
    for u, v, p in EDGES:
        x0, y0 = pos[u]; x1, y1 = pos[v]
        col = GOLD if u in g else LINE
        cv.arrow(x0, y0 + 20, x1, y1 - 20, color=col, width=2 if u in g else 1)

    for n in NODES:
        x, y = pos[n]
        if n in g:
            cv.node(x, y, 24, GOLD_L, GOLD, 3)
        else:
            cv.node(x, y, 24, RED_L, RED, 2)
        cv.text(x, y, n, 18, INK, True, 'mm')

    cv.text(120, 545, '金框 = 贪心选出的 3 个种子', 14, GOLD, True)
    cv.text(120, 570, '金色箭头 = 种子出发的传播边', 14, MUTED)
    cv.text(120, 600, '小网络只有 8 个节点、13 条边', 13, MUTED)
    cv.text(120, 622, '够看清「连锁反应」即可', 13, MUTED)

    # 右：对比柱状
    bx, by, bw = 780, 560, 300
    cv.text(700, 120, '三种选法的期望触达（含种子）', 16, INK, True)
    maxv = len(NODES)
    for i, (name, seeds, s) in enumerate(rows):
        y = 170 + i * 120
        col = GOLD if name == '贪心' else (RED if name == '按度数' else MUTED)
        cv.text(700, y, '%s（%s）' % (name, ''.join(seeds)), 15, INK, True)
        w = int(bw * s / maxv)
        cv.rect(700, y + 26, w, 26, fill=col, outline=col, r=4)
        cv.rect(700, y + 26, bw, 26, fill=None, outline=LINE, width=1, r=4)
        cv.text(700 + w + 10, y + 28, '%.2f' % s, 16, col, True)

    cv.text(700, 545, '最优值上限 = 8（全网被激活）', 13, MUTED)
    cv.text(700, 572, '贪心已达 %.0f%%' % (rows[0][2] / len(NODES) * 100), 14, GOLD, True)
    cv.text(700, 600, '每轮都要重算边际收益', 13, MUTED)
    cv.text(700, 622, '这就是「贪心」的代价与收益', 13, MUTED)

    cv.text(30, 665, '数据：8 节点 IC 模型，200 次模拟；程序 ch3_influence_greedy.py', 12, MUTED)
    return cv.save(path)


if __name__ == '__main__':
    demo()
