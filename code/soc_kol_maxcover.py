#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
soc_kol_maxcover.py -- 影响力最大化：独立级联模型 + 贪心选点

用途：
    在给定预算只能合作 K 个达人的情况下，用独立级联（IC）模型模拟传播，
    用贪心算法选覆盖最大的 K 个种子，并与随机选择对比。

输入：
    合成的小型有向社交图（不需要外部数据）
输出：
    贪心 vs 随机 的影响覆盖对比、每步边际增益、最终种子集合
依赖：
    numpy（可选 chartlib 出 SVG 图）
对应课件：
    模块五第三章 §4 算法 4-1 影响力最大化

说明：
    真实场景中需要平台的社交关系图；这里用合成图演示算法逻辑，
    重点看「边际增益递减」与「贪心优于随机」这两个现象。
"""

import os

import numpy as np

P_ACTIVATE = 0.22          # 每条边的激活概率
N_NODES = 40
N_SIM = 400                # 蒙特卡洛模拟次数（越大越平滑）
AVG_DEGREE = 4


def build_graph(n=N_NODES, deg=AVG_DEGREE, seed=7):
    """构造一个有社区结构的随机有向图（邻接表）。"""
    rng = np.random.default_rng(seed)
    adj = [[] for _ in range(n)]
    # 三个社区，社区内连接更密（模拟"同好圈"）
    groups = [range(0, n // 3), range(n // 3, 2 * n // 3), range(2 * n // 3, n)]
    for g in groups:
        members = list(g)
        for u in members:
            for _ in range(deg):
                v = members[rng.integers(0, len(members))]
                if v != u:
                    adj[u].append(int(v))
    # 少量跨社区边（桥接）
    for _ in range(n // 6):
        u = int(rng.integers(0, n))
        v = int(rng.integers(0, n))
        if v != u:
            adj[u].append(v)
    return adj


def simulate_ic(adj, seeds, p=P_ACTIVATE, rng=None):
    """一次 IC 模拟，返回被激活的节点数。"""
    if rng is None:
        rng = np.random.default_rng()
    n = len(adj)
    active = np.zeros(n, dtype=bool)
    active[list(seeds)] = True
    frontier = list(seeds)
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if not active[v] and rng.random() < p:
                    active[v] = True
                    nxt.append(v)
        frontier = nxt
    return int(active.sum())


def spread(adj, seeds, n_sim=N_SIM, seed=0):
    """多次模拟取平均，估计 sigma(S)。"""
    rng = np.random.default_rng(seed)
    vals = [simulate_ic(adj, seeds, rng=rng) for _ in range(n_sim)]
    return float(np.mean(vals)), float(np.std(vals) / np.sqrt(n_sim))


def greedy(adj, k, n_sim=N_SIM):
    """贪心选点：每步取边际增益最大的节点。"""
    chosen, gains = [], []
    for _ in range(k):
        best_v, best_gain = None, -1.0
        base = spread(adj, chosen, n_sim)[0] if chosen else 0.0
        for v in range(len(adj)):
            if v in chosen:
                continue
            val = spread(adj, chosen + [v], n_sim)[0]
            gain = val - base
            if gain > best_gain:
                best_gain, best_v = gain, v
        chosen.append(best_v)
        gains.append(best_gain)
    return chosen, gains


def random_pick(adj, k, seed=3):
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.choice(len(adj), size=k, replace=False)]


def demo():
    adj = build_graph()
    edges = sum(len(a) for a in adj)
    print("=== 影响力最大化（独立级联模型）===")
    print(f"节点数 {len(adj)} | 边数 {edges} | 平均出度 {edges/len(adj):.1f} | 激活概率 p={P_ACTIVATE}")

    k = 6
    seeds_g, gains = greedy(adj, k)
    seeds_r = random_pick(adj, k)

    m_g, se_g = spread(adj, seeds_g)
    m_r, se_r = spread(adj, seeds_r)

    print(f"\n贪心选点 {k} 个：{seeds_g}")
    print(f"  期望覆盖 {m_g:.1f} 人（标准误 {se_g:.2f}），覆盖率 {m_g/len(adj)*100:.0f}%")
    print(f"随机选点 {k} 个：{seeds_r}")
    print(f"  期望覆盖 {m_r:.1f} 人（标准误 {se_r:.2f}），覆盖率 {m_r/len(adj)*100:.0f}%")
    print(f"\n贪心相对随机的提升：{(m_g/max(m_r,1e-9)-1)*100:.1f}%")

    print("\n=== 每步边际增益（应递减）===")
    for i, g in enumerate(gains, 1):
        print(f"  第 {i} 个种子：边际增益 {g:5.2f} 人")

    return {"seeds_greedy": seeds_g, "gains": gains,
            "spread_greedy": m_g, "spread_random": m_r}


# ---------------------------------------------------------------- 可视化
def save_visual(path='visuals/soc_kol_maxcover.svg'):
    """输出边际增益随已选数量递减的曲线图。"""
    from chartlib import LineChart
    r = demo()
    gains = r["gains"]
    xs = list(range(1, len(gains) + 1))
    ch = LineChart('影响力最大化：每步边际增益递减', '已选达人数量 K', '该步边际增益（人）')
    ch.add_line('贪心选点', xs, [round(g, 2) for g in gains])
    ch.add_note(f'K={len(gains)} 时总覆盖 {r["spread_greedy"]:.0f} 人，'
                f'随机选仅 {r["spread_random"]:.0f} 人')
    return ch.save(path)


if __name__ == '__main__':
    demo()
