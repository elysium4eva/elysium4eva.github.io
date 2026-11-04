#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rec_rank_eval.py -- 双塔召回与 HR@K / NDCG@K 评估

用途：
    实现一个极简双塔（用户塔 + 物品塔，内积打分，BPR 式训练），
    并按不同 K 计算 HR@K 与 NDCG@K，展示「K 越大指标越高但一屏放不下」的现象。

输入：
    合成的用户-物品交互数据
输出：
    不同 K 下的 HR 与 NDCG、双塔与随机基线的对比
依赖：
    numpy（可选 chartlib 出 SVG 图）
对应课件：
    模块五第三章 §5 算法 5-3 双塔、算法 5-7 评估指标

说明：
    真实双塔用神经网络编码特征；这里用 embedding 内积演示同样的结构，
    重点看「双塔可离线预计算物品向量」这一点带来的工程价值。
"""

import numpy as np

N_USER = 150
N_ITEM = 300
DIM = 8
SEED = 33


def make_data(seed=SEED):
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 1, (N_USER, 3))
    V = rng.normal(0, 1, (N_ITEM, 3))
    score = U @ V.T
    pairs = []
    for u in range(N_USER):
        k = rng.integers(3, 8)
        for i in np.argsort(-score[u])[:k]:
            pairs.append((u, int(i)))
    rng.shuffle(pairs)
    cut = int(len(pairs) * 0.8)
    return pairs[:cut], pairs[cut:]


def to_matrix(pairs):
    M = np.zeros((N_USER, N_ITEM))
    for u, i in pairs:
        M[u, i] = 1.0
    return M


def train_two_tower(M, dim=DIM, epochs=80, lr=0.05, reg=0.01, seed=SEED):
    """双塔：两个 embedding 矩阵，内积打分，BPR 成对损失。"""
    rng = np.random.default_rng(seed)
    EU = rng.normal(0, 0.1, (N_USER, dim))
    EI = rng.normal(0, 0.1, (N_ITEM, dim))
    pos = [(u, i) for u in range(N_USER) for i in range(N_ITEM) if M[u, i] > 0]
    for _ in range(epochs):
        rng.shuffle(pos)
        for u, i in pos:
            j = int(rng.integers(0, N_ITEM))
            if M[u, j] > 0:
                continue
            x = float(EU[u] @ (EI[i] - EI[j]))
            g = 1.0 / (1.0 + np.exp(x))
            EU[u] += lr * (g * (EI[i] - EI[j]) - reg * EU[u])
            EI[i] += lr * (g * EU[u] - reg * EI[i])
            EI[j] += lr * (-g * EU[u] - reg * EI[j])
    return EU, EI


def hr_ndcg(scores, M, test_by_user, k):
    hr, ndcg, n = 0, 0.0, 0
    for u, items in test_by_user.items():
        s = scores(u).copy()
        s[M[u] > 0] = -1e9
        top = np.argsort(-s)[:k]
        hits = [1 if int(it) in items else 0 for it in top]
        if any(hits):
            hr += 1
            dcg = sum(h / np.log2(p + 2) for p, h in enumerate(hits))
            idcg = sum(1 / np.log2(p + 2) for p in range(min(len(items), k)))
            ndcg += dcg / idcg
        n += 1
    return hr / max(n, 1), ndcg / max(n, 1)


def demo():
    tr, te = make_data()
    M = to_matrix(tr)
    EU, EI = train_two_tower(M)
    R = EU @ EI.T

    test_by_user = {}
    for u, i in te:
        test_by_user.setdefault(u, set()).add(i)

    print("=== 双塔召回 + HR@K / NDCG@K ===")
    print(f"用户 {N_USER} | 物品 {N_ITEM} | 训练交互 {len(tr)} | 测试交互 {len(te)}")
    print(f"物品向量维度 {DIM} —— 可离线预计算，线上只算用户向量与近邻\n")

    print(f"{'K':>4} {'HR@K':>8} {'NDCG@K':>9} {'随机 HR':>9}")
    rows = []
    rng = np.random.default_rng(1)
    for k in (1, 5, 10, 15, 20):
        hr, nd = hr_ndcg(lambda u: R[u], M, test_by_user, k)
        rand_hr = k / N_ITEM
        rows.append((k, hr, nd, rand_hr))
        print(f"{k:>4} {hr*100:>7.1f}% {nd*100:>8.1f}% {rand_hr*100:>8.1f}%")

    print("\n观察：K 越大 HR 越高（更容易命中），但 NDCG 增长变慢")
    print("      —— 因为位置越靠后，命中带来的折损奖励越小。")
    print("      这也是为什么「一屏只有前 K 个位置」，不能靠堆 K 提升效果。")
    return rows


# ---------------------------------------------------------------- 可视化
def save_visual(path='visuals/rec_rank_eval.svg'):
    """输出 HR@K / NDCG@K 随 K 变化的曲线。"""
    from chartlib import LineChart
    rows = demo()
    ks = [r[0] for r in rows]
    hr = [round(r[1] * 100, 1) for r in rows]
    nd = [round(r[2] * 100, 1) for r in rows]
    ch = LineChart('HR@K 与 NDCG@K 随 K 的变化', 'K（推荐列表长度）', '百分比（%）')
    ch.add_line('HR@K', ks, hr)
    ch.add_line('NDCG@K', ks, nd)
    ch.add_note('K 越大 HR 越高，但 NDCG 增长变慢——位置越靠后奖励折损越大')
    return ch.save(path)


if __name__ == '__main__':
    demo()
