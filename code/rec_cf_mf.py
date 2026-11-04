#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rec_cf_mf.py -- 协同过滤与矩阵分解（BPR）

用途：
    用同一份隐式反馈（点击）数据，对比三种推荐的 HR@K：
    1) UserCF  2) ItemCF  3) BPR 矩阵分解（成对损失，SGD 训练）

输入：
    合成的用户-物品交互数据（隐式反馈，1 表示有点击）
输出：
    三种方法的 HR@K 与 NDCG@K 对比、训练损失曲线
依赖：
    numpy（可选 chartlib 出 SVG 图）
对应课件：
    模块五第三章 §5 算法 5-1 协同过滤、算法 5-2 矩阵分解（BPR）

说明：
    真实场景用 MovieLens 等公开数据；这里用合成数据保证开箱即跑。
"""

import numpy as np

N_USER = 120
N_ITEM = 60
N_INTER = 1400
K = 10
SEED = 21


def make_data(seed=SEED):
    """合成隐式反馈：用户有潜在偏好分组，物品有潜在主题。"""
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 1, (N_USER, 4))
    V = rng.normal(0, 1, (N_ITEM, 4))
    score = U @ V.T
    # 每个用户取分数最高的若干物品作为正样本
    inter = []
    for u in range(N_USER):
        k = rng.integers(4, 12)
        top = np.argsort(-score[u])[:k]
        for i in top:
            inter.append((u, int(i)))
    # 加一点噪声点击
    extra = rng.integers(0, N_ITEM, N_INTER - len(inter))
    eu = rng.integers(0, N_USER, len(extra))
    inter += list(zip(eu.tolist(), extra.tolist()))
    inter = list(dict.fromkeys(inter))
    return inter


def split(inter, ratio=0.8, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(inter))
    cut = int(len(inter) * ratio)
    tr = [inter[i] for i in idx[:cut]]
    te = [inter[i] for i in idx[cut:]]
    return tr, te


def user_item_matrix(inter):
    M = np.zeros((N_USER, N_ITEM))
    for u, i in inter:
        M[u, i] = 1.0
    return M


def user_cf_scores(M, u, topk=20):
    """UserCF：找相似用户，用他们的交互加权。"""
    sims = []
    for v in range(M.shape[0]):
        if v == u:
            continue
        a, b = M[u], M[v]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom > 0:
            sims.append((v, float(a @ b / denom)))
    sims.sort(key=lambda x: -x[1])
    s = np.zeros(N_ITEM)
    for v, w in sims[:topk]:
        s += w * M[v]
    return s


def item_cf_scores(M, u, topk=20):
    """ItemCF：用用户已交互物品的相似物品打分。"""
    sim = M.T @ M
    norm = np.linalg.norm(M, axis=0) + 1e-9
    sim = sim / np.outer(norm, norm)
    pos = np.where(M[u] > 0)[0]
    if len(pos) == 0:
        return np.zeros(N_ITEM)
    s = sim[pos].sum(axis=0)
    return s


def bpr_train(M, dim=16, epochs=180, lr=0.02, reg=0.02, seed=SEED):
    """BPR 矩阵分解：成对损失 + SGD。"""
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 0.1, (N_USER, dim))
    Q = rng.normal(0, 0.1, (N_ITEM, dim))
    pos_pairs = [(u, i) for u in range(N_USER) for i in range(N_ITEM) if M[u, i] > 0]
    losses = []
    for ep in range(epochs):
        rng.shuffle(pos_pairs)
        tot = 0.0
        for u, i in pos_pairs:
            j = int(rng.integers(0, N_ITEM))
            if M[u, j] > 0:
                continue
            x = float(P[u] @ (Q[i] - Q[j]))
            g = 1.0 / (1.0 + np.exp(x))          # -sigmoid(-x)
            tot += float(np.log1p(np.exp(-x)))
            P[u] += lr * (g * (Q[i] - Q[j]) - reg * P[u])
            Q[i] += lr * (g * P[u] - reg * Q[i])
            Q[j] += lr * (-g * P[u] - reg * Q[j])
        losses.append(tot / max(len(pos_pairs), 1))
    return P, Q, losses


def evaluate(scores_fn, M, test, k=K):
    """计算 HR@K 与 NDCG@K。"""
    hr, ndcg, n = 0, 0.0, 0
    test_by_user = {}
    for u, i in test:
        test_by_user.setdefault(u, set()).add(i)
    for u, items in test_by_user.items():
        s = scores_fn(u)
        s = s.copy()
        s[M[u] > 0] = -1e9                    # 屏蔽已交互
        top = np.argsort(-s)[:k]
        hits = [1 if it in items else 0 for it in top]
        if any(hits):
            hr += 1
            dcg = sum(h / np.log2(pos + 2) for pos, h in enumerate(hits))
            idcg = sum(1 / np.log2(p + 2) for p in range(min(len(items), k)))
            ndcg += dcg / idcg
        n += 1
    return hr / max(n, 1), ndcg / max(n, 1)


def demo():
    inter = make_data()
    tr, te = split(inter)
    M = user_item_matrix(tr)
    print("=== 协同过滤与矩阵分解 ===")
    print(f"用户 {N_USER} | 物品 {N_ITEM} | 交互 {len(inter)} | 训练 {len(tr)} | 测试 {len(te)}")
    print(f"稀疏度 {(1 - M.sum()/(N_USER*N_ITEM))*100:.1f}%")

    P, Q, losses = bpr_train(M)
    R = P @ Q.T

    res = {}
    for name, fn in [
        ("UserCF", lambda u: user_cf_scores(M, u)),
        ("ItemCF", lambda u: item_cf_scores(M, u)),
        ("BPR-MF", lambda u: R[u]),
    ]:
        hr, nd = evaluate(fn, M, te)
        res[name] = (hr, nd)
        print(f"  {name:<8} HR@{K} {hr*100:5.1f}%   NDCG@{K} {nd*100:5.1f}%")

    print(f"\nBPR 训练损失：第 1 轮 {losses[0]:.4f} -> 第 {len(losses)} 轮 {losses[-1]:.4f}")
    best = max(res.items(), key=lambda kv: kv[1][0])
    print(f"\n本数据上 HR@10 最高的是：{best[0]}（{best[1][0]*100:.1f}%）")
    print("说明：合成数据规模小、主题结构清晰，基于相似度的方法已接近上限；")
    print("      BPR 的优势通常要在更大规模、更稀疏、更偏排序目标的数据上才明显。")
    return res, losses


# ---------------------------------------------------------------- 可视化
def save_visual(path='visuals/rec_cf_mf.svg'):
    """输出三种方法的 HR@K 与 NDCG@K 对比。"""
    from chartlib import BarChart
    res, _ = demo()
    names = list(res.keys())
    hr = [round(v[0] * 100, 1) for v in res.values()]
    nd = [round(v[1] * 100, 1) for v in res.values()]
    ch = BarChart(f'三种推荐方法的 HR@{K} 与 NDCG@{K}', '方法', '百分比（%）')
    ch.add_series(f'HR@{K}', names, hr)
    ch.add_series(f'NDCG@{K}', names, nd)
    ch.add_note('BPR 直接优化排序目标，通常在隐式反馈上优于相似度方法')
    return ch.save(path)


if __name__ == '__main__':
    demo()
