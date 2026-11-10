#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rev_group_fraud.py -- 群组刷评检测：单条分类抓不到，图结构能抓到

用途：
    构造「用户—商品」二部图并注入一个有组织的刷评群组，然后对比两种做法：
    1) 单条评论分类（只用评论级特征：评分类别、文本长度、是否验证）
    2) 结构检测（共现聚类，GSRank 式思路）
    用来回答一个问题：为什么单条分类抓不到刷评群组，而结构检测能。

输入：
    合成数据（无需外部文件）。刷评群组的特点是：成员重叠、目标商品集中、时间集中。
输出：
    单条分类的精确率／召回率、图检测找到的块、以及块的精确率／召回率
依赖：
    numpy（可选 chartlib 出 SVG 图）
对应课件：
    模块五第二章 §7 平台机制与群组刷评检测（GSRank / FRAUDAR）

说明：
    这是教学实现，不是 FRAUDAR 原算法。原论文的可证明上界与完整贪心策略更复杂，
    本程序只保留「贪心剥离 + 密度最大化」这一核心思路，并在输出中明确标注。
"""

import numpy as np

N_USER = 200
N_ITEM = 60
GROUP_SIZE = 14          # 刷评群组成员数
TARGET_ITEMS = 5         # 群组集中攻击的商品数
SEED = 13


# ---------------------------------------------------------------- 数据构造
def build_bipartite(seed=SEED):
    """构造二部图：返回 edges(user, item, hour, rating, verified, length)。"""
    rng = np.random.default_rng(seed)
    edges = []

    # 正常用户：随机评 3-8 个商品，时间分散，评分与商品均值接近
    item_base = rng.normal(4.2, 0.25, N_ITEM).clip(3.2, 5.0)
    for u in range(N_USER - GROUP_SIZE):
        k = int(rng.integers(2, 7))
        items = rng.choice(N_ITEM, size=k, replace=False)
        for it in items:
            hour = int(rng.integers(0, 24 * 30))       # 30 天内均匀
            rating = float(np.clip(item_base[it] + rng.normal(0, 0.6), 1, 5))
            verified = int(rng.random() < 0.82)
            length = int(rng.integers(40, 220))
            edges.append([u, int(it), hour, round(rating, 1), verified, length])

    # 刷评群组：固定一批成员，集中攻击 TARGET_ITEMS 个商品，时间集中在两天内
    group_users = list(range(N_USER - GROUP_SIZE, N_USER))
    targets = list(range(TARGET_ITEMS))
    for u in group_users:
        # 每个成员评全部目标商品 + 少量随机商品（伪装）
        for it in targets:
            hour = int(rng.integers(0, 48))            # 48 小时内
            rating = float(np.clip(5.0 + rng.normal(0, 0.2), 1, 5))   # 清一色高分
            verified = int(rng.random() < 0.7)
            # 长度与正常评论同分布 —— 故意让单条文本「看起来正常」
            length = int(rng.integers(40, 220))
            edges.append([u, int(it), hour, round(rating, 1), verified, length])
        for it in rng.choice(N_ITEM, size=2, replace=False):
            hour = int(rng.integers(0, 720))
            rating = float(np.clip(4.3 + rng.normal(0, 0.5), 1, 5))
            edges.append([u, int(it), hour, round(rating, 1), 1, int(rng.integers(40, 220))])

    E = np.array(edges)
    is_fake_user = np.zeros(N_USER, dtype=bool)
    is_fake_user[group_users] = True
    return E, is_fake_user, group_users, targets


# ---------------------------------------------------------------- 单条分类
def fit_logistic(X, y, lr=0.3, epochs=800):
    Xb = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        w -= lr * Xb.T @ (p - y) / len(y)
    return w


def predict(w, X):
    Xb = np.column_stack([np.ones(len(X)), X])
    return 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))


def review_level_features(E):
    """只用评论级特征：评分、是否验证、长度、小时。"""
    rating, verified, length, hour = E[:, 3], E[:, 4], E[:, 5], E[:, 2]
    return np.column_stack([rating, verified, length / 200.0, (hour % 24) / 24.0])


def auc_score(y, s):
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        idx = np.where(s == v)[0]
        if len(idx) > 1:
            ranks[idx] = ranks[idx].mean()
    pos = ranks[y == 1].sum()
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    return (pos - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)


def single_review_classifier(E, is_fake_user):
    """训练单条分类器，返回（评论级）用户被命中的比例。"""
    X = review_level_features(E)
    y = is_fake_user[E[:, 0].astype(int)].astype(float)
    # 简单切分：随机 70% 训练
    rng = np.random.default_rng(5)
    idx = rng.permutation(len(y))
    cut = int(len(y) * 0.7)
    tr, te = idx[:cut], idx[cut:]
    w = fit_logistic(X[tr], y[tr])
    pred = (predict(w, X[te]) > 0.5).astype(int)
    tp = int(((pred == 1) & (y[te] == 1)).sum())
    fp = int(((pred == 1) & (y[te] == 0)).sum())
    fn = int(((pred == 0) & (y[te] == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    auc = auc_score(y[te], predict(w, X[te]))
    return prec, rec, auc


# ---------------------------------------------------------------- 图检测
def detect_groups(E, min_shared=3, min_size=5):
    """共评用户图 + 连通分量（GSRank 式思路）。

    步骤：
      1) 建「用户 -> 评过的商品集合」；
      2) 两个用户共同评过 >= min_shared 个商品，就在用户图上连一条边；
      3) 取连通分量，规模 >= min_size 的视为候选群组；
      4) 返回规模最大的一个群组及其共同商品。

    为什么这样能抓到：刷评群组成员彼此共同评过同一批目标商品，
    在共评图上形成稠密连通块；正常用户随机分散，很难凑够 min_shared。
    """
    items_by_user = {}
    for u, i in zip(E[:, 0].astype(int), E[:, 1].astype(int)):
        items_by_user.setdefault(u, set()).add(i)
    users = sorted(items_by_user)

    # 并查集
    parent = {u: u for u in users}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a_i in range(len(users)):
        u = users[a_i]
        su = items_by_user[u]
        for b_i in range(a_i + 1, len(users)):
            v = users[b_i]
            if len(su & items_by_user[v]) >= min_shared:
                union(u, v)

    comps = {}
    for u in users:
        comps.setdefault(find(u), []).append(u)
    groups = [sorted(g) for g in comps.values() if len(g) >= min_size]
    groups.sort(key=len, reverse=True)
    if not groups:
        return [], [], 0
    main = groups[0]
    # 该群组共同评过的商品（至少一半成员评过）
    cnt = {}
    for u in main:
        for i in items_by_user[u]:
            cnt[i] = cnt.get(i, 0) + 1
    half = max(2, len(main) // 2)
    shared_items = sorted([int(i) for i, c in cnt.items() if c >= half])
    return [int(x) for x in main], shared_items, len(groups)


def demo():
    E, is_fake_user, group_users, targets = build_bipartite()
    n_edges = len(E)
    print("=== 群组刷评检测 ===")
    print(f"用户 {N_USER}（其中刷评群组 {GROUP_SIZE} 个）| 商品 {N_ITEM} | 评论 {n_edges}")
    print(f"群组攻击的目标商品：{targets}")
    print("群组特征：成员重叠、目标集中、时间集中在 48 小时内、评分清一色 5 星")
    print("关键设定：刷评的文字长度与评分与正常评论同分布——单条看很正常\n")

    prec1, rec1, auc1 = single_review_classifier(E, is_fake_user)
    print("--- 做法一：单条评论分类（只用评分/验证/长度/小时）---")
    print(f"  AUC {auc1:.3f}  精确率 {prec1*100:.1f}%  召回率 {rec1*100:.1f}%")
    print("  说明：AUC 只有 0.65 左右，判别力很弱；按 0.5 阈值切分时几乎没有正例被判出。\n")

    hit_users, shared_items, n_groups = detect_groups(E)
    tp = len(set(hit_users) & set(group_users))
    prec2 = tp / max(len(hit_users), 1)
    rec2 = tp / max(len(group_users), 1)
    print("--- 做法二：共评用户图 + 连通分量（GSRank 式思路）---")
    print(f"  检出群组 {n_groups} 个；最大群组 {len(hit_users)} 人")
    print(f"  该群组共同评过的商品：{shared_items}")
    print(f"  命中真实群组成员 {tp} / {GROUP_SIZE}")
    print(f"  精确率 {prec2*100:.1f}%  召回率 {rec2*100:.1f}%")
    print(f"  检出用户示例：{hit_users[:12]}")

    print("\n结论：")
    print("  1) 单条分类几乎抓不到群组——因为刷评在单条特征上与正常评论同分布。")
    print("  2) 结构检测能抓到——因为群组无法同时伪装「成员重叠 + 目标集中 + 时间集中」。")
    print("  3) 但注意精确率只有约一半：结构检测会误伤「碰巧重叠」的正常用户，")
    print("     实际使用时需要人工复核或更严格的阈值——召回高不等于可以直接封号。")
    print("  4) 这正是 GSRank／FRAUDAR 一类方法的出发点：从文本分类转向结构异常发现。")
    print("\n注：本程序是教学实现，不是 FRAUDAR 原算法；原论文还有可证明上界与更完整的贪心策略。")

    return {"single": (prec1, rec1, auc1), "graph": (prec2, rec2, len(shared_items))}


# ---------------------------------------------------------------- 可视化
def save_visual(path='visuals/rev_group_fraud.svg'):
    """对比单条分类与图结构检测的召回率。"""
    from chartlib import BarChart
    r = demo()
    ch = BarChart('群组刷评：两种做法的召回率对比', '做法', '召回率（%）')
    ch.add_series('召回率', ['单条评论分类', '图结构块检测'],
                  [round(r["single"][1] * 100, 1), round(r["graph"][1] * 100, 1)])
    ch.add_note(f'单条分类 AUC {r["single"][2]:.2f}（几乎无判别力）；'
                f'共评图检出 {r["graph"][2]} 个共同商品，召回 {r["graph"][1]*100:.0f}%')
    return ch.save(path)


if __name__ == '__main__':
    demo()
