#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rev_feature_layers.py -- 三层特征体系的增量价值

用途：
    合成一批真实／虚假评论，构造语言层、行为层、图结构层三组特征，
    分别训练「仅语言」「+行为」「+图结构」三个模型，比较 AUC 与召回率，
    用来验证课件里的一句话：越往后越难伪造，也越有效。

输入：
    合成评论数据（真实与虚假评论的生成规则不同，但都刻意做得「像真的」）
输出：
    三层模型各自的 AUC 与召回率、各组特征的单独判别力
依赖：
    numpy（可选 chartlib 出 SVG 图）
对应课件：
    模块五第二章 §7 特征工程三层体系

说明：
    虚假评论的构造方式是：语言层与行为层刻意与真实评论同分布，
    只在「图结构」上留下痕迹（同一批商品上评论者高度重叠、时间聚集）。
    这样才能说明为什么单靠语言层不够。
"""

import numpy as np

N_REAL = 700
N_FAKE = 300
N_PRODUCT = 40
SEED = 23


# ---------------------------------------------------------------- 数据构造
def build_reviews(seed=SEED):
    """返回特征字典与标签。"""
    rng = np.random.default_rng(seed)
    n = N_REAL + N_FAKE
    is_fake = np.r_[np.zeros(N_REAL, dtype=int), np.ones(N_FAKE, dtype=int)]

    # ---------- 语言层：刻意让真假同分布 ----------
    length = rng.normal(140, 45, n).clip(30, 320)
    uniq_ratio = rng.normal(0.62, 0.08, n).clip(0.35, 0.9)
    exclaim = rng.poisson(1.1, n)
    first_person = rng.poisson(1.8, n)

    # ---------- 行为层：也基本同分布 ----------
    # 行为层刻意做得接近：虚假评论只是略偏高，且带有重叠分布
    rating = np.where(is_fake == 1,
                      rng.normal(4.55, 0.45, n),
                      rng.normal(4.35, 0.60, n)).clip(1, 5)
    verified = (rng.random(n) < 0.80).astype(int)
    hour = rng.integers(0, 24, n)

    # ---------- 图结构层：虚假评论在这里留痕 ----------
    # 虚假评论集中攻击前 8 个商品；真实评论分散在 40 个商品上
    hot = 8
    product = np.where(is_fake == 1,
                       rng.integers(0, hot, n),
                       rng.integers(0, N_PRODUCT, n))
    # 共现强度：虚假评论者之间高度重叠（用「同商品同小时内的评论数」近似）
    cooc = np.where(is_fake == 1,
                    rng.normal(8.6, 2.6, n),
                    rng.normal(3.6, 2.4, n)).clip(0, 20)
    # 时间聚集度：虚假评论集中在极短时间内
    burst = np.where(is_fake == 1,
                     rng.normal(0.78, 0.14, n),
                     rng.normal(0.36, 0.19, n)).clip(0, 1)

    X = {
        '语言层': np.column_stack([length / 300.0, uniq_ratio, exclaim / 3.0, first_person / 4.0]),
        '行为层': np.column_stack([rating / 5.0, verified, hour / 24.0]),
        '图结构层': np.column_stack([product / N_PRODUCT, cooc / 20.0, burst]),
    }
    names = {
        '语言层': ['长度', '词汇多样性', '感叹号数', '第一人称数'],
        '行为层': ['评分', '是否验证', '发布小时'],
        '图结构层': ['所属商品热度位', '评论者共现强度', '时间聚集度'],
    }
    return X, names, is_fake


# ---------------------------------------------------------------- 模型
def fit_logistic(X, y, lr=0.15, epochs=1500):
    Xb = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        w -= lr * Xb.T @ (p - y) / len(y)
    return w


def predict(w, X):
    Xb = np.column_stack([np.ones(len(X)), X])
    return 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))


def auc_score(y, s):
    """AUC 的秩和算法（处理并列时取平均秩）。"""
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # 处理并列：按分数分组取平均秩
    uniq = np.unique(s)
    for v in uniq:
        idx = np.where(s == v)[0]
        if len(idx) > 1:
            ranks[idx] = ranks[idx].mean()
    pos = ranks[y == 1].sum()
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    return (pos - n1 * (n1 + 1) / 2) / (n1 * n0)


def evaluate(Xs, y, seed=7):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(len(y) * 0.7)
    tr, te = idx[:cut], idx[cut:]
    X = np.hstack(Xs)
    w = fit_logistic(X[tr], y[tr])
    s = predict(w, X[te])
    a = auc_score(y[te], s)
    # 按分数取前 30%（与真实虚假比例一致）计算召回与精确
    k = max(1, int(len(s) * 0.30))
    top = set(np.argsort(-s)[:k])
    yt = y[te]
    tp = int(sum(1 for i in top if yt[i] == 1))
    fp = int(sum(1 for i in top if yt[i] == 0))
    fn = int(int((yt == 1).sum()) - tp)
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    return a, rec, prec


def demo():
    X, names, y = build_reviews()
    print("=== 三层特征体系的增量价值 ===")
    print(f"样本 {len(y)}（真实 {N_REAL} / 虚假 {N_FAKE}）| 商品 {N_PRODUCT}")
    print("构造要点：虚假评论的语言层与行为层刻意与真实评论同分布，只在图结构上留痕\n")

    print("--- 各组特征的单独判别力（AUC）---")
    for layer in ['语言层', '行为层', '图结构层']:
        a, _, _ = evaluate([X[layer]], y)
        print(f"  仅 {layer}: AUC {a:.3f}")

    print("\n--- 三层累加 ---")
    combos = [
        ('仅语言层', ['语言层']),
        ('+ 行为层', ['语言层', '行为层']),
        ('+ 图结构层', ['语言层', '行为层', '图结构层']),
    ]
    res = []
    for label, layers in combos:
        a, rec, prec = evaluate([X[k] for k in layers], y)
        res.append((label, a, rec, prec))
        print(f"  {label:<10} AUC {a:.3f}  前30%召回 {rec*100:5.1f}%  精确率 {prec*100:5.1f}%")

    print("\n--- 各层特征的名称 ---")
    for layer in ['语言层', '行为层', '图结构层']:
        print(f"  {layer}: {', '.join(names[layer])}")

    print("\n结论：")
    print("  1) 语言层单独看几乎没有判别力——因为假评论被刻意写得「像真的」。")
    print("  2) 加入行为层有提升，但有限——刷单者也能模仿正常行为。")
    print("  3) 加入图结构层后提升最大——协同信号是最难伪造的一层。")
    print("  4) 这与课件结论一致：三层不是替代关系，而是叠加；越往后越贵也越有效。")
    return res


# ---------------------------------------------------------------- 可视化
def save_visual(path='visuals/rev_feature_layers.svg'):
    """三层累加的 AUC 与召回率。"""
    from chartlib import BarChart
    res = demo()
    labels = [r[0] for r in res]
    auc = [round(r[1], 3) for r in res]
    rec = [round(r[2] * 100, 1) for r in res]
    ch = BarChart('三层特征累加：AUC 与召回率', '特征组合', '指标值')
    ch.add_series('AUC（0-1）', labels, auc)
    ch.add_series('前30%召回率（%）', labels, rec)
    ch.add_note('语言层与行为层刻意同分布，图结构层才留痕——所以第三层提升最大')
    return ch.save(path)


if __name__ == '__main__':
    demo()
