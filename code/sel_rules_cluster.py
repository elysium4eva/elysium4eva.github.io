#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sel_rules_cluster.py —— 选品两件工具：关联规则（找搭配）+ K-means 聚类（找价格空档）

对应课件：模块五 第二章 §4「智能选品的典型算法」
          — 关联规则族（Apriori / FP-Growth）
          — 聚类族（K-means / DBSCAN）

用途：
    A) 关联规则：从订单/购物车记录里找「经常一起买」的商品对，
       算 支持度(support)、置信度(confidence)、提升度(lift)，用于套装设计与搭配选品。
    B) K-means 聚类：把竞品按「价格—评分」聚成几群，找出哪一群稀疏（价格空档）。

为什么用 lift 而不是只看 support：
    support 高的组合往往只是「两个都常见」（比如手机壳和充电线），不代表有搭配关系。
    lift > 1 才说明「买了 A 之后买 B 的概率上升」，这才是真正可用的搭配信号。

依赖：numpy、pandas（本课程环境自带）
运行：python sel_rules_cluster.py

数据说明：全部为教学用示例数据，不对应真实店铺。
"""

import itertools
from collections import Counter

import numpy as np
import pandas as pd


# ================================================================ A) 关联规则
def apriori_pairs(transactions, min_support=0.05):
    """只做「二阶频繁项集」：够用、好懂、跑得快。

    真实场景里 FP-Growth 更快，但二阶结论与 Apriori 等价，
    教学上先用最朴素的计数把道理讲清楚。
    """
    n = len(transactions)
    pair_counter = Counter()
    item_counter = Counter()

    for basket in transactions:
        items = sorted(set(basket))
        for it in items:
            item_counter[it] += 1
        for a, b in itertools.combinations(items, 2):
            pair_counter[(a, b)] += 1

    rows = []
    for (a, b), cnt in pair_counter.items():
        sup_ab = cnt / n
        if sup_ab < min_support:
            continue
        sup_a = item_counter[a] / n
        sup_b = item_counter[b] / n
        # A -> B
        conf_ab = sup_ab / sup_a
        lift_ab = conf_ab / sup_b
        # B -> A
        conf_ba = sup_ab / sup_b
        lift_ba = conf_ba / sup_a
        rows.append(
            {
                "商品 A": a,
                "商品 B": b,
                "共现次数": cnt,
                "支持度": round(sup_ab, 4),
                "置信度 A→B": round(conf_ab, 4),
                "提升度 A→B": round(lift_ab, 3),
                "置信度 B→A": round(conf_ba, 4),
                "提升度 B→A": round(lift_ba, 3),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["提升度 A→B", "支持度"], ascending=False)
    return df


def demo_rules():
    print("=" * 74)
    print("A) 关联规则：哪些商品经常一起被买？")
    print("=" * 74)

    # 40 笔购物篮（教学示例）
    transactions = [
        ["音箱", "挂绳"], ["音箱", "挂绳"], ["音箱", "收纳包"], ["音箱", "挂绳"],
        ["音箱", "充电线"], ["音箱", "挂绳"], ["收纳包", "挂绳"], ["音箱", "收纳包"],
        ["音箱", "挂绳"], ["充电线", "充电头"], ["音箱", "充电线"], ["音箱", "挂绳"],
        ["收纳包", "充电线"], ["音箱", "收纳包"], ["音箱", "挂绳"], ["音箱", "挂绳"],
        ["充电线", "充电头"], ["音箱", "挂绳"], ["收纳包", "挂绳"], ["音箱", "收纳包"],
        ["音箱", "挂绳"], ["音箱", "充电线"], ["音箱", "挂绳"], ["收纳包", "收纳包"],
        ["音箱", "挂绳"], ["音箱", "收纳包"], ["充电线", "充电头"], ["音箱", "挂绳"],
        ["音箱", "挂绳"], ["收纳包", "挂绳"], ["音箱", "充电线"], ["音箱", "挂绳"],
        ["音箱", "收纳包"], ["音箱", "挂绳"], ["充电线", "充电头"], ["音箱", "挂绳"],
        ["音箱", "收纳包"], ["音箱", "挂绳"], ["音箱", "挂绳"], ["收纳包", "挂绳"],
    ]

    df = apriori_pairs(transactions, min_support=0.05)
    print(df.head(12).to_string(index=False))

    print("\n怎么读：")
    print("  · 支持度 = 同时出现这俩的订单占比（有没有量）")
    print("  · 置信度 = 买了 A 的人里有多少也买 B（强不强）")
    print("  · 提升度 > 1 = 买 A 会提高买 B 的概率（是不是真搭配）；≈ 1 说明互不相干")
    strong = df[(df["提升度 A→B"] > 1.2) & (df["支持度"] > 0.05)]
    if not strong.empty:
        print("\n可考虑做套装/搭配推荐的组合（提升度 > 1.2）：")
        for _, r in strong.head(5).iterrows():
            print(f"  · {r['商品 A']} + {r['商品 B']}（提升度 {r['提升度 A→B']:.2f}，"
                  f"支持度 {r['支持度']:.1%}）")


# ================================================================ B) 聚类
def cluster_demo_data():
    """30 个竞品的（价格 美元, 评分）——三段各 10 个，用于演示分群。"""
    rng = np.random.default_rng(3)
    low = np.column_stack([rng.normal(12, 1.8, 10), rng.normal(4.1, 0.18, 10)])
    mid = np.column_stack([rng.normal(22, 2.2, 10), rng.normal(4.4, 0.15, 10)])
    high = np.column_stack([rng.normal(42, 3.0, 10), rng.normal(4.5, 0.12, 10)])
    return np.vstack([low, mid, high])


def kmeans(X, k, iters=100, seed=42):
    """手写 K-means：初始化 → 分配 → 更新质心 → 迭代到收敛。

    用 k-means++ 的思路做初始化（随机选第一个，之后按距离平方概率选），
    避免全部初始点挤在一起。返回（标签, 质心, 惯性）。
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    # k-means++ 初始化
    centers = [X[rng.integers(n)]]
    for _ in range(k - 1):
        d2 = np.min(((X[:, None, :] - np.array(centers)[None, :, :]) ** 2).sum(-1), axis=1)
        probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
        centers.append(X[rng.choice(n, p=probs)])
    C = np.array(centers, dtype=float)

    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        new_labels = d.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            if (labels == j).any():
                C[j] = X[labels == j].mean(axis=0)

    inertia = float(((X - C[labels]) ** 2).sum())
    return labels, C, inertia


def choose_k(X, k_range=(2, 6), seed=42):
    """用肘部法（惯性下降幅度）粗选 k，不画图，直接打印。"""
    rows = []
    for k in range(k_min := k_range[0], k_range[1] + 1):
        _, _, inertia = kmeans(X, k, seed=seed)
        rows.append({"k": k, "惯性 SSE": round(inertia, 2)})
    df = pd.DataFrame(rows)
    df["下降幅度"] = (df["惯性 SSE"].shift(1) - df["惯性 SSE"]).round(2)
    return df


def demo_cluster():
    print("\n" + "=" * 74)
    print("B) K-means 聚类：竞品价格带上有没有空档？")
    print("=" * 74)

    X = cluster_demo_data()
    df = pd.DataFrame(X, columns=["价格", "评分"])
    df["价格"] = df["价格"].round(2)
    df["评分"] = df["评分"].round(2)

    print("\n聚类数 k 的选择（肘部法）：")
    print(choose_k(df[["价格", "评分"]].to_numpy(), (2, 5)).to_string(index=False))

    labels, C, inertia = kmeans(df[["价格", "评分"]].to_numpy(), 3)
    df["簇"] = labels
    order = np.argsort(C[:, 0])
    remap = {int(old): i for i, old in enumerate(order)}
    df["簇"] = df["簇"].map(remap)

    summary = df.groupby("簇").agg(
        数量=("价格", "size"),
        均价=("价格", "mean"),
        最低价=("价格", "min"),
        最高价=("价格", "max"),
        均评分=("评分", "mean"),
    ).round(2)
    print("\n各簇画像：")
    print(summary.to_string())

    # 价格空档：相邻簇之间的间隔
    print("\n价格空档分析（相邻簇的价格间隔）：")
    ordered = summary.sort_values("均价")
    prices = ordered["均价"].to_numpy()
    gaps = []
    for i in range(len(prices) - 1):
        gaps.append((ordered.index[i], ordered.index[i + 1], prices[i + 1] - prices[i]))
    for a, b, g in gaps:
        flag = "← 间隔大，存在供给稀薄区间" if g > 8 else ""
        print(f"  簇{a} → 簇{b}：均价差 {g:.2f} 美元 {flag}")

    biggest = max(gaps, key=lambda t: t[2])
    print(f"\n建议切入：在簇{biggest[0]}与簇{biggest[1]}之间（约 "
          f"{prices[list(ordered.index).index(biggest[0])]:.0f}–"
          f"{prices[list(ordered.index).index(biggest[1])]:.0f} 美元区间）验证是否有需求缺口。")
    print("注意：簇间隔大只说明「供给少」，不等于「有需求」——必须再查该价位是否有销量。")



# ---------------------------------------------------------------- 可视化
def save_visual(path='visuals/sel_rules_cluster.svg'):
    """输出竞品聚类散点图（价格-评分平面，含质心与簇间隔提示）。"""
    from chartlib import ScatterChart
    X = cluster_demo_data()
    labels, C, _ = kmeans(X, 3)
    order = np.argsort(C[:, 0])
    remap = {int(o): i for i, o in enumerate(order)}
    ch = ScatterChart('竞品聚类：价格-评分平面', '价格（美元）', '评分')
    names = ['低价带', '中价带', '高价带']
    marks = ['circle', 'square', 'triangle']
    for k in range(3):
        idx = [i for i, l in enumerate(labels) if remap[int(l)] == k]
        ch.add_group(names[k],
                     [float(X[i, 0]) for i in idx], [float(X[i, 1]) for i in idx],
                     marker=marks[k])
    ch.add_centroids([float(C[o, 0]) for o in order], [float(C[o, 1]) for o in order])
    ch.add_note('簇间空档只说明供给少，不等于有需求')
    return ch.save(path)

if __name__ == "__main__":
    demo_rules()
    demo_cluster()
