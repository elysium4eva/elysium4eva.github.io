#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sel_scorecard.py —— 选品多准则决策：加权评分卡 + TOPSIS

对应课件：模块五 第二章 §4「智能选品的典型算法」— 多准则决策族

用途：
    给定若干候选商品在多个维度上的打分，做两件事：
      A) 加权评分：按六维权重加权求和，得到综合得分与排序
      B) TOPSIS：按「与理想解的相对接近度」排序，避免权重主观性带来的偏差
    并做一次权重敏感性分析，看排序在最坏情况下会不会翻转。

为什么两个方法一起用：
    加权评分直观、易解释，但权重稍改排序就可能变；TOPSIS 用「到最优/最劣
    方案的距离」排序，对量纲不敏感、结论更稳健。两者结论一致时可信度高，
    不一致时说明候选之间差距很小，不该凭打分硬选。

依赖：numpy、pandas（本课程环境自带）
运行：python sel_scorecard.py

数据说明：
    下面的候选与打分全部是【教学用示例数据】，不对应任何真实商品。
    换成自己的数据时，只需替换 CANDIDATES 与 CRITERIA 两个结构。
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 示例数据
# 六个维度（对应课件的六维评分卡），direction=+1 表示越大越好，-1 表示越小越好
CRITERIA = [
    {"key": "需求规模", "weight": 0.20, "direction": +1, "desc": "市场容量能否支撑目标"},
    {"key": "竞争强度", "weight": 0.25, "direction": +1, "desc": "越大越好：集中度低、门槛低"},
    {"key": "毛利空间", "weight": 0.20, "direction": +1, "desc": "净毛利率"},
    {"key": "差异化空间", "weight": 0.15, "direction": +1, "desc": "可改进的功能点多少"},
    {"key": "物流合规", "weight": 0.10, "direction": +1, "desc": "越大越好：体积小、认证易"},
    {"key": "供应链可控", "weight": 0.10, "direction": +1, "desc": "交期稳定、可小批量"},
]

# 5 分制打分
CANDIDATES = {
    "A 防水蓝牙音箱":   [4, 4, 4, 3, 3, 4],
    "B 车载手机支架":   [5, 2, 3, 3, 4, 3],
    "C 户外电源":       [3, 2, 5, 4, 2, 2],
    "D 便携风扇":       [4, 3, 2, 2, 4, 4],
    "E 无线充电板":     [4, 3, 4, 2, 4, 3],
}


def build_frame():
    names = list(CANDIDATES.keys())
    crit_keys = [c["key"] for c in CRITERIA]
    df = pd.DataFrame(CANDIDATES, index=crit_keys).T
    df.index.name = "候选"
    return df, names, crit_keys


# ---------------------------------------------------------------- A) 加权评分
def weighted_score(df, crit_keys):
    """加权求和。把权重向量与打分矩阵做矩阵乘法。"""
    w = np.array([c["weight"] for c in CRITERIA], dtype=float)
    assert abs(w.sum() - 1.0) < 1e-9, "权重之和必须为 1"
    scores = df[crit_keys].to_numpy(dtype=float) @ w
    return pd.Series(scores, index=df.index, name="加权得分")


# ---------------------------------------------------------------- B) TOPSIS
def topsis(df, crit_keys):
    """TOPSIS：向量归一化 → 加权 → 找理想解/负理想解 → 算相对接近度。

    步骤（与课件一致，四步）：
      1. 归一化：x_ij / sqrt(sum_i x_ij^2)，消去量纲
      2. 加权：乘权重
      3. 定理想解 V+ / 负理想解 V-
      4. 算距离 D+、D-，接近度 C = D- / (D+ + D-)，C 越大越好
    """
    X = df[crit_keys].to_numpy(dtype=float)
    w = np.array([c["weight"] for c in CRITERIA], dtype=float)
    direction = np.array([c["direction"] for c in CRITERIA], dtype=float)

    # 1) 向量归一化
    norm = np.sqrt((X ** 2).sum(axis=0))
    norm[norm == 0] = 1.0
    R = X / norm

    # 2) 加权
    V = R * w

    # 3) 理想解 / 负理想解（按方向取 max 或 min）
    ideal = np.where(direction > 0, V.max(axis=0), V.min(axis=0))
    anti = np.where(direction > 0, V.min(axis=0), V.max(axis=0))

    # 4) 距离与相对接近度
    d_pos = np.sqrt(((V - ideal) ** 2).sum(axis=1))
    d_neg = np.sqrt(((V - anti) ** 2).sum(axis=1))
    closeness = d_neg / (d_pos + d_neg)
    return pd.Series(closeness, index=df.index, name="TOPSIS 接近度")


# ---------------------------------------------------------------- C) 敏感性
def sensitivity(df, crit_keys, base_rank, trials=2000, jitter=0.5, seed=7):
    """随机扰动权重 ±jitter（相对值），看排序首位被翻转的概率。

    jitter=0.5 表示每个权重乘以 [0.5, 1.5] 之间的随机系数后再归一化。
    结论解读：翻转概率高说明候选之间差距小，不应只凭打分下结论。
    """
    rng = np.random.default_rng(seed)
    base_w = np.array([c["weight"] for c in CRITERIA], dtype=float)
    X = df[crit_keys].to_numpy(dtype=float)
    flip = 0
    for _ in range(trials):
        w = base_w * rng.uniform(1 - jitter, 1 + jitter, size=base_w.shape)
        w = w / w.sum()
        s = X @ w
        if int(np.argmax(s)) != base_rank:
            flip += 1
    return flip / trials


def main():
    pd.set_option("display.width", 120)
    pd.set_option("display.unicode.east_asian_width", True)

    df, names, crit_keys = build_frame()
    print("=" * 68)
    print("候选打分矩阵（5 分制）")
    print("=" * 68)
    print(df.to_string())

    print("\n权重：" + "，".join(f'{c["key"]} {c["weight"]:.2f}' for c in CRITERIA))

    ws = weighted_score(df, crit_keys)
    ts = topsis(df, crit_keys)

    result = pd.DataFrame({"加权得分": ws.round(3), "TOPSIS 接近度": ts.round(3)})
    result["加权排名"] = ws.rank(ascending=False, method="min").astype(int)
    result["TOPSIS 排名"] = ts.rank(ascending=False, method="min").astype(int)
    result = result.sort_values("加权排名")

    print("\n" + "=" * 68)
    print("两种方法的排序对比")
    print("=" * 68)
    print(result.to_string())

    # 一致性检查
    same = (result["加权排名"].to_numpy() == result["TOPSIS 排名"].to_numpy()).all()
    print(f"\n两种方法排序是否完全一致：{'是' if same else '否'}")
    if not same:
        print("→ 不一致说明候选间差距较小，建议补充数据或做小规模试销验证，而不是凭打分硬选。")

    # 敏感性分析
    top = result.index[0]
    base_rank = list(df.index).index(top)
    p = sensitivity(df, crit_keys, base_rank)
    print(f"\n敏感性分析（权重扰动 ±50%，2000 次）：首位『{top}』被翻转的概率 {p:.1%}")
    if p > 0.20:
        print("→ 翻转概率偏高：说明第一名优势不稳固，建议优先验证排名 1–2 的候选。")
    else:
        print("→ 首位较稳固：可以据此进入下一步（小批量试单）。")

    print("\n提示：打分是主观输入，方法只保证「在你给的分下算得对」。")
    print("      真正的验证仍是小批量试单（见课件 §2「小步验证」）。")


if __name__ == "__main__":
    main()
