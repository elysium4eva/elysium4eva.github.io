#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
soc_uplift.py -- Uplift 建模：该给谁发券

用途：
    用 T-learner（分别对处理组与对照组建模）估计每个人的增量效果 tau(x)，
    并把人群分成四类（说服型 / 自然转化 / 无动于衷 / 反效果），
    说明为什么「整体转化率」会误导投放决策。

输入：
    合成实验数据（用户特征 + 是否发券 T + 是否购买 Y）
输出：
    四类人群的人数与增量、整体转化率对比、按 uplift 排序的投放建议
依赖：
    numpy（可选 chartlib 出 SVG 图）
对应课件：
    模块五第三章 §4 算法 4-3 Uplift 建模

说明：
    真实业务需要随机对照实验（AB 测试）数据；没有真实对照，uplift 无法识别。
"""

import numpy as np

N = 3000
SEED = 11


def make_data(n=N, seed=SEED):
    """合成数据：特征 x1（活跃度）、x2（价格敏感度），异质性处理效应。"""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)          # 活跃度
    x2 = rng.normal(0, 1, n)          # 价格敏感度（越高越敏感）
    T = rng.integers(0, 2, n)          # 随机发券

    # 基准购买倾向：活跃度高更可能买，价格敏感度高更不买
    base = -1.2 + 0.9 * x1 - 0.5 * x2
    # 异质性处理效应：只有「活跃且价格敏感」的人因券而买
    tau = 1.4 * np.tanh(0.9 * x1 - 0.7 * x2 + 0.3)
    logit = base + T * tau
    p = 1 / (1 + np.exp(-logit))
    Y = (rng.random(n) < p).astype(int)
    return x1, x2, T, Y, tau


def fit_logistic(X, y, lr=0.1, epochs=400):
    """简易逻辑回归（梯度下降），返回权重。"""
    Xb = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-Xb @ w))
        grad = Xb.T @ (p - y) / len(y)
        w -= lr * grad
    return w


def predict(w, X):
    Xb = np.column_stack([np.ones(len(X)), X])
    return 1 / (1 + np.exp(-Xb @ w))


def t_learner(X, T, Y):
    """T-learner：对处理组与对照组分别训练，返回 tau(x) 的函数。"""
    w1 = fit_logistic(X[T == 1], Y[T == 1])
    w0 = fit_logistic(X[T == 0], Y[T == 0])
    return lambda Xn: predict(w1, Xn) - predict(w0, Xn)


def demo():
    x1, x2, T, Y, tau_true = make_data()
    X = np.column_stack([x1, x2])

    print("=== Uplift 建模（T-learner）===")
    print(f"样本 {len(Y)} | 处理组 {int(T.sum())} | 对照组 {int((1-T).sum())}")

    # 整体转化率
    r1 = Y[T == 1].mean()
    r0 = Y[T == 0].mean()
    print(f"\n处理组转化率 {r1*100:.1f}% | 对照组转化率 {r0*100:.1f}% | 差值 {(r1-r0)*100:.1f} 个百分点")

    # 估计 uplift
    tau_fn = t_learner(X, T, Y)
    tau_hat = tau_fn(X)

    # 四类人群：用 uplift 符号 + 基准购买倾向划分
    thr = 0.02
    w_all = fit_logistic(X, Y)                 # 全样本模型（含 T）用于估基准倾向
    base_p = predict(w_all, X)
    neutral = np.abs(tau_hat) <= thr
    cut = float(np.median(base_p[neutral])) if neutral.any() else 0.5
    groups = {
        '说服型（该投）': tau_hat > thr,
        '反效果型（发了更差）': tau_hat < -thr,
        '自然转化型（发不发都买）': neutral & (base_p > cut),
        '无动于衷型（发不发都不买）': neutral & (base_p <= cut),
    }
    print("\n=== 四类人群 ===")
    for name, mask in groups.items():
        n_g = int(mask.sum())
        avg = float(tau_hat[mask].mean()) if n_g else 0.0
        print(f"  {name:<24} {n_g:5d} 人  平均 uplift {avg:+.3f}")

    # 只比较「增量」，不要拿总转化去比
    n = len(Y)
    top = np.argsort(-tau_hat)[: int(n * 0.2)]
    inc_all = float(tau_hat.sum())
    inc_top = float(tau_hat[top].sum())
    print("\n=== 投放策略对比（只比增量转化，假设每张券成本 1 元）===")
    print(f"  全量发券：{n:5d} 张，增量转化 {inc_all:7.1f}，每增量转化成本 {n/max(inc_all,1e-9):5.2f} 元")
    print(f"  只投 top20%：{len(top):5d} 张，增量转化 {inc_top:7.1f}，"
          f"每增量转化成本 {len(top)/max(inc_top,1e-9):5.2f} 元")
    if inc_top > 0:
        print(f"  效率提升：{(inc_all/max(inc_top,1e-9)-1)*100:.0f}% 的预算被省下来")

    print("\n结论：整体转化率告诉我们「发券有没有用」，")
    print("      而 uplift 告诉我们「该发给谁」——后者才决定预算效率。")

    return {"tau_hat": tau_hat, "r1": r1, "r0": r0, "groups": groups}


# ---------------------------------------------------------------- 可视化
def save_visual(path='visuals/soc_uplift.svg'):
    """输出四类人群的人数与平均 uplift。"""
    from chartlib import BarChart
    r = demo()
    names, vals = [], []
    for name, mask in r["groups"].items():
        short = name.split('（')[0]
        names.append(short)
        vals.append(round(float(r["tau_hat"][mask].mean()) if mask.any() else 0.0, 3))
    ch = BarChart('四类人群的平均 uplift', '人群', '平均 uplift')
    ch.add_series('平均 uplift', names, vals)
    ch.add_note('只有「说服型」值得发券；「反效果型」发券反而降低转化')
    return ch.save(path)


if __name__ == '__main__':
    demo()
