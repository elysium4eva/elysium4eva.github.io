#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mkt_mmm.py -- 营销组合模型（MMM）的最小实现

用途：
    用最小二乘把销量拆成「基线 + 各渠道贡献 + 季节」，并演示两个关键变换：
      1) Adstock：广告效果的时间衰减（今天投的钱，几周内还在起作用）；
      2) 饱和：边际收益递减（投得越多，每一块钱的效果越小）。
    最后给出「再加 10% 预算该加在哪」的建议。

输入：
    合成数据：104 周，两个渠道的周投放额，以及销量
输出：
    基线、各渠道系数、以及预算再分配建议
依赖：
    numpy（可选 chartlib 出 SVG 图）
对应课件：
    模块五第四章 §4 算法 4-4 MMM 营销组合模型
"""

import numpy as np

WEEKS = 104
SEED = 17


def adstock(x, decay=0.5):
    """Adstock 变换：x_t' = x_t + decay * x_{t-1}'。"""
    out = np.zeros_like(x, dtype=float)
    for t in range(len(x)):
        out[t] = x[t] + (decay * out[t - 1] if t > 0 else 0.0)
    return out


def hill(x, k=1.0, s=None):
    """Hill 饱和函数：x^s / (k^s + x^s)，把投入压到 0~1。"""
    s = 1.0 if s is None else s
    return (x ** s) / (k ** s + x ** s)


def ols(X, y):
    """最小二乘（含截距）。"""
    Xb = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return beta


def build_data(seed=SEED):
    rng = np.random.default_rng(seed)
    t = np.arange(WEEKS)
    # 两个渠道的周投放额（千元）
    s1 = 20 + 10 * np.sin(2 * np.pi * t / 26) + rng.normal(0, 2, WEEKS)
    s2 = 15 + 8 * np.sin(2 * np.pi * t / 13 + 1) + rng.normal(0, 2, WEEKS)
    s1 = np.clip(s1, 1, None)
    s2 = np.clip(s2, 1, None)
    # 真实生成过程：基线 120 + 渠道 1（强、易饱和）+ 渠道 2（弱、线性）+ 季节 + 噪声
    base = 120.0
    e1 = 18.0 * hill(adstock(s1 / 20.0, 0.6), k=1.2, s=1.5)
    e2 = 6.0 * hill(adstock(s2 / 20.0, 0.3), k=2.0, s=1.0)
    season = 12 * np.sin(2 * np.pi * t / 52) + 5 * np.sin(2 * np.pi * t / 13)
    y = base + e1 * 6 + e2 * 6 + season + rng.normal(0, 4, WEEKS)
    return s1, s2, y


def demo():
    s1, s2, y = build_data()
    print('=== 营销组合模型（MMM）最小实现 ===')
    print('数据：%d 周，两个渠道的周投放额与销量\n' % WEEKS)

    # 变换
    a1 = adstock(s1 / 20.0, decay=0.6)
    a2 = adstock(s2 / 20.0, decay=0.3)
    f1 = hill(a1, k=1.2, s=1.5)
    f2 = hill(a2, k=2.0, s=1.0)

    # 季节项（用两个正弦）
    t = np.arange(WEEKS)
    sin52 = np.sin(2 * np.pi * t / 52)
    cos52 = np.cos(2 * np.pi * t / 52)
    sin13 = np.sin(2 * np.pi * t / 13)

    X = np.column_stack([f1, f2, sin52, cos52, sin13])
    beta = ols(X, y)
    base_hat = beta[0]
    b1, b2 = beta[1], beta[2]
    y_hat = np.column_stack([np.ones(WEEKS), X]) @ beta
    r2 = 1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2)

    print('--- 拟合结果 ---')
    print('  基线（不投广告也有的销量）: %.1f' % base_hat)
    print('  渠道 1 系数（饱和后）    : %.1f' % b1)
    print('  渠道 2 系数（饱和后）    : %.1f' % b2)
    print('  拟合优度 R^2             : %.3f' % r2)
    print()

    contrib1 = b1 * f1
    contrib2 = b2 * f2
    print('--- 平均贡献占比 ---')
    tot = contrib1.mean() + contrib2.mean()
    print('  渠道 1 平均贡献: %.1f（占广告贡献的 %.0f%%）' % (
        contrib1.mean(), contrib1.mean() / tot * 100))
    print('  渠道 2 平均贡献: %.1f（占广告贡献的 %.0f%%）' % (
        contrib2.mean(), contrib2.mean() / tot * 100))
    print()

    print('--- 边际收益：加预算要看边际，不看平均 ---')
    print('  定义：每多投 1 个单位（千元）带来的销量增量\n')
    print('  %-8s %10s %12s %12s %12s' % ('渠道', '中位投放', '当前边际', '+50% 后边际', '边际衰减'))
    print('  ' + '-' * 58)
    decay = {'渠道 1': 0.6, '渠道 2': 0.3}
    marg = {}
    for name, s_arr, k, ss, b in [('渠道 1', s1, 1.2, 1.5, b1), ('渠道 2', s2, 2.0, 1.0, b2)]:
        cur = float(np.median(s_arr))

        def unit_marginal(x):
            lo = hill(adstock(np.array([x / 20.0]), decay[name])[0], k=k, s=ss)
            hi = hill(adstock(np.array([(x + 1.0) / 20.0]), decay[name])[0], k=k, s=ss)
            return float(b * (hi - lo))

        m_now = unit_marginal(cur)
        m_up = unit_marginal(cur * 1.5)
        marg[name] = (m_now, m_up)
        print('  %-8s %10.1f %12.3f %12.3f %11.0f%%' % (
            name, cur, m_now, m_up,
            (1 - m_up / m_now) * 100 if m_now else 0))
    print()

    better = '渠道 1' if marg['渠道 1'][0] > marg['渠道 2'][0] else '渠道 2'
    steeper = '渠道 1' if (marg['渠道 1'][0] and (1 - marg['渠道 1'][1] / marg['渠道 1'][0])
                           > (1 - marg['渠道 2'][1] / marg['渠道 2'][0])) else '渠道 2'
    print('结论：')
    print('  1) 基线约 %.0f——不建模它，就会把所有销量都算成广告功劳。' % base_hat)
    print('  2) 渠道 1 的平均贡献远高于渠道 2（%.0f%% vs %.0f%%），' % (
        contrib1.mean() / tot * 100, contrib2.mean() / tot * 100))
    print('     但看边际：当前每多投 1 单位，渠道 1 带来 %.3f、渠道 2 带来 %.3f。' % (
        marg['渠道 1'][0], marg['渠道 2'][0]))
    print('  3) 按当前边际，加预算应优先给【%s】；但它的边际衰减最快（%s），' % (better, steeper))
    print('     所以「加多少」比「加给谁」更重要——饱和之后再多投就是浪费。')
    print('  4) Adstock 刻画延迟：本周投放的效果会延续几周，只看当周会低估总效果。')
    print('  5) MMM 与归因分工：MMM 管预算分配（宏观长期），归因管渠道评估（微观短期）。')
    return {'base': base_hat, 'b1': b1, 'b2': b2, 'r2': r2}


def save_visual(path='visuals/mkt_mmm.svg'):
    """饱和曲线：投入 vs 边际效果。"""
    from chartlib import LineChart
    ch = LineChart('饱和曲线：为什么加预算的边际效果会递减', '投入水平（相对基准）', '边际销量贡献')
    xs = [round(float(x), 1) for x in np.linspace(0.2, 4.0, 9)]
    for name, k, s_ in [('渠道 1（易饱和）', 1.2, 1.5), ('渠道 2（较线性）', 2.0, 1.0)]:
        ys = [round(float(hill(np.array([x]), k=k, s=s_)[0]) * 10, 2) for x in xs]
        ch.add_line(name, xs, ys)
    ch.add_note('渠道 1 早期更强但很快饱和；渠道 2 平缓但在高位反超——预算该看边际')
    return ch.save(path)


if __name__ == '__main__':
    demo()
