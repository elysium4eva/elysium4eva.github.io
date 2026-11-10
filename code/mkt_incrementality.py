#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mkt_incrementality.py -- 增量实验：样本量、置信区间与「看起来有提升」

用途：
    演示三件事：
      1) 样本量公式：想把可检测差异缩一半，样本量要翻四倍；
      2) Geo 实验模拟：「看起来有提升」在多大概率上只是噪声；
      3) 小样本跑三天得出的结论，为什么不能信。

输入：
    模拟数据（基线转化率、真实提升幅度、样本量）
输出：
    所需样本量、不同样本量下的检出率（功效）、以及一次具体实验的置信区间
依赖：
    numpy（可选 chartlib 出 SVG 图）
对应课件：
    模块五第四章 §5 营销实验与增量测量
"""

import numpy as np

BASE_RATE = 0.03      # 基线转化率 3%
SEED = 31

# 标准正态分位数（避免依赖 scipy）
Z_ALPHA = 1.959964    # 双侧 95%
Z_BETA = 0.841621     # 功效 80%


def sample_size(p1, p2, z_a=Z_ALPHA, z_b=Z_BETA):
    """比例指标的两组样本量（每组）。"""
    return int(np.ceil((z_a + z_b) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2))


def simulate(p1, p2, n, trials=400, seed=SEED):
    """模拟 trials 次实验，返回「检出显著」的比例（即经验功效）。"""
    rng = np.random.default_rng(seed)
    hit = 0
    for _ in range(trials):
        x1 = rng.binomial(n, p1)
        x2 = rng.binomial(n, p2)
        r1, r2 = x1 / n, x2 / n
        se = np.sqrt(r1 * (1 - r1) / n + r2 * (1 - r2) / n)
        if se > 0 and abs(r2 - r1) / se > Z_ALPHA:
            hit += 1
    return hit / trials


def one_experiment(p1, p2, n, seed=99):
    """跑一次实验，返回（两率, 绝对增量, 95% CI）。"""
    rng = np.random.default_rng(seed)
    r1 = rng.binomial(n, p1) / n
    r2 = rng.binomial(n, p2) / n
    diff = r2 - r1
    se = np.sqrt(r1 * (1 - r1) / n + r2 * (1 - r2) / n)
    return r1, r2, diff, (diff - Z_ALPHA * se, diff + Z_ALPHA * se)


def demo():
    print('=== 增量实验：样本量、功效与置信区间 ===')
    print('基线转化率 %.1f%%\n' % (BASE_RATE * 100))

    print('--- 第一部分：想检测多大的提升，需要多少样本 ---')
    print('%12s %14s %10s' % ('相对提升', '每组样本量', '绝对差值'))
    print('-' * 40)
    plans = []
    for rel in (0.05, 0.10, 0.20):
        p2 = BASE_RATE * (1 + rel)
        n = sample_size(BASE_RATE, p2)
        plans.append((rel, n))
        print('%11.0f%% %14s %10.4f' % (rel * 100, f'{n:,}', p2 - BASE_RATE))
    print()
    n5, n10 = plans[0][1], plans[1][1]
    print('  验证「缩一半要翻四倍」：%.0f%% 需 %s，%.0f%% 需 %s，倍数 %.1f' % (
        5, f'{n5:,}', 10, f'{n10:,}', n5 / n10))
    print()

    print('--- 第二部分：样本量不足时，「显著」有多不可靠 ---')
    print('%12s %10s %10s' % ('每组样本量', '功效', '结论'))
    print('-' * 36)
    for n in (2000, 10000, 50000, 200000):
        p2 = BASE_RATE * 1.10
        power = simulate(BASE_RATE, p2, n)
        note = '能检出' if power >= 0.75 else ('很不稳' if power >= 0.3 else '基本测不出')
        print('%12s %9.0f%% %10s' % (f'{n:,}', power * 100, note))
    print()

    print('--- 第三部分：一次真实规模的实验 ---')
    p2 = BASE_RATE * 1.10
    n = sample_size(BASE_RATE, p2)
    r1, r2, diff, (lo, hi) = one_experiment(BASE_RATE, p2, n)
    print('  每组 %s 人：对照组 %.2f%%，实验组 %.2f%%' % (f'{n:,}', r1 * 100, r2 * 100))
    print('  绝对增量 %+.4f（%.2f 个百分点）' % (diff, diff * 100))
    print('  95%% CI [%+.4f, %+.4f] -> %s' % (
        lo, hi, '显著（不含 0）' if lo > 0 else '不显著（跨过 0）'))
    print()

    print('  再跑一次「只投三天、每组 800 人」的小实验：')
    r1s, r2s, ds, (los, his) = one_experiment(BASE_RATE, p2, 800, seed=7)
    print('    对照组 %.2f%%，实验组 %.2f%%，增量 %+.4f，CI [%+.4f, %+.4f] -> %s' % (
        r1s * 100, r2s * 100, ds, los, his,
        '显著' if los > 0 else '不显著——但真实提升确实存在'))
    print()

    print('结论：')
    print('  1) 样本量与可检测差异的平方成反比：差异缩一半，样本翻四倍。')
    print('  2) 样本不足时，「不显著」只说明证据不足，不说明没有效果。')
    print('  3) 小样本实验很容易「什么都测不出来」，也容易「假阳」——所以要先算样本量。')
    print('  4) 判读要看「点估计 + 置信区间 + 经济阈值」，不能只看 p 值。')
    return plans


def save_visual(path='visuals/mkt_incrementality.svg'):
    """功效随样本量的变化。"""
    from chartlib import LineChart
    ch = LineChart('样本量越大，越能检出真实提升', '每组样本量（对数刻度）', '检出概率（%）')
    xs = [2000, 10000, 50000, 200000]
    ys = [round(simulate(BASE_RATE, BASE_RATE * 1.10, n) * 100, 1) for n in xs]
    ch.add_line('真实提升 10%', xs, ys)
    ch.add_note('样本不足时，「不显著」只说明证据不足，不代表没有效果')
    return ch.save(path)


if __name__ == '__main__':
    demo()
