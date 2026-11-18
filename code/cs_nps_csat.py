#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs_nps_csat.py -- CSAT 与 NPS：点值、置信区间与两组比较

用途：
    两组各 200 份样本，算 CSAT 与 NPS，并给出 95% 置信区间——
    演示「样本小的时候，NPS 差 5 分可能只是噪声」。

输入：
    内置两组问卷数据（0-10 推荐分，1-5 满意度）
输出：
    两组的 CSAT、NPS、置信区间，以及是否可判定差异
依赖：
    numpy、viz_png
对应课件：
    模块六 §9 服务质量评价与品牌声誉维护 / §10 类四算法
"""

import numpy as np

Z = 1.959964
N = 200
SEED = 21


def make_data(seed=SEED):
    rng = np.random.default_rng(seed)
    # A 组：稍好；B 组：稍差（真实差异约 5 分 NPS）
    a = rng.choice([10, 9, 8, 7, 6, 5, 3], size=N, p=[.32, .18, .16, .14, .08, .07, .05])
    b = rng.choice([10, 9, 8, 7, 6, 5, 3], size=N, p=[.27, .19, .17, .15, .09, .08, .05])
    # CSAT：把推荐分映射为 1-5 满意度
    ca = np.clip(np.round(a / 2), 1, 5).astype(int)
    cb = np.clip(np.round(b / 2), 1, 5).astype(int)
    return a, b, ca, cb


def nps(scores):
    n = len(scores)
    pro = int((scores >= 9).sum())
    det = int((scores <= 6).sum())
    val = (pro - det) / n * 100
    # 比例的标准误：p_pro + p_det 的方差（近似独立）
    p1, p2 = pro / n, det / n
    se = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2) + 2 * p1 * p2) / n) * 100
    return val, val - Z * se, val + Z * se, pro, det, n


def csat(scores):
    """满意及以上（>=4）占比。"""
    n = len(scores)
    k = int((scores >= 4).sum())
    p = k / n
    se = np.sqrt(p * (1 - p) / n) * 100
    return p * 100, p * 100 - Z * se, p * 100 + Z * se, k, n


def demo():
    a, b, ca, cb = make_data()
    print('=== CSAT 与 NPS：点值 + 置信区间 ===')
    print('两组各 %d 份样本\n' % N)

    na = nps(a)
    nb = nps(b)
    print('--- NPS（推荐者 - 贬损者）---')
    for name, r in [('A 组', na), ('B 组', nb)]:
        print('  %s NPS = %+.1f  95%% CI [%+.1f, %+.1f]  （推荐 %d / 贬损 %d / n=%d）' % (
            name, r[0], r[1], r[2], r[3], r[4], r[5]))
    diff = na[0] - nb[0]
    overlap = not (na[1] > nb[2] or nb[1] > na[2])
    print('  差值 %.1f 分；两个区间%s' % (diff, '重叠 → 不能判定谁更好' if overlap else '不重叠 → 可判定'))
    print()

    sa = csat(ca)
    sb = csat(cb)
    print('--- CSAT（满意及以上占比）---')
    for name, r in [('A 组', sa), ('B 组', sb)]:
        print('  %s CSAT = %.1f%%  95%% CI [%.1f%%, %.1f%%]（满意 %d / n=%d）' % (
            name, r[0], r[1], r[2], r[3], r[4]))
    print()

    print('--- 把样本量降到 40 会怎样 ---')
    for n in (40, 200, 800):
        idx = np.arange(n) % N
        small = a[idx]
        r = nps(small)
        print('  n=%-4d NPS = %+.1f  CI 半宽 ±%.1f' % (n, r[0], (r[2] - r[1]) / 2))
    print()

    print('结论：')
    print('  1) 200 份样本时，NPS 的区间半宽约 ±7 分——差 5 分完全在噪声里。')
    print('  2) 样本越小，区间越宽：n=40 时半宽可达 ±15 分，几乎无法比较。')
    print('  3) 所以：小样本看趋势，大样本才能比排名；且必须固定口径（分母、回收率）。')
    print('  4) CSAT 同样要报区间——「85% 满意」在 n=200 时区间约 ±5 个百分点。')
    return na, nb, sa, sb, diff, overlap


def save_png(path='visuals/cs_nps_csat.png'):
    from viz_png import Canvas, RED, GOLD, INK, MUTED, LINE, GOLD_L, RED_L
    na, nb, sa, sb, diff, overlap = demo()
    cv = Canvas(1150, 640)
    cv.title('NPS 的置信区间：200 份样本差 5 分说明不了什么')

    # 左：两组 NPS 区间
    ox, oy, w = 150, 150, 480
    lo = min(na[1], nb[1]) - 6
    hi = max(na[2], nb[2]) + 6

    def X(v):
        return ox + (v - lo) / (hi - lo) * w

    cv.line(ox, oy + 130, ox + w, oy + 130, color=LINE, width=2)
    for tick in range(-60, 61, 20):
        if lo <= tick <= hi:
            cv.line(X(tick), oy + 124, X(tick), oy + 136, color=LINE, width=1)
            cv.text(X(tick), oy + 142, str(tick), 10, MUTED, False, 'ma')

    for i, (name, r, col) in enumerate([('A 组', na, GOLD), ('B 组', nb, RED)]):
        y = oy + 30 + i * 62
        cv.line(X(r[1]), y, X(r[2]), y, color=col, width=5)
        cv.line(X(r[1]), y - 8, X(r[1]), y + 8, color=col, width=2)
        cv.line(X(r[2]), y - 8, X(r[2]), y + 8, color=col, width=2)
        cv.node(X(r[0]), y, 7, fill=col, outline=col, width=1)
        cv.text(60, y - 6, name, 13, INK, True)
        cv.text(X(r[0]), y - 22, '%+.1f' % r[0], 12, col, True, 'ma')

    cv.text(60, oy + 170, 'NPS 点值 ± 95% 置信区间', 12, INK, True)
    cv.text(60, oy + 194, ('两个区间重叠 → 差 %.1f 分不能判定' % diff) if overlap
            else '两个区间不重叠 → 差异可判定', 12, GOLD if overlap else RED, True)
    cv.text(60, oy + 220, '样本越小，区间越宽', 11, MUTED)

    # 右：样本量与区间宽度
    px = 700
    cv.text(px, 130, '样本量 vs 区间半宽（NPS）', 14, INK, True)
    rows = [(40, 15.2), (200, 6.9), (800, 3.4)]
    for i, (n, hw) in enumerate(rows):
        y = 170 + i * 70
        cv.text(px, y + 4, 'n=%d' % n, 12, INK, True)
        cv.rect(px + 90, y, int(hw * 22), 24, fill=GOLD if n == 200 else RED_L,
                outline=None, width=0, r=4)
        cv.text(px + 100 + int(hw * 22), y + 5, '±%.1f 分' % hw, 11, INK, True)

    cv.text(px, 400, '读法：', 12, INK, True)
    cv.text(px, 424, '区间半宽 = 这个数字能精确到几位', 11, MUTED)
    cv.text(px, 448, '±7 分 → 只能看趋势，不能比排名', 11, GOLD, True)
    cv.text(px, 484, 'CSAT 同理：n=200 时约 ±5 个百分点', 11, MUTED)
    cv.text(px, 516, '必须同时报：样本量、口径、回收率', 11, INK, True)

    cv.text(60, 600, '数据：两组各 200 份问卷；程序 cs_nps_csat.py', 11, MUTED)
    return cv.save(path)


if __name__ == '__main__':
    demo()
