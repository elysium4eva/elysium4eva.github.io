#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs_rfm_segment.py -- RFM 分层与 K-means 对比

用途：
    30 个客户，按 R/F/M 三维打分分成八个群，再用 K-means 聚一次，
    对比「规则分箱」与「数据聚类」的差别。

输入：
    内置 30 个客户的最近购买天数、购买次数、累计金额
输出：
    八个分群的客户数、K-means 的四个群心、两者的差异
依赖：
    numpy、viz_png
对应课件：
    模块六 §8 CRM 与客户分层 / §10 类四算法
"""

import numpy as np

# (最近购买天数, 购买次数, 累计金额)
CUSTOMERS = [
    (5, 8, 2400), (12, 6, 1800), (30, 5, 1500), (60, 4, 1200),
    (8, 3, 900), (25, 2, 700), (90, 9, 3000), (150, 7, 2600),
    (3, 1, 300), (40, 1, 250), (200, 2, 800), (300, 1, 500),
    (15, 10, 4200), (22, 4, 1100), (70, 3, 950), (110, 6, 2100),
    (7, 2, 600), (45, 8, 2800), (180, 3, 1000), (250, 2, 650),
    (10, 5, 1600), (35, 7, 2300), (85, 2, 550), (130, 1, 400),
    (2, 3, 850), (55, 5, 1700), (160, 4, 1300), (280, 1, 320),
    (20, 9, 3600), (65, 6, 2000),
]

SEG_NAME = {
    ('H', 'H', 'H'): '重要价值', ('L', 'H', 'H'): '重要唤回',
    ('H', 'L', 'H'): '重要发展', ('L', 'L', 'H'): '重要挽留',
    ('H', 'H', 'L'): '一般价值', ('L', 'H', 'L'): '一般保持',
    ('H', 'L', 'L'): '新客',     ('L', 'L', 'L'): '流失风险',
}


def rfM_scores():
    arr = np.array(CUSTOMERS, dtype=float)
    # R 越小越好 -> 取负后按中位数分档
    r_med = np.median(arr[:, 0])
    f_med = np.median(arr[:, 1])
    m_med = np.median(arr[:, 2])
    out = []
    for r, f, m in arr:
        rl = 'H' if r <= r_med else 'L'
        fl = 'H' if f >= f_med else 'L'
        ml = 'H' if m >= m_med else 'L'
        out.append((rl, fl, ml))
    return out, (r_med, f_med, m_med)


def kmeans(X, k=4, iters=30, seed=3):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), k, replace=False)
    C = X[idx].copy()
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        lab = d.argmin(axis=1)
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(axis=0)
    return lab, C


def demo():
    print('=== RFM 分层与 K-means 对比 ===')
    print('客户 %d 个\n' % len(CUSTOMERS))

    labels, med = rfM_scores()
    print('--- 分档阈值（中位数）---')
    print('  R 中位数 %.0f 天（越小越好）  F %.0f 次  M %.0f 元' % med)
    print()

    print('--- 八个分群 ---')
    counts = {}
    for lb in labels:
        counts[lb] = counts.get(lb, 0) + 1
    for key in [('H','H','H'),('L','H','H'),('H','L','H'),('L','L','H'),
                ('H','H','L'),('L','H','L'),('H','L','L'),('L','L','L')]:
        n = counts.get(key, 0)
        print('  %-6s %s  %2d 人' % (SEG_NAME[key], ''.join(key), n))
    print()

    # K-means（标准化后）
    X = np.array(CUSTOMERS, dtype=float)
    Xn = (X - X.mean(axis=0)) / X.std(axis=0)
    lab, C = kmeans(Xn, k=4)
    print('--- K-means（k=4，标准化后）群心 ---')
    for j in range(4):
        n = int((lab == j).sum())
        cnt = X[lab == j]
        print('  群%d：%2d 人，R 均值 %.0f 天，F 均值 %.1f 次，M 均值 %.0f 元' % (
            j + 1, n, cnt[:, 0].mean(), cnt[:, 1].mean(), cnt[:, 2].mean()))
    print()

    print('结论：')
    print('  1) RFM 分箱用中位数切，结果可解释、跨期可比——适合对外汇报与考核。')
    print('  2) K-means 按数据分布聚，能看到「人想不到」的组合——但换数据群心会变，不可跨期比。')
    print('  3) 分层的真正价值是「决定谁不投」：流失风险群只做低成本召回。')
    print('  4) 别忘标准化：金额量纲远大于频次，不标准化会被金额主导。')
    return labels, lab, C


def save_png(path='visuals/cs_rfm_segment.png'):
    from viz_png import Canvas, RED, GOLD, INK, MUTED, LINE, GOLD_L, RED_L
    labels, lab, C = demo()
    cv = Canvas(1150, 660)
    cv.title('RFM 分层：八个分群与 K-means 的差别')

    # 左：R vs F 散点，颜色=分群（按 M 高/低 + RF 组合）
    ox, oy, w, h = 130, 130, 420, 400
    cv.rect(ox, oy, w, h, fill=(252, 250, 249), outline=LINE, width=1, r=6)
    maxr, maxf = 320.0, 10.0
    for (r, f, m), lb in zip(CUSTOMERS, labels):
        x = ox + (r / maxr) * (w - 30) + 15
        y = oy + h - (f / maxf) * (h - 30) - 15
        col = GOLD if lb[2] == 'H' else RED
        cv.node(x, y, 6 if lb[2] == 'H' else 5, fill=col, outline=col, width=1)
    cv.text(ox + w / 2, oy + h + 14, 'R：最近购买天数（越左越近）→', 11, MUTED, False, 'ma')
    cv.text(ox - 12, oy + h / 2, 'F 频次', 11, MUTED, False, 'rm')
    cv.text(ox + 12, oy + 10, '金 = M 高（高价值）', 11, GOLD, True)
    cv.text(ox + 12, oy + 30, '红 = M 低', 11, RED, True)

    # 右：八群人数
    px = 600
    cv.text(px, 110, '八个分群的人数', 14, INK, True)
    counts = {}
    for lb in labels:
        counts[lb] = counts.get(lb, 0) + 1
    keys = [('H','H','H'),('L','H','H'),('H','L','H'),('L','L','H'),
            ('H','H','L'),('L','H','L'),('H','L','L'),('L','L','L')]
    for i, k in enumerate(keys):
        y = 140 + i * 44
        n = counts.get(k, 0)
        name = SEG_NAME[k]
        cv.rect(px, y, 460, 36, fill=GOLD_L if k[2] == 'H' else (255, 255, 255),
                outline=GOLD if k[2] == 'H' else LINE, width=2 if k[2] == 'H' else 1, r=6)
        cv.text(px + 12, y + 9, name, 12, INK, True)
        cv.text(px + 130, y + 10, ''.join(k), 11, MUTED)
        cv.rect(px + 210, y + 12, int(n * 22), 12, fill=GOLD if k[2] == 'H' else RED,
                outline=None, width=0, r=3)
        cv.text(px + 440, y + 8, '%d 人' % n, 11, INK, True, 'ra')

    cv.text(px, 500, 'K-means 与 RFM 的关键差别：', 12, INK, True)
    cv.text(px, 524, '分箱是「绝对的」（跨期可比），聚类是「相对的」（换数据就变）', 11, MUTED)
    cv.text(px, 552, '分层的价值：决定谁不投，而不是给谁更多', 11, GOLD, True)
    cv.text(60, 620, '数据：30 个客户；程序 cs_rfm_segment.py', 11, MUTED)
    return cv.save(path)


if __name__ == '__main__':
    demo()
