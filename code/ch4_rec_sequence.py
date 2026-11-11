#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ch4_rec_sequence.py -- 序列推荐：一阶转移（极简版）

用途：
    用一条只有 12 步的浏览序列，演示「序列推荐」和「普通推荐」的区别：
    普通推荐只看「你总共喜欢什么」，序列推荐看「你上一步看了什么」。

输入：
    内置一条行为序列（商品 id）+ 全体用户的一阶转移统计
输出：
    下一个商品的预测排序（序列模型 vs 热门基线）、命中率对比
依赖：
    numpy、viz_png
对应课件：
    模块五第四章 §9 算法三（序列推荐）

说明：
    这是 SASRec 的「最小骨架」——SASRec 用自注意力看更长的历史，
    这里只用前 1 步（一阶马尔可夫）。原理一样：用历史预测下一个。
"""

import numpy as np

ITEMS = ['耳机', '手机壳', '充电线', '音箱', '支架', '贴膜']

# 训练序列（多条用户行为，item 用下标表示）
SEQS = [
    [0, 1], [0, 2], [0, 2], [1, 5], [1, 5], [2, 4],
    [3, 2], [3, 4], [4, 5], [3, 2], [2, 4], [5, 1],
    [0, 2], [3, 2], [1, 5],
]

# 待预测的测试序列：看最后一步之后会买什么
TEST = [0, 2]        # 「耳机 -> 充电线」之后？
TRUTH = 4            # 真实下一位是「支架」


def transitions(seqs, n=len(ITEMS)):
    """统计一阶转移次数 T[a][b] = 序列中 a 后面接 b 的次数。"""
    T = np.zeros((n, n))
    for s in seqs:
        for a, b in zip(s[:-1], s[1:]):
            T[a, b] += 1
    return T


def popularity(seqs, n=len(ITEMS)):
    c = np.zeros(n)
    for s in seqs:
        for x in s[1:]:
            c[x] += 1
    return c


def predict_seq(T, last, n=len(ITEMS)):
    """序列模型：直接看上一步的转移分布。"""
    row = T[last].copy()
    return row / row.sum() if row.sum() else np.ones(n) / n


def demo():
    print('=== 序列推荐：一阶转移（极简版）===')
    print('商品：' + '、'.join('%d=%s' % (i, x) for i, x in enumerate(ITEMS)))
    print('训练序列 %d 条，总步数 %d\n' % (len(SEQS), sum(len(s) for s in SEQS)))

    T = transitions(SEQS)
    print('--- 一阶转移矩阵（行=当前，列=下一个）---')
    print('        ' + '  '.join('%4s' % x for x in ITEMS))
    for i, name in enumerate(ITEMS):
        print('%4s    ' % name + '  '.join('%4.0f' % v for v in T[i]))
    print()

    last = TEST[-1]
    p_seq = predict_seq(T, last)
    p_pop = popularity(SEQS)
    p_pop = p_pop / p_pop.sum()

    print('--- 测试：看过「%s」之后，下一个买什么 ---' % ITEMS[last])
    order_seq = np.argsort(-p_seq)
    order_pop = np.argsort(-p_pop)
    print('  序列模型排序：' + ' > '.join(ITEMS[i] for i in order_seq[:4]))
    print('  热门基线排序：' + ' > '.join(ITEMS[i] for i in order_pop[:4]))
    print()

    def rank_of(order, target):
        return int(np.where(order == target)[0][0]) + 1

    r_seq = rank_of(order_seq, TRUTH)
    r_pop = rank_of(order_pop, TRUTH)
    print('--- 真实下一位是「%s」 ---' % ITEMS[TRUTH])
    print('  序列模型排名：第 %d 位  %s' % (r_seq, '命中@1' if r_seq == 1 else '未中@1'))
    print('  热门基线排名：第 %d 位  %s' % (r_pop, '命中@1' if r_pop == 1 else '未中@1'))
    print()

    print('结论：')
    print('  1) 热门基线是「全局最喜欢什么」，序列模型是「这一步之后最可能什么」。')
    print('  2) 当行为有先后依赖（先买耳机再买线），序列模型会明显更强。')
    print('  3) SASRec 做的事就是把这个「一阶」扩成「看前 N 步」——用自注意力加权。')
    print('  4) 代价：序列模型需要更多数据；新用户没有历史时退化为热门基线（冷启动）。')
    return p_seq, p_pop, r_seq, r_pop


def save_png(path='visuals/ch4_rec_sequence.png'):
    from viz_png import Canvas, RED, RED_L, GOLD, GOLD_L, INK, MUTED, LINE, lerp

    p_seq, p_pop, r_seq, r_pop = demo()
    T = transitions(SEQS)
    cv = Canvas(1200, 700)
    cv.title('序列推荐：看「上一步」还是看「全局」')

    # 上：行为序列条
    cv.text(60, 100, '一条用户行为序列（12 步）', 16, INK, True)
    seq_demo = [0, 2, 4, 5, 1, 2, 4, 3, 2, 4, 1, 5]
    x0, y0, ww, gap = 60, 140, 78, 14
    for i, it in enumerate(seq_demo):
        x = x0 + i * (ww + gap)
        cv.rect(x, y0, ww, 58, fill=RED_L if i < 11 else GOLD_L,
                outline=RED if i < 11 else GOLD, width=2, r=8)
        cv.text(x + ww / 2, y0 + 20, ITEMS[it], 13, INK, True, 'mm')
        cv.text(x + ww / 2, y0 + 42, 't%d' % (i + 1), 11, MUTED, False, 'mm')
    cv.text(x0 + 11 * (ww + gap) + ww + 16, y0 + 26, '→ ？', 20, GOLD, True)
    cv.text(60, y0 + 74, '金框 = 待预测的下一个商品', 13, GOLD, True)

    # 左下：转移矩阵
    mx, my, cs = 90, 290, 62
    cv.text(mx - 20, my - 12, '一阶转移矩阵（行=当前，列=下一个；越深越多）', 14, INK, True)
    for j, name in enumerate(ITEMS):
        cv.text(mx + j * cs + cs / 2, my + 8, name[:2], 11, MUTED, False, 'mm')
    for i, name in enumerate(ITEMS):
        cv.text(mx - 8, my + 30 + i * cs + cs / 2, name[:2], 11, INK, False, 'rm')
        for j in range(len(ITEMS)):
            v = T[i, j]
            t = min(v / 4.0, 1.0)
            x, y = mx + j * cs, my + 30 + i * cs
            cv.rect(x + 1, y + 1, cs - 2, cs - 2, fill=lerp((250, 247, 246), RED, t * 0.85),
                    outline=None, width=0, r=3)
            if v:
                cv.text(x + cs / 2, y + cs / 2, '%d' % v, 13,
                        (255, 255, 255) if t > 0.5 else INK, True, 'mm')

    # 右下：两种排序
    px, py = 640, 290
    cv.text(px, py - 12, '下一个买什么：两种排序', 16, INK, True)
    order_seq = np.argsort(-p_seq)[:4]
    order_pop = np.argsort(-p_pop)[:4]
    for row_i, (label, order, col) in enumerate([
            ('序列模型（看上一步 %s）' % ITEMS[TEST[-1]], order_seq, GOLD),
            ('热门基线（看全局销量）', order_pop, RED)]):
        yy = py + 30 + row_i * 150
        cv.text(px, yy, label, 14, INK, True)
        for k, idx in enumerate(order):
            x = px + k * 118
            hit = (idx == TRUTH)
            cv.rect(x, yy + 28, 104, 72, fill=GOLD_L if hit else (255, 255, 255),
                    outline=col, width=3 if hit else 1, r=8)
            cv.text(x + 52, yy + 48, ITEMS[idx], 13, INK, True, 'mm')
            cv.text(x + 52, yy + 76, '#%d' % (k + 1), 12, col, True, 'mm')
            if hit:
                cv.text(x + 52, yy + 112, '命中', 12, GOLD, True, 'mm')

    cv.text(px, py + 300, '真实下一位：%s → 序列模型第 %d 位，热门基线第 %d 位'
            % (ITEMS[TRUTH], r_seq, r_pop), 14, INK, True)
    cv.text(px, py + 328, 'SASRec = 把这个一阶扩成「看前 N 步」', 12, MUTED)

    cv.text(30, 665, '数据：15 条训练序列、6 个商品；程序 ch4_rec_sequence.py', 12, MUTED)
    return cv.save(path)


if __name__ == '__main__':
    demo()
