#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loc_quality_eval.py -- 翻译质量评估：BLEU 分解与多版对比

用途：
    对同一句原文的多版英文译文计算 BLEU，并把 BLEU 拆成四个 n-gram 阶数，
    演示三件事：
      1) 词序一变，某阶精确率为 0，整体 BLEU 直接归零；
      2) 同义改写拿不到高分，尽管意思是对的；
      3) 数字写错只掉一点分 —— 所以数字必须单独校验。

输入：
    原文（中文，仅作对照）、参考译文、三版候选译文（内置）
输出：
    每版译文的分阶精确率、长度惩罚、BLEU 值，以及数字校验结果
依赖：
    numpy（可选 chartlib 出 SVG 图）
对应课件：
    模块五第四章 §3 算法 3-4 翻译质量评估
"""

import re
from collections import Counter

REFERENCE = 'IPX7 waterproof with 12 hours of battery life, built for outdoor use'

CANDIDATES = {
    'A 忠实用词': 'IPX7 waterproof with 12 hours of battery life, built for outdoor use',
    'B 同义改写': 'Waterproof to IPX7, 12h playback, designed for the outdoors',
    'C 词序打乱': 'outdoor for built, life battery of hours 12 with waterproof IPX7',
    'D 数字写错': 'IPX7 waterproof with 24 hours of battery life, built for outdoor use',
}


def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def bleu_detail(ref, cand, max_n=4):
    """返回（各阶精确率, 长度惩罚, BLEU）。"""
    rt = ref.lower().split()
    ct = cand.lower().split()
    precisions = []
    for n in range(1, max_n + 1):
        rc = Counter(ngrams(rt, n))
        cc = Counter(ngrams(ct, n))
        clip = sum(min(v, rc[g]) for g, v in cc.items())
        total = sum(cc.values())
        precisions.append(clip / total if total else 0.0)
    c, r = len(ct), len(rt)
    bp = 1.0 if c >= r else (0.0 if c == 0 else pow(2.718281828, 1 - r / c))
    # 任一阶为 0 则整体为 0（标准 BLEU 用对数平均，log(0) 未定义，按 0 处理）
    if any(p == 0 for p in precisions):
        return precisions, bp, 0.0
    import math
    score = bp * math.exp(sum(math.log(p) for p in precisions) / max_n)
    return precisions, bp, score


def number_check(ref, cand):
    """数字回填校验：候选里的数字集合是否与参考一致。"""
    rn = set(re.findall(r'\d+', ref))
    cn = set(re.findall(r'\d+', cand))
    return sorted(rn), sorted(cn), (rn == cn)


def demo():
    print('=== 翻译质量评估：BLEU 分解 ===')
    print('参考译文: %s\n' % REFERENCE)
    print('%-12s %7s %7s %7s %7s %7s %7s' % (
        '候选', 'P1', 'P2', 'P3', 'P4', 'BP', 'BLEU'))
    print('-' * 62)
    rows = []
    for name, cand in CANDIDATES.items():
        p, bp, score = bleu_detail(REFERENCE, cand)
        print('%-12s %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f' % (
            name, p[0], p[1], p[2], p[3], bp, score))
        rows.append((name, score))
    print()

    print('--- 数字回填校验（BLEU 抓不到的那类错误）---')
    for name, cand in CANDIDATES.items():
        rn, cn, ok = number_check(REFERENCE, cand)
        flag = 'OK  ' if ok else '错误'
        print('  %-12s 参考数字 %-10s 候选数字 %-10s %s' % (
            name, ','.join(rn), ','.join(cn) or '(空)', flag))
    print()

    print('结论：')
    print('  1) B（同义改写）意思完全对，但 BLEU 明显低于 A —— 它惩罚「换了说法」。')
    print('  2) C（词序打乱）某一阶精确率为 0，整体 BLEU 归零 —— 极端但真实。')
    print('  3) D（24 小时写成 12 小时）BLEU 只掉一点点 —— 这是最危险的一类错误：')
    print('     指标看不出，用户会退货。所以必须单独做数字回填校验。')
    print('  4) 因此：BLEU 只能做「版本回归对比」，不能做「绝对质量判定」；')
    print('     正式验收要加人工 MQM 与数字/单位校验。')
    return rows


def save_visual(path='visuals/loc_quality_eval.svg'):
    """四版译文的 BLEU 对比。"""
    from chartlib import BarChart
    rows = demo()
    ch = BarChart('四版译文的 BLEU：指标看不出最危险的错误', '译文版本', 'BLEU')
    ch.add_series('BLEU', [r[0] for r in rows], [round(r[1], 3) for r in rows])
    ch.add_note('D 把 12 小时写成 24 小时，BLEU 只掉一点——数字必须单独校验')
    return ch.save(path)


if __name__ == '__main__':
    demo()
