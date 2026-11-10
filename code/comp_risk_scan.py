#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comp_risk_scan.py -- 合规风险扫描：规则 + 语义两级

用途：
    对一批商品文案做两级风险扫描：
      第一级 规则扫描：禁用词、绝对化用语、未授权认证；
      第二级 语义扫描：用「已知违规表述」的字符级相似度抓同义改写。
    重点演示：规则扫描会漏掉同义改写，而语义扫描能补上，但会引入假阳。

输入：
    待检文案（内置 8 条）+ 禁用词表 + 违规样例库（内置）
输出：
    每条文案的风险分、命中的规则、语义最近样例，以及改写建议
依赖：
    numpy（可选 chartlib 出 SVG 图）
对应课件：
    模块五第四章 §7 合规的技术手段与算法
"""

import re
from difflib import SequenceMatcher

# 禁用词与绝对化用语（简化示例，实际要按市场与平台维护）
BLOCKLIST = {
    'best': 3, '#1': 3, 'only': 2, '100%': 3, 'lowest price': 3,
    'cure': 3, 'treat ': 3, 'prevent cancer': 3,
    'military-grade': 2, 'national authority': 2, 'top-rated': 2,
    'original price': 1,
}

# 已确认违规的表述样例库（语义扫描用）
EXEMPLARS = [
    'no other product comes close',
    'clinically proven to cure disease',
    'guaranteed lowest price online',
    'certified by national authority',
    'military-grade durability test',
]

TEXTS = [
    'IPX7 waterproof, 12-hour battery life, tested at 70% volume.',
    'No other speaker on the market comes close to this one.',
    'Clinically proven to cure sleep problems.',
    'We guarantee the lowest price anywhere online.',
    'Passed a 1.5 m drop test onto concrete.',
    'Military-grade durability, certified by national authority.',
    'Original price 99.99, now 39.99.',
    'Best sound in its class according to our own lab test.',
]


def rule_scan(text):
    """规则扫描：返回（分数, 命中列表）。"""
    hits = []
    score = 0
    for kw, w in BLOCKLIST.items():
        if kw in text:
            hits.append(kw)
            score += w
    return score, hits


def semantic_scan(text, thr=0.55):
    """语义扫描（字符级相似度近似）：返回（最高相似度, 最近样例）。"""
    def wordset(x):
        return set(re.findall(r"[a-z0-9]+", x.lower()))

    tw = wordset(text)
    best, best_ex = 0.0, ''
    for ex in EXEMPLARS:
        ew = wordset(ex)
        jac = len(tw & ew) / max(len(tw | ew), 1)
        seq = SequenceMatcher(None, text.lower(), ex.lower()).ratio()
        r = max(jac, seq)
        if r > best:
            best, best_ex = r, ex
    return best, best_ex


def scan(texts, alpha=0.6, beta=0.4):
    """两级扫描，输出风险分 = alpha*规则归一 + beta*语义相似度。"""
    rows = []
    for t in texts:
        rs, hits = rule_scan(t)
        ss, ex = semantic_scan(t)
        # 规则分归一化：满分按 3 分算
        risk = alpha * min(rs / 3.0, 1.0) + beta * ss
        verdict = '阻断' if risk >= 0.75 else ('警告' if risk >= 0.35 else '通过')
        rows.append({
            'text': t, 'rule': rs, 'hits': hits,
            'sem': ss, 'ex': ex, 'risk': risk, 'verdict': verdict,
        })
    rows.sort(key=lambda r: -r['risk'])
    return rows


def demo():
    print('=== 合规风险扫描：规则 + 语义两级 ===')
    print('禁用词表 %d 条，违规样例库 %d 条\n' % (len(BLOCKLIST), len(EXEMPLARS)))
    rows = scan(TEXTS)

    print('%-4s %-6s %-5s %-5s %-6s %s' % ('#', '规则分', '语义', '风险', '判定', '文案'))
    print('-' * 96)
    for i, r in enumerate(rows, 1):
        print('%-4d %-6d %-5.2f %-5.2f %-6s %s' % (
            i, r['rule'], r['sem'], r['risk'], r['verdict'], r['text'][:52]))
    print()

    print('--- 逐条明细（命中与近似样例）---')
    for i, r in enumerate(rows, 1):
        if r['verdict'] == '通过':
            continue
        print('  #%d 判定：%s  风险 %.2f' % (i, r['verdict'], r['risk']))
        print('     文案：%s' % r['text'])
        if r['hits']:
            print('     规则命中：%s' % '、'.join(r['hits']))
        if r['sem'] >= 0.55:
            print('     语义最近：%s（相似度 %.2f）' % (r['ex'], r['sem']))
        print()

    n_rule_only = sum(1 for r in rows if r['rule'] == 0 and r['sem'] >= 0.55)
    beta = 0.4  # 与 scan() 的默认权重保持一致
    print('结论：')
    print('  1) 规则扫描能抓住 %d 条（含禁用词/绝对化用语）。' %
          sum(1 for r in rows if r['rule'] > 0))
    print('  2) 有 %d 条文案规则完全没命中，但语义相似度已达 %.2f——' % (
        n_rule_only, max([r['sem'] for r in rows if r['rule'] == 0] or [0])))
    print('     比如「No other speaker comes close」（没有别的能比）——这就是同义改写。')
    print('     注意它当前被判为「通过」：语义分 %.2f 乘以权重 %.1f 只有 %.2f，' % (
        max([r['sem'] for r in rows if r['rule'] == 0] or [0]), beta,
        beta * max([r['sem'] for r in rows if r['rule'] == 0] or [0])))
    print('     没到警告线。这说明「权重与阈值」本身就是需要调的业务参数——')
    print('     想让这类改写被标出来，要么提高语义权重，要么降低警告阈值。')
    print('  3) 所以只做规则扫描会给你「虚假的安全感」。')
    print('  4) 但语义扫描也有假阳：相似度只看字面，可能把正常表述判为可疑——')
    print('     所以语义命中的结果必须人工复核，不能直接阻断。')
    print('  5) 正确架构：规则硬阻断 + 语义标可疑 + 人工签字放行。')
    return rows


def save_visual(path='visuals/comp_risk_scan.svg'):
    """八条文案的风险分排序。"""
    from chartlib import HBarChart
    rows = demo()
    labels = ['#%d  %s' % (i, r['text'][:28]) for i, r in enumerate(rows, 1)]
    ch = HBarChart('八条文案的风险分（规则 + 语义）', '风险分', '文案')
    ch.add(labels, [round(r['risk'], 2) for r in rows], color='#711426')
    ch.add_note('规则漏掉的同义改写由语义补上；但语义命中需人工复核，不能直接阻断')
    return ch.save(path)


if __name__ == '__main__':
    demo()
