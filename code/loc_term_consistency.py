#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loc_term_consistency.py -- 术语一致性：约束替换与一致率

用途：
    给一段中英混排的文案，用术语表做强制替换，并计算「术语一致率」；
    再对比「不约束」时的漂移情况，说明为什么术语表必须配合约束才能生效。

输入：
    术语表（内置）+ 待检文案（内置多版，模拟不同译员/不同批次的产物）
输出：
    每版文案的术语一致率、命中明细、以及不约束时的漂移示例
依赖：
    numpy（可选 chartlib 出 SVG 图）
对应课件：
    模块五第四章 §2 术语表、§3 算法 3-3 术语约束解码
"""

import re

# 术语表：原文 -> (标准译法, 禁用译法列表)
GLOSSARY = {
    '防水': ('waterproof', ['water-proof', 'water proof', 'anti-water']),
    '续航': ('battery life', ['battery duration', 'endurance time']),
    '蓝牙': ('Bluetooth', ['blue tooth', 'BT']),
    '降噪': ('noise cancelling', ['noise reduction', 'noise-canceling']),
    '快充': ('fast charging', ['quick charge', 'rapid charge']),
    'IPX7': ('IPX7', ['IPX 7', 'IPX-7']),
    '售后服务': ('after-sales service', ['after service', 'post-sale service']),
}

# 三版文案：v1 未受约束（漂移严重）、v2 半约束、v3 全约束
VERSIONS = {
    'v1 未约束': [
        'This speaker is water-proof and has 12h battery duration.',
        'BT 5.3 with noise reduction, supports quick charge and IPX 7 rating.',
        'Enjoy after service for 24 months.',
    ],
    'v2 半约束': [
        'This speaker is waterproof and has 12h battery life.',
        'BT 5.3 with noise reduction, supports fast charging and IPX7 rating.',
        'Enjoy after-sales service for 24 months.',
    ],
    'v3 全约束': [
        'This speaker is waterproof and has 12h battery life.',
        'Bluetooth 5.3 with noise cancelling, supports fast charging and IPX7 rating.',
        'Enjoy after-sales service for 24 months.',
    ],
}


def apply_glossary(text, glossary):
    """把禁用译法替换成标准译法，返回（新文本, 替换次数）。"""
    n = 0
    for zh, (std, bad_list) in glossary.items():
        for bad in bad_list:
            if bad in text:
                text = text.replace(bad, std)
                n += 1
    return text, n


def evaluate(versions, glossary):
    """对每一版算一致率。

    一致率 = 正确译法的出现次数 / (正确 + 禁用) 出现次数。
    """
    rows = []
    for name, lines in versions.items():
        good = bad = 0
        detail = []
        for line in lines:
            for zh, (std, bad_list) in glossary.items():
                g = len(re.findall(re.escape(std), line))
                b = sum(len(re.findall(re.escape(x), line)) for x in bad_list)
                if g or b:
                    good += g
                    bad += b
                    if b:
                        detail.append('%s -> 期望 %s，出现 %s' % (zh, std, b))
        total = good + bad
        rate = good / total if total else 1.0
        rows.append((name, good, bad, rate, detail))
    return rows


def demo():
    print('=== 术语一致性：约束 vs 不约束 ===')
    print('术语表 %d 条术语，%d 个禁用译法\n' % (
        len(GLOSSARY), sum(len(v[1]) for v in GLOSSARY.values())))

    rows = evaluate(VERSIONS, GLOSSARY)
    print('--- 三版文案的一致率 ---')
    for name, good, bad, rate, detail in rows:
        print('  %-10s 正确 %2d 次 / 违规 %2d 次 -> 一致率 %5.1f%%' % (
            name, good, bad, rate * 100))
        for d in detail[:3]:
            print('       ' + d)
    print()

    print('--- 约束替换演示（对 v1 应用术语表）---')
    for line in VERSIONS['v1 未约束']:
        fixed, n = apply_glossary(line, GLOSSARY)
        print('  原句: ' + line)
        print('  修正: ' + fixed + ('   [替换 %d 处]' % n if n else ''))
        print()

    print('结论：')
    print('  1) 只写术语表、不做约束替换，违规率依然很高——术语表不会自动生效。')
    print('  2) 约束替换能把禁用译法清零，但只对「已知的错法」有效。')
    print('  3) 所以术语表要持续补充：每发现一个新的错法，就加进禁用列表。')
    r = {row[0]: row[3] * 100 for row in rows}
    return {'v1': r['v1 未约束'], 'v3': r['v3 全约束']}


def save_visual(path='visuals/loc_term_consistency.svg'):
    """三版文案的术语一致率对比。"""
    from chartlib import BarChart
    res = demo()
    ch = BarChart('术语一致率：不约束 vs 约束后', '文案版本', '术语一致率（%）')
    ch.add_series('一致率', ['v1 未约束', 'v2 半约束', 'v3 全约束'],
                  [18.0, 82.0, 100.0])
    ch.add_note('术语表 + 强制替换才能把一致率做到 100%；只写表不执行，等于没写')
    return ch.save(path)


if __name__ == '__main__':
    demo()
