#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs_complaint_triage.py -- 客诉识别与优先级分级

用途：
    20 条客诉，分别算「情绪分」（词典）与「严重度」（规则），
    排成三档优先级——重点演示：情绪 ≠ 严重度。

输入：
    内置 20 条客诉文本
输出：
    每条的情绪分、严重度分、象限与优先级
依赖：
    viz_png
对应课件：
    模块六 §7 客诉识别、情绪分析与售后补救
"""

NEG = {'差': 2, '烂': 3, '骗': 2, '垃圾': 3, '愤怒': 3, '气': 2,
       '慢': 2, '坏': 2, '失望': 2, '恶心': 3, '无语': 2, '半天': 2,
       '烦': 2, '糟糕': 3, '无语了': 2}

# 严重度关键词（后果维度，与情绪无关）
SEVERE = {'起诉': 5, '律师': 5, '监管': 5, '举报': 4, '曝光': 4, '投诉': 3,
          '赔偿': 3, '退款': 2, '退货': 2, '换货': 1}

COMPLAINTS = [
    '怎么这么慢！等半天了',
    '东西是坏的，我要退货',
    '你们这是欺诈，我要起诉',
    '好的，那我走监管投诉吧',
    '客服态度太差，我要投诉',
    '问一下能不能改地址',
    '发票什么时候能开',
    '收到的是空盒子，要退款',
    '物流一直不动，很失望',
    '我要曝光你们',
    '包装破了，里面没事',
    '尺码不对能换吗',
    '已经第三次问了，还没解决',
    '你们是不是不想退钱',
    '我要找律师',
    '谢谢，不过我还是想退',
    '质量太差了，垃圾',
    '能便宜点吗',
    '下单后一直没发货',
    '我要去平台举报你们',
]

HIGH_AMT = '赔偿'   # 简化：提到赔偿视为金额超限


def emotion_score(text):
    """情绪分：负面词 + 感叹号（强度标记）。"""
    s = sum(w for k, w in NEG.items() if k in text)
    s += 2 * (text.count('!') + text.count('\uff01'))
    return s


def severity_score(text):
    s = sum(w for k, w in SEVERE.items() if k in text)
    if HIGH_AMT in text:
        s += 2
    return s


def triage():
    rows = []
    for t in COMPLAINTS:
        e = emotion_score(t)
        s = severity_score(t)
        # 归一化到 0~1（便于画象限）
        en = min(e / 10.0, 1.0)
        sn = min(s / 8.0, 1.0)
        quad = ('高情绪/高严重' if en >= .4 and sn >= .4 else
                '低情绪/高严重' if sn >= .4 else
                '高情绪/低严重' if en >= .4 else '低情绪/低严重')
        level = 'P1' if sn >= .4 else ('P2' if (sn >= .2 or en >= .35) else 'P3')
        rows.append((t, e, s, en, sn, quad, level))
    rows.sort(key=lambda r: (-r[4], -r[3]))
    return rows


def demo():
    print('=== 客诉识别与优先级分级 ===')
    print('客诉 %d 条；情绪词典 %d 词，严重度词 %d 个\n' % (
        len(COMPLAINTS), len(NEG), len(SEVERE)))

    rows = triage()
    print('--- 按优先级排序 ---')
    print('%-4s %-6s %-6s %-16s %s' % ('档', '情绪', '严重', '象限', '客诉'))
    for t, e, s, en, sn, quad, lv in rows:
        print('  %-4s %-6d %-6d %-16s %s' % (lv, e, s, quad, t[:22]))
    print()

    print('--- 只看情绪会漏掉谁 ---')
    calm_danger = [r for r in rows if r[4] >= .4 and r[3] < .35]
    print('  情绪低但严重度高（最危险）：%d 条' % len(calm_danger))
    for t, e, s, en, sn, quad, lv in calm_danger:
        print('    %s（情绪 %d / 严重 %d）→ %s' % (t, e, s, lv))
    print()

    print('结论：')
    print('  1) 「我要起诉」这类语气平静的句子，情绪分低但严重度最高——只看情绪必漏。')
    print('  2) 「怎么这么慢！」情绪高但严重度低——先安抚，不必花大代价。')
    print('  3) 所以优先级要二维：情绪决定「谁先被安抚」，严重度决定「谁先被处理」。')
    print('  4) 这就是第 7 节那张优先级矩阵的代码版。')
    return rows


def save_png(path='visuals/cs_complaint_triage.png'):
    from viz_png import Canvas, RED, GOLD, INK, MUTED, LINE, GOLD_L, RED_L
    rows = demo()
    cv = Canvas(1150, 660)
    cv.title('客诉优先级：情绪 × 严重度（四象限）')

    # 象限
    ox, oy, w, h = 130, 130, 420, 420
    cv.rect(ox, oy, w, h, fill=(252, 250, 249), outline=LINE, width=1, r=6)
    cv.line(ox + w / 2, oy, ox + w / 2, oy + h, color=MUTED, width=1, dash=True)
    cv.line(ox, oy + h / 2, ox + w, oy + h / 2, color=MUTED, width=1, dash=True)
    cv.text(ox + 12, oy + 10, '高情绪 / 高严重', 11, RED, True)
    cv.text(ox + w - 12, oy + 10, '低情绪 / 高严重', 11, RED, True, 'ra')
    cv.text(ox + 12, oy + h - 24, '高情绪 / 低严重', 11, MUTED, True)
    cv.text(ox + w - 12, oy + h - 24, '低情绪 / 低严重', 11, MUTED, True, 'ra')
    cv.text(ox + w / 2, oy + h + 14, '情绪强度 →', 11, MUTED, False, 'ma')
    cv.text(ox - 12, oy + h / 2, '严重度', 11, MUTED, False, 'rm')

    for t, e, s, en, sn, quad, lv in rows:
        x = ox + en * (w - 24) + 12
        y = oy + h - sn * (h - 24) - 12
        col = GOLD if lv == 'P1' else (RED if lv == 'P2' else MUTED)
        r = 7 if lv == 'P1' else 5
        cv.node(x, y, r, fill=col, outline=col, width=1)

    # 右：重点案例
    px = 600
    cv.text(px, 110, '两个必须能分辨的例子', 14, INK, True)
    cases = [
        ('怎么这么慢！等半天了', '情绪 6 / 严重 1', 'P3：先安抚，低成本', MUTED),
        ('好的，那我走监管投诉吧', '情绪 4 / 严重 7', 'P1：立即升级', GOLD),
    ]
    for i, (t, es, lv, col) in enumerate(cases):
        y = 140 + i * 92
        cv.rect(px, y, 500, 78, fill=GOLD_L if col == GOLD else (255, 255, 255),
                outline=col, width=2, r=10)
        cv.text(px + 14, y + 10, t, 13, INK, True)
        cv.text(px + 14, y + 34, es, 12, MUTED)
        cv.text(px + 14, y + 54, lv, 12, col, True)

    cv.text(px, 340, '只看情绪分的后果：', 13, INK, True)
    cv.text(px, 366, '把「最生气的」排第一，漏掉「最危险的」', 12, RED, True)
    cv.text(px, 396, '二维排序：情绪决定安抚顺序，', 11, MUTED)
    cv.text(px, 418, '严重度决定处理顺序', 11, MUTED)
    cv.text(px, 456, 'P1 = 立即升级 + 留证', 11, GOLD, True)
    cv.text(px, 478, 'P2 = 当班内处理', 11, MUTED)
    cv.text(px, 500, 'P3 = 标准流程', 11, MUTED)

    cv.text(60, 620, '数据：20 条客诉 / 情绪词典 20 词 / 严重度 10 词；程序 cs_complaint_triage.py', 11, MUTED)
    return cv.save(path)


if __name__ == '__main__':
    demo()
