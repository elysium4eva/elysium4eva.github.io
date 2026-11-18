#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs_intent_classify.py -- 客服意图识别（极简版）

用途：
    用 25 条极小语料、5 个意图，演示意图识别怎么做、为什么会混淆。
    重点不是准确率，而是让你看到「退货 / 换货」这类近似意图必然混。

输入：
    内置标注语料（5 意图 × 5 条）
输出：
    混淆矩阵、每个测试句的判分、最容易混的两类
依赖：
    numpy、viz_png
对应课件：
    模块六 §6 自动回复与多轮对话设计 / §10 类一算法
"""

import numpy as np

INTENTS = ['退货', '换货', '物流', '发票', '投诉']

CORPUS = [
    ('我想退货', '退货'), ('这个可以退吗', '退货'), ('不想要了能退吗', '退货'),
    ('申请退款退货', '退货'), ('退货流程怎么走', '退货'),
    ('想换个颜色', '换货'), ('能换一个吗', '换货'), ('这个坏了想换', '换货'),
    ('换货要多久', '换货'), ('尺码不对能换吗', '换货'),
    ('我的包裹到哪了', '物流'), ('快递怎么还没到', '物流'), ('物流一直不动', '物流'),
    ('什么时候能收到', '物流'), ('查一下快递', '物流'),
    ('能开发票吗', '发票'), ('发票怎么申请', '发票'), ('要一张增值税发票', '发票'),
    ('发票抬头写错了', '发票'), ('电子发票在哪下载', '发票'),
    ('我要投诉你们', '投诉'), ('客服态度太差', '投诉'), ('我要找你们领导', '投诉'),
    ('这个质量太差了', '投诉'), ('我要去平台举报', '投诉'),
]

TESTS = [
    ('这个可以退吗', '退货'),
    ('想换一个尺码', '换货'),
    ('快递三天没更新', '物流'),
    ('麻烦开个发票', '发票'),
    ('我要投诉', '投诉'),
    ('能不能换成别的颜色', '换货'),
    ('退钱', '退货'),
    # 下面三条是真正会混的（无意图关键词，或同时提到两个意图）
    ('东西坏了', '投诉'),
    ('退了能换吗', '换货'),
    ('退换都行', '退货'),
]


def char_ngrams(s, n=2):
    s = s.strip()
    return set(s[i:i + n] for i in range(len(s) - n + 1)) | set(s)


def vectorize(texts):
    """把所有文本转成字符 n-gram 计数向量（共享词表）。"""
    vocab = {}
    for t in texts:
        for g in char_ngrams(t):
            if g not in vocab:
                vocab[g] = len(vocab)
    X = np.zeros((len(texts), len(vocab)))
    for i, t in enumerate(texts):
        for g in char_ngrams(t):
            X[i, vocab[g]] += 1.0
    # L2 归一化
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n == 0, 1, n), vocab


def train_centroids():
    texts = [t for t, _ in CORPUS]
    X, vocab = vectorize(texts)
    cents = {}
    for k in INTENTS:
        idx = [i for i, (_, y) in enumerate(CORPUS) if y == k]
        cents[k] = X[idx].mean(axis=0)
    return cents, vocab


def predict(text, cents, vocab):
    """最近质心分类：把文本向量与 5 个质心比余弦相似。"""
    dim = len(vocab)
    v = np.zeros(dim)
    for g in char_ngrams(text):
        if g in vocab:
            v[vocab[g]] += 1.0
    nv = np.linalg.norm(v)
    if nv == 0:
        return None, {}
    v = v / nv
    scores = {}
    for k, c in cents.items():
        nc = np.linalg.norm(c)
        scores[k] = float(v @ c / (nc if nc else 1))
    best = max(scores, key=scores.get)
    return best, scores


def demo():
    print('=== 客服意图识别（极简版）===')
    print('意图 %d 个：%s' % (len(INTENTS), '、'.join(INTENTS)))
    print('语料 %d 条（每类 %d 条）\\n' % (len(CORPUS), len(CORPUS) // len(INTENTS)))

    cents, vocab = train_centroids()
    print('特征：字符 2-gram，共 %d 维\\n' % len(vocab))

    print('--- 测试句判定 ---')
    rows = []
    for text, truth in TESTS:
        pred, scores = predict(text, cents, vocab)
        ok = 'OK' if pred == truth else 'X '
        rows.append((text, truth, pred))
        top2 = sorted(scores.items(), key=lambda x: -x[1])[:2]
        print('  %-12s 真实=%-3s 预测=%-3s %s   次优=%s(%.2f)' % (
            text, truth, pred, ok, top2[1][0] if len(top2) > 1 else '-',
            top2[1][1] if len(top2) > 1 else 0))

    acc = sum(1 for _, t, p in rows if t == p) / len(rows)
    print('\\n  准确率 %.1f%%（%d/%d）' % (acc * 100, sum(1 for _, t, p in rows if t == p), len(rows)))

    # 混淆矩阵
    cm = np.zeros((len(INTENTS), len(INTENTS)), dtype=int)
    for _, t, p in rows:
        if p in INTENTS:
            cm[INTENTS.index(t), INTENTS.index(p)] += 1
    print('\\n--- 混淆矩阵（行=真实，列=预测）---')
    print('        ' + ''.join('%6s' % k for k in INTENTS))
    for i, k in enumerate(INTENTS):
        print('%6s  ' % k + ''.join('%6d' % v for v in cm[i]))

    print('\\n结论：')
    print('  1) 「退货」与「换货」共用「换」「退」这类单字，字符级特征必然混淆。')
    print('  2) 短句（如「退钱」）信息量太少，只靠文本很难判准。')
    print('  3) 对策：加意图专属关键词（退款、换新）作为强特征；短句直接给选项。')
    print('  4) 更要紧的是：判错时要有兜底——第 6 节讲过，意图可切换、可追问。')
    return cm, rows


def save_png(path='visuals/cs_intent_classify.png'):
    from viz_png import Canvas, RED, GOLD, INK, MUTED, LINE, lerp
    cm, rows = demo()
    cv = Canvas(1150, 660)
    cv.title('客服意图识别：为什么「退货/换货」必然混')

    # 左：混淆矩阵
    ox, oy, cs = 150, 150, 78
    cv.text(ox - 10, oy - 26, '混淆矩阵（行=真实，列=预测）', 15, INK, True)
    for j, k in enumerate(INTENTS):
        cv.text(ox + j * cs + cs / 2, oy - 6, k, 12, MUTED, True, 'mm')
    vmax = max(cm.max(), 1)
    for i, k in enumerate(INTENTS):
        cv.text(ox - 12, oy + i * cs + cs / 2, k, 12, INK, True, 'rm')
        for j in range(len(INTENTS)):
            x, y = ox + j * cs, oy + i * cs
            v = cm[i, j]
            t = v / vmax
            cv.rect(x + 2, y + 2, cs - 4, cs - 4, fill=lerp((250, 247, 246), RED, t * 0.85),
                    outline=None, width=0, r=6)
            if v:
                cv.text(x + cs / 2, y + cs / 2, str(v), 16,
                        (255, 255, 255) if t > 0.5 else INK, True, 'mm')

    cv.text(ox - 10, oy + 5 * cs + 18, '对角线上是判对的；对角外的格就是混淆', 12, MUTED)

    # 右：测试句判定
    px = 660
    cv.text(px, oy - 26, '七条测试句的判定', 15, INK, True)
    for i, (text, truth, pred) in enumerate(rows):
        y = oy - 6 + i * 52
        ok = truth == pred
        col = GOLD if ok else RED
        cv.rect(px, y, 440, 42, fill=(255, 255, 255), outline=col, width=2 if ok else 2, r=8)
        cv.text(px + 12, y + 11, text, 13, INK, True)
        cv.text(px + 300, y + 12, '真实 %s' % truth, 11, MUTED)
        cv.text(px + 380, y + 12, pred, 12, col, True)
        cv.text(px + 424, y + 11, 'OK' if ok else 'X', 13, col, True, 'rm')

    cv.text(px, oy + 7 * 52 + 10, '短句与近义意图最容易错——所以必须有追问与兜底', 12, GOLD, True)
    cv.text(60, 620, '数据：25 条语料 / 5 意图 / 7 条测试；程序 cs_intent_classify.py', 11, MUTED)
    return cv.save(path)


if __name__ == '__main__':
    demo()
