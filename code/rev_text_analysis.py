#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rev_text_analysis.py —— 评论分析三件套：关键词、主题（LDA）、情感

对应课件：模块五 第二章 §6「评论分析的主要算法」
          — 关键词提取（TF-IDF / TextRank）
          — 主题发现（LDA / NMF / BERTopic）
          — 情感分析（词典法基线）

用途：
    A) 关键词：按星级分桶算 TF-IDF，分别得到「痛点词」与「卖点词」
    B) 主题：用坍缩吉布斯采样实现 LDA，自动发现评论里的几个主题
    C) 情感：词典法 + 否定词处理，给出句子级正负判断

为什么先做「按星级分桶」：
    全量情感分析在混合情感、反讽、非母语表达上误差很大。
    先按星级分桶，再在桶内做分析，能把问题简化一半——这是实务中最有效的降复杂度手段。

实现说明：
    · LDA 用吉布斯采样手写，不依赖 sklearn，便于看清算法在做什么
    · 情感用词典法，故意保留其局限（反讽处理不了），用来引出「为什么需要预训练模型」

依赖：numpy（本课程环境自带）
运行：python rev_text_analysis.py

数据说明：评论为教学用合成数据，英文，模拟跨境电商评论的语言特征。
"""

import math
import re
from collections import Counter, defaultdict

import numpy as np

STOP = set("""a an the is are was were be been being it its this that these those
i my me we our you your he she they them and or but if then so of to in on at for
with without from by as not no nor do does did done have has had can could would
should will just very too also more most much many some any all each other than
there here what which who whom when where why how""".split())

POS_WORDS = {
    "good": 1, "great": 1, "excellent": 1.5, "perfect": 1.5, "love": 1.5,
    "works": 1, "solid": 1, "sturdy": 1, "clear": 1, "loud": 1, "long": 0.5,
    "happy": 1, "recommend": 1.5, "amazing": 1.5, "nice": 1, "easy": 1,
    "waterproof": 0.8, "durable": 1, "value": 1, "fast": 0.8,
}
NEG_WORDS = {
    "bad": -1, "poor": -1, "terrible": -1.5, "awful": -1.5, "broken": -1.5,
    "leak": -1.5, "leaks": -1.5, "stopped": -1.2, "dead": -1.5, "fail": -1.5,
    "fake": -1.5, "cheap": -1, "useless": -1.5, "waste": -1.5, "disappointed": -1.2,
    "return": -1, "returned": -1, "refund": -1, "noise": -0.8, "quiet": 0.4,
    "weak": -1, "slow": -1, "short": -0.8, "drain": -1.2, "drains": -1.2,
    "disconnect": -1.2, "disconnects": -1.2, "flimsy": -1, "problem": -1,
}
NEGATORS = {"not", "no", "never", "n't", "cannot", "hardly", "barely", "without"}


# ---------------------------------------------------------------- 数据
# (星级, 是否已验证购买, 文本)
REVIEWS = [
    (5, True,  "battery life is great, lasts all day on one charge. waterproof works well in rain."),
    (5, True,  "sound is clear and loud enough for outdoor use, build feels solid and sturdy."),
    (4, True,  "good value for the price, easy to pair with my phone, bluetooth connects fast."),
    (5, True,  "love it, took it camping and it survived a drop into the river, still works."),
    (4, True,  "decent sound, battery lasts long, but the strap feels a bit cheap."),
    (2, True,  "battery drains fast after two months, now it barely holds charge for an hour."),
    (1, True,  "water leaked inside after one shower, speaker stopped working completely, total waste."),
    (1, True,  "bluetooth disconnects constantly, very annoying, returned it for a refund."),
    (2, True,  "sound is weak and has noise at high volume, not worth the money."),
    (3, True,  "it is okay, battery is short and charging is slow, but sound is acceptable."),
    (1, True,  "stopped charging after three weeks, contacted support, no reply, disappointed."),
    (2, True,  "waterproof claim is misleading, it is only splash resistant, not for shower."),
    (5, True,  "excellent little speaker, sturdy build, waterproof and loud, highly recommend."),
    (4, True,  "clear sound and long battery, easy to carry, good value."),
    (1, True,  "flimsy build, the strap broke on day one, cheap plastic, problem after problem."),
    (2, True,  "volume is loud but the bass is weak, not great for music."),
    (5, True,  "used it hiking for a week, battery lasted, waterproof held up, very happy."),
    (3, True,  "average speaker, battery is fine, sound is just okay, nothing amazing."),
    (1, True,  "dead on arrival, would not turn on, waste of money, refund requested."),
    (4, True,  "nice design, easy to use, bluetooth pairs fast, battery is decent."),
    # 未验证购买 + 极端正面（虚假评论的可疑特征）
    (5, False, "best speaker ever amazing perfect excellent love it highly recommend!!!"),
    (5, False, "amazing perfect excellent best product ever love love love!!!"),
    (5, False, "perfect perfect perfect great great amazing!!!"),
    (1, False, "worst product terrible awful broken bad bad bad!!!") ,
]


def tokenize(text):
    toks = re.findall(r"[a-z']+", text.lower())
    out = []
    for t in toks:
        t = t.replace("'", "")
        if t and t not in STOP and len(t) > 1:
            out.append(t)
    return out


# ---------------------------------------------------------------- A) 关键词
def top_terms_by_bucket(reviews, bucket_fn, k=8):
    """按桶算 TF-IDF。用「桶」而不是「文档」做单位，得到的是这一桶的典型用词。"""
    buckets = defaultdict(list)
    for r in reviews:
        buckets[bucket_fn(r)].append(tokenize(r[2]))

    # 文档频率按桶计
    N = len(buckets)
    df = Counter()
    for b, docs in buckets.items():
        seen = set()
        for d in docs:
            seen |= set(d)
        for t in seen:
            df[t] += 1

    result = {}
    for b, docs in buckets.items():
        tf = Counter(t for d in docs for t in d)
        scores = {}
        for t, c in tf.items():
            if c < 2:
                continue
            idf = math.log((N + 1) / (df[t] + 1)) + 1
            scores[t] = (1 + math.log(c)) * idf
        result[b] = sorted(scores.items(), key=lambda x: -x[1])[:k]
    return result


def demo_keywords():
    print("=" * 74)
    print("A) 关键词：按星级分桶，分别提取痛点词与卖点词")
    print("=" * 74)
    res = top_terms_by_bucket(REVIEWS, lambda r: "好评 4-5 星" if r[0] >= 4 else "差评 1-2 星")
    for bucket, terms in res.items():
        print(f"\n{bucket}（TF-IDF 前 8）：")
        print("  " + "，".join(f"{t}({s:.2f})" for t, s in terms))
    print("\n用法：好评词 → 标题与卖点语料；差评词 → 改进清单与详情页 FAQ。")


# ---------------------------------------------------------------- B) LDA
def lda_gibbs(docs, K=3, alpha=0.1, beta=0.01, iters=400, seed=42):
    """坍缩吉布斯采样 LDA。

    每个词先在当前文档-Dirichlet 与 主题-词 Dirichlet 下重采样一个主题，
    迭代足够多次后，主题-词分布就收敛到「哪几个词总是一起出现」。
    """
    rng = np.random.default_rng(seed)
    vocab = sorted({t for d in docs for t in d})
    wid = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)

    # 文档转 id
    D = [[wid[t] for t in d] for d in docs]
    M = len(D)

    z = [[rng.integers(K) for _ in w] for w in D]           # 每个词的主题
    ndk = np.zeros((M, K))                                  # 文档-主题计数
    nkw = np.zeros((K, V))                                  # 主题-词计数
    nk = np.zeros(K)                                        # 主题总词数

    for m, w in enumerate(D):
        for i, word in enumerate(w):
            k = z[m][i]
            ndk[m, k] += 1
            nkw[k, word] += 1
            nk[k] += 1

    for _ in range(iters):
        for m, w in enumerate(D):
            for i, word in enumerate(w):
                k_old = z[m][i]
                ndk[m, k_old] -= 1
                nkw[k_old, word] -= 1
                nk[k_old] -= 1

                p = (ndk[m] + alpha) * (nkw[:, word] + beta) / (nk + V * beta)
                p = p / p.sum()
                k_new = int(rng.choice(K, p=p))

                z[m][i] = k_new
                ndk[m, k_new] += 1
                nkw[k_new, word] += 1
                nk[k_new] += 1

    return vocab, nkw, ndk


def demo_lda():
    print("\n" + "=" * 74)
    print("B) 主题发现：LDA（吉布斯采样）自动找出评论里的几个主题")
    print("=" * 74)
    docs = [tokenize(r[2]) for r in REVIEWS]
    vocab, nkw, ndk = lda_gibbs(docs, K=3)

    print("\n三个主题的高频词：")
    for k in range(nkw.shape[0]):
        top = np.argsort(-nkw[k])[:7]
        words = "，".join(f"{vocab[i]}" for i in top if nkw[k, i] > 0)
        share = nkw[k].sum() / nkw.sum()
        print(f"  主题{k+1}（占 {share:.0%}）：{words}")

    print("\n主题占比（可用于判断买家最关心什么）：")
    doc_topic = (ndk + 0.1) / (ndk + 0.1).sum(axis=1, keepdims=True)
    dominant = doc_topic.argmax(axis=1)
    cnt = Counter(dominant)
    for k in range(nkw.shape[0]):
        print(f"  主题{k+1}：{cnt[k]} 条评论以此为主")

    print("\n局限：主题数是人工指定的；小样本时结果不稳定；主题词需要人给它命名。")


# ---------------------------------------------------------------- C) 情感
def sentiment(text):
    """词典法 + 否定词处理：否定词后的情感词极性翻转并衰减。"""
    toks = re.findall(r"[a-z']+", text.lower())
    toks = [t.replace("'", "") for t in toks]
    score, hits = 0.0, []
    for i, t in enumerate(toks):
        s = POS_WORDS.get(t, 0) + NEG_WORDS.get(t, 0)
        if s == 0:
            continue
        window = toks[max(0, i - 2):i]
        if any(w in NEGATORS for w in window):
            s = -s * 0.7
            hits.append(f"{t}(否定→{s:+.1f})")
        else:
            hits.append(f"{t}({s:+.1f})")
        score += s
    return score, hits


def demo_sentiment():
    print("\n" + "=" * 74)
    print("C) 情感：词典法（含否定处理）")
    print("=" * 74)
    rows = []
    for star, verified, text in REVIEWS[:12]:
        sc, hits = sentiment(text)
        pred = "正面" if sc > 0.5 else ("负面" if sc < -0.5 else "中性")
        rows.append((star, pred, round(sc, 2), text[:46], hits[:3]))

    print(f"{'星级':<4}{'判断':<6}{'得分':<7}{'文本'}")
    for star, pred, sc, t, _ in rows:
        print(f"{star:<4}{pred:<6}{sc:<7}{t}")

    # 与星级一致性
    agree = sum(
        1 for star, sc, in [(r[0], sentiment(r[2])[0]) for r in REVIEWS[:12]]
        if (star >= 4 and sc > 0) or (star <= 2 and sc < 0) or star == 3
    )
    print(f"\n与星级一致：{agree}/12")
    print("不一致的通常是反讽、双重否定或非母语表达——这正是需要预训练模型的原因。")
    print("实务做法：先按星级分桶，再在桶内做情感分析，误差会小很多。")


if __name__ == "__main__":
    demo_keywords()
    demo_lda()
    demo_sentiment()
