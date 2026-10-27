#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sel_semantic.py —— 选品的语义检索：TF-IDF + 余弦相似度（手写实现）

对应课件：模块五 第二章 §4「智能选品的典型算法」— 语义匹配族
          对应第一章提到的 ACL 2026 BLaIR 基准：商品文本可作语义编码输入

用途：
    给定一句「买家用的话」（可能是复杂查询，比如
    “露营用的防水音箱 30 美元以内”），从商品库里找出语义最接近的商品。

为什么不用关键词精确匹配：
    买家写的是场景与需求，商品标题写的是品类与参数，字面往往对不上。
    向量化之后比较的是「用词的分布是否相似」，能缓解这种词不匹配问题。

实现说明（教学版，纯 numpy）：
    真实系统用稠密向量（embedding 模型）做语义编码；这里用 TF-IDF 稀疏向量，
    原理一致（都是把文本映射到向量空间再算相似度），但不需要下载模型、离线可跑。
    课件里会说明两者的差别与升级路径。

依赖：numpy（本课程环境自带）
运行：python sel_semantic.py
"""

import math
import re
from collections import Counter

import numpy as np

# ---------------------------------------------------------------- 商品库
# 每条：标题 + 五点节选（真实系统会把评论、类目、属性一起编码）
DOCS = {
    "BS-520 防水蓝牙音箱": "bluetooth speaker waterproof IPX7 outdoor camping shower 12h battery portable lightweight",
    "BS-300 迷你蓝牙音箱": "bluetooth speaker mini portable small travel 6h battery light",
    "PB-20 户外电源": "portable power station outdoor camping battery 500W solar charging",
    "CH-10 车载支架": "car phone holder mount dashboard magnetic",
    "CH-30 车载快充": "car charger fast charging USB-C 65W",
    "FS-01 便携风扇": "portable fan rechargeable handheld cooling outdoor",
    "WC-05 无线充电板": "wireless charger pad 15W qi compatible",
    "CB-01 收纳包": "storage case pouch for speaker and cable travel",
}

# 查询：故意用「场景 + 属性」的复杂说法，而不是品类词
QUERIES = [
    "waterproof speaker for camping under 30 dollars",
    "something to charge my phone fast in the car",
    "portable cooling device for hot weather outdoor",
    "case to carry my speaker while traveling",
]

# 极简停用词（只去掉对区分类别无帮助的高频词）
STOP = {
    "a", "an", "the", "for", "to", "my", "in", "on", "of", "and", "with",
    "under", "dollars", "something", "while", "is", "it", "this", "that",
}


def tokenize(text):
    """小写化 → 提取英文词与数字 → 去停用词。"""
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in toks if t not in STOP and len(t) > 1]


def build_tfidf(docs, queries):
    """把商品与查询放进同一个词表，算 TF-IDF 矩阵。

    TF   = 词频（这里用 1 + log(tf) 做次线性缩放）
    IDF  = log(N / df)，df 为包含该词的文档数（含查询，保持同一空间）
    """
    corpus = [tokenize(d) for d in docs.values()] + [tokenize(q) for q in queries]
    vocab = sorted({t for doc in corpus for t in doc})
    wid = {t: i for i, t in enumerate(vocab)}

    N = len(corpus)
    df = np.zeros(len(vocab))
    for doc in corpus:
        for t in set(doc):
            df[wid[t]] += 1
    idf = np.log((N + 1) / (df + 1)) + 1.0   # 平滑，避免除零

    def vec(tokens):
        tf = Counter(tokens)
        v = np.zeros(len(vocab))
        for t, c in tf.items():
            v[wid[t]] = (1 + math.log(c)) * idf[wid[t]]
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    doc_mat = np.vstack([vec(tokenize(d)) for d in docs.values()])
    q_mat = np.vstack([vec(tokenize(q)) for q in queries])
    return vocab, doc_mat, q_mat, idf


def explain(vocab, idf, query):
    """打印查询里权重最高的词，让「为什么匹配到这条」可解释。"""
    toks = tokenize(query)
    tf = Counter(toks)
    scored = sorted(
        ((t, (1 + math.log(tf[t])) * idf[i]) for t, i in
         [(t, -1) for t in tf] if t in vocab),
        key=lambda x: -x[1],
    )
    # 上面那句写法过绕，改为直接查表
    wid = {t: i for i, t in enumerate(vocab)}
    scored = sorted(
        ((t, (1 + math.log(c)) * idf[wid[t]]) for t, c in tf.items() if t in wid),
        key=lambda x: -x[1],
    )
    return scored


def main():
    print("=" * 74)
    print("语义检索：用「买家的说法」找商品（TF-IDF + 余弦相似度）")
    print("=" * 74)

    vocab, D, Q, idf = build_tfidf(DOCS, QUERIES)
    names = list(DOCS.keys())

    for qi, q in enumerate(QUERIES):
        sims = D @ Q[qi]                      # 余弦相似度（向量已归一化）
        order = np.argsort(-sims)[:3]
        print(f"\n查询：{q}")
        scored = explain(vocab, idf, q)
        top_terms = "，".join(f"{t}({s:.2f})" for t, s in scored[:5] if s > 0)
        print(f"  查询里的高权重词：{top_terms or '（无有效词）'}")
        print("  最相似商品：")
        for k in order:
            if sims[k] <= 0:
                continue
            bar = "█" * max(1, int(sims[k] * 30))
            print(f"    {sims[k]:.3f} {bar:<30} {names[k]}")

    # 展示「词不匹配」为什么需要语义方法
    print("\n" + "-" * 74)
    print("为什么不能只靠字面匹配：")
    print("  查询 'cooling device' 与商品 'portable fan ... cooling outdoor' 只共享 cooling 一词，")
    print("  但语义上高度相关。TF-IDF 靠共享词，稠密向量靠语义——课件会对比两者的升级路径。")

    # 简单的一致性演示：同一查询换成同义词，看排序是否稳定
    print("\n稳健性检查：把 'camping' 换成 'hiking' 后，首位是否仍然相同？")
    alt = ["waterproof speaker for hiking under 30 dollars"]
    # 关键：原查询与替换查询必须放进【同一个词表】，否则向量维度不同无法比较
    _, D2, Q2, _ = build_tfidf(DOCS, QUERIES + alt)
    a = names[int(np.argmax(D2 @ Q2[0]))]
    b = names[int(np.argmax(D2 @ Q2[len(QUERIES)]))]
    print(f"  原查询首位：{a}")
    print(f"  换词后首位：{b}")
    print(f"  结论：{'一致，说明区分主要靠 waterproof/speaker' if a == b else '不一致，说明模型对该词依赖过强'}")

    print("\n升级路径（课件会展开）：")
    print("  1) TF-IDF（本程序）—— 靠共享词，快、可解释、无依赖")
    print("  2) 稠密 embedding（如 BGE / OpenAI embedding）—— 靠语义，能处理同义改写")
    print("  3) 任务专用基准评测（如 ACL 2026 BLaIR）—— 选模型必须用自己的品类数据实测")



# ---------------------------------------------------------------- 可视化
def save_visual(path='visuals/sel_semantic.svg'):
    """输出第一个查询的 Top-5 相似度条形图。"""
    from chartlib import HBarChart
    vocab, D, Q, idf = build_tfidf(DOCS, QUERIES)
    sims = D @ Q[0]
    pairs = sorted(zip(list(DOCS.keys()), [float(v) for v in sims]), key=lambda x: -x[1])[:5]
    ch = HBarChart('语义检索 Top-5（查询：waterproof speaker for camping）', '余弦相似度')
    ch.add([p[0].split()[0] for p in pairs], [p[1] for p in pairs])
    return ch.save(path)

if __name__ == "__main__":
    main()
