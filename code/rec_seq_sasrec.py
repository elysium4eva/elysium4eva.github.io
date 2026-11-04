#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rec_seq_sasrec.py -- 序列推荐中的因果自注意力（演示版）

用途：
    用「已知主题结构的序列」演示因果自注意力的计算与可解释性：
    - 只能看过去（因果掩码）
    - 注意力权重会集中在与目标同主题的历史步上
    并用基于 embedding 相似度的规则给出 HR@10 作为对照基线。

输入：
    合成序列（每个用户沿一个主题点击；主题结构已知，仅用于演示）
输出：
    HR@10、某个样本的注意力权重分布
依赖：
    numpy（可选 chartlib 出 SVG 图）
对应课件：
    模块五第三章 §5 算法 5-4 序列推荐（SASRec 与自注意力）

说明：
    本程序不训练神经网络，目的是把「因果掩码 + 注意力权重」这一步看清。
    真实 SASRec 用多层多头 Transformer + 训练得到；训练版见 RecBole 等库。
"""

import numpy as np

N_USER = 200
N_ITEM = 60
N_TOPIC = 5
SEQ_LEN = 10
DIM = 12
SEED = 5


def make_data(seed=SEED):
    """合成数据：物品有主题，用户沿主题点击。"""
    rng = np.random.default_rng(seed)
    topic = rng.integers(0, N_TOPIC, N_ITEM)
    seqs, targets = [], []
    for _ in range(N_USER):
        t = int(rng.integers(0, N_TOPIC))
        items = np.where(topic == t)[0]
        if len(items) < 3:
            items = np.arange(N_ITEM)
        s = [int(x) for x in rng.choice(items, size=SEQ_LEN + 1, replace=True)]
        # 注入 30% 的「跨主题噪声」，让注意力有区分空间
        for j in range(len(s) - 1):        # 不动最后一个（目标），否则「同主题」前提失效
            if rng.random() < 0.25:
                s[j] = int(rng.integers(0, N_ITEM))
        seqs.append(s[:-1])
        targets.append(s[-1])
    return seqs, targets, topic


def make_embedding(topic, seed=SEED):
    """物品 embedding：主题方向 + 小噪声（模拟训练好的向量）。"""
    rng = np.random.default_rng(seed + 1)
    E = np.zeros((N_ITEM, DIM))
    for i in range(N_ITEM):
        E[i, topic[i]] = 1.0
        E[i] += rng.normal(0, 0.15, DIM)
    # 归一化
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    return E


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def causal_attention(seq, E, Wq, Wk, Wv):
    """单头因果自注意力：返回表征 H 与注意力矩阵 A。"""
    X = E[seq]                          # (L, D)
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    S = Q @ K.T / np.sqrt(DIM)          # (L, L)
    mask = np.triu(np.ones_like(S), k=1).astype(bool)
    S = np.where(mask, -1e9, S)         # 因果掩码：不能看未来
    A = softmax(S, axis=-1)
    H = A @ V
    return H, A


def evaluate(seqs, targets, E, rank_k=10):
    """用「历史平均表征 + 内积」预测下一个物品，算 HR@K。"""
    hr, mrr = 0, 0.0
    for s, tgt in zip(seqs, targets):
        h = E[s].mean(axis=0)
        logits = E @ h
        logits[s] = -1e9                # 屏蔽已看过
        rank = int(np.sum(logits > logits[tgt]))
        if rank < rank_k:
            hr += 1
        mrr += 1.0 / (rank + 1)
    n = len(seqs)
    return hr / n, mrr / n


def demo():
    seqs, targets, topic = make_data()
    E = make_embedding(topic)
    rng = np.random.default_rng(SEED + 2)
    Wq, Wk, Wv = (rng.normal(0, 1, (DIM, DIM)) / np.sqrt(DIM) for _ in range(3))

    hr, mrr = evaluate(seqs, targets, E)
    print("=== 序列推荐：因果自注意力演示 ===")
    print(f"用户序列 {len(seqs)} | 物品 {N_ITEM} | 主题 {N_TOPIC} | 序列长度 {SEQ_LEN} | 维度 {DIM}")
    print(f"基于历史平均表征的 HR@10 {hr*100:.1f}%   MRR {mrr:.3f}   （随机基线 ≈ {(10/N_ITEM)*100:.1f}%）")

    # 取一个样本看注意力
    s = seqs[0]
    tgt = targets[0]
    H, A = causal_attention(s, E, Wq, Wk, Wv)
    w = A[-1]
    tgt_topic = topic[tgt]
    print(f"\n=== 样本：第 0 个用户，目标物品 {tgt}（主题 {tgt_topic}）===")
    same = 0
    for t, val in enumerate(w, 1):
        it = s[t - 1]
        mark = ' <同主题>' if topic[it] == tgt_topic else ''
        bar = '#' * int(val * 50)
        if topic[it] == tgt_topic:
            same += 1
        print(f"  第 {t:2d} 步 物品 {it:2d}（主题 {topic[it]}）：{val:5.3f} {bar}{mark}")

    print(f"\n{len(s)} 步历史中有 {same} 步与目标同主题。")
    ws = [w[i] for i, it in enumerate(s) if topic[it] == tgt_topic]
    wo = [w[i] for i, it in enumerate(s) if topic[it] != tgt_topic]
    print(f"同主题历史平均权重 {sum(ws)/max(len(ws),1):.3f} | "
          f"跨主题历史平均权重 {sum(wo)/max(len(wo),1):.3f}")
    print("\n结论有两条：")
    print("  1) 因果掩码保证「不能看未来」——这是结构约束，不需要靠训练获得。")
    print("  2) 未训练的注意力接近均匀（同主题与跨主题权重几乎一样），")
    print("     说明注意力不会自动变聪明，必须靠训练把「该看哪几步」学出来。")
    print("     这正是 SASRec 要在大规模序列上训练的原因。")
    return w, s, topic, tgt_topic, topic, hr, mrr


# ---------------------------------------------------------------- 可视化
def save_visual(path='visuals/rec_seq_sasrec.svg'):
    """输出一个样本的注意力权重条形图。"""
    from chartlib import BarChart
    w, s, topic, tgt_topic, _, hr, mrr = demo()
    labels = []
    for t, it in enumerate(s, 1):
        mark = '*' if topic[it] == tgt_topic else ''
        labels.append(f'{t}{mark}')
    ch = BarChart('最后位置对各历史步的注意力（* 为同主题）', '历史步', '注意力权重')
    ch.add_series('注意力', labels, [round(float(v), 3) for v in w])
    same_mean = float(np.mean([w[i] for i, it in enumerate(s) if topic[it] == tgt_topic]))
    other = [w[i] for i, it in enumerate(s) if topic[it] != tgt_topic]
    other_mean = float(np.mean(other)) if other else 0.0
    ch.add_note(f'HR@10 {hr*100:.1f}%；未训练时同主题权重 {same_mean:.3f} 与'
                f'跨主题 {other_mean:.3f} 几乎相同——所以必须训练')
    return ch.save(path)


if __name__ == '__main__':
    demo()
