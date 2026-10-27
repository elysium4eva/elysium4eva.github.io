#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rev_absa_fake.py —— 评论两件事：方面级情感（ABSA）+ 虚假评论识别

对应课件：模块五 第二章 §6「评论分析的主要算法」
          — 方面级情感分析（ABSA：抽取式 / 生成式）
          — 虚假评论检测（特征工程 + GBDT / 图神经网络 / LLM）

用途：
    A) ABSA：把「这条评论好不好」拆成「它在哪些方面好、哪些方面差」，
       输出「方面 × 情感 × 提及次数」三维表——这就是痛点矩阵的数据底稿。
    B) 虚假评论：用可解释的特征给每条评论打可疑分，而不是黑箱判定。
       特征包括：星级与文本情感不一致、未验证购买、文本过短、标点异常、
       近似重复、时间聚集。

为什么用规则而不是上模型：
    虚假评论检测在实务中必须「可解释」——你要能对卖家说清「这条为什么可疑」。
    特征工程的每条特征都能解释；模型只是把它们加权的工具。
    本程序用规则模拟「特征 + 权重」的逻辑，换成 GBDT 时特征不变。

依赖：numpy（本课程环境自带）
运行：python rev_absa_fake.py

数据说明：评论为教学用合成数据，日期与账号信息为演示编造。
"""

import re
from collections import Counter, defaultdict
from itertools import combinations

# ---------------------------------------------------------------- 方面词典
# 方面 → 触发词。真实系统用 BERT 做方面词抽取；这里用词典，可解释、可修改。
ASPECTS = {
    "续航": ["battery", "charge", "charging", "drain", "drains", "lasts", "life"],
    "防水": ["waterproof", "water", "leaked", "leak", "rain", "shower", "splash"],
    "音质": ["sound", "bass", "volume", "loud", "noise", "clear", "audio"],
    "连接": ["bluetooth", "connect", "connects", "disconnects", "pair", "pairs"],
    "做工": ["build", "sturdy", "flimsy", "plastic", "strap", "broke", "solid", "cheap"],
    "充电": ["charging", "charge", "cable", "port", "usb"],
}

ASPECT_POS = {
    "good": 1, "great": 1, "excellent": 1.5, "perfect": 1.5, "love": 1.5,
    "solid": 1, "sturdy": 1, "clear": 1, "loud": 1, "long": 0.8, "works": 1,
    "lasts": 1, "survived": 1, "happy": 1, "durable": 1, "fast": 0.8, "decent": 0.6,
}
ASPECT_NEG = {
    "weak": -1, "broken": -1.5, "broke": -1.2, "leaked": -1.5, "leak": -1.5,
    "drain": -1.2, "drains": -1.2, "disconnects": -1.2, "flimsy": -1, "cheap": -0.8,
    "dead": -1.5, "stopped": -1.2, "noise": -0.8, "short": -0.8, "slow": -1,
    "misleading": -1.5, "problem": -1, "waste": -1.5, "simple": -0.3,
}
NEGATORS = {"not", "no", "never", "barely", "hardly", "without", "isnt", "wasnt"}


# (id, 星级, 是否已验证, 日期, 文本)
REVIEWS = [
    ("R01", 5, True,  "2026-03-02", "battery life is great, lasts all day. waterproof works in rain."),
    ("R02", 5, True,  "2026-03-04", "sound is clear and loud, build feels sturdy and solid."),
    ("R03", 4, True,  "2026-03-05", "good value, easy to pair, bluetooth connects fast."),
    ("R04", 2, True,  "2026-03-09", "battery drains fast after two months, barely holds charge."),
    ("R05", 1, True,  "2026-03-11", "water leaked inside after one shower, speaker stopped working."),
    ("R06", 1, True,  "2026-03-12", "bluetooth disconnects constantly, very annoying."),
    ("R07", 2, True,  "2026-03-15", "sound is weak and has noise at high volume."),
    ("R08", 3, True,  "2026-03-18", "battery is short and charging is slow, sound is okay."),
    ("R09", 1, True,  "2026-03-20", "flimsy build, the strap broke on day one, cheap plastic."),
    ("R10", 5, True,  "2026-03-22", "used it hiking, battery lasted, waterproof held up."),
    ("R11", 4, True,  "2026-03-25", "nice design, easy to use, bluetooth pairs fast."),
    ("R12", 2, True,  "2026-03-26", "waterproof claim is misleading, only splash resistant."),
    # 可疑：未验证 + 极端正面 + 极短 + 感叹号密集 + 近似重复（集中在两天内）
    ("F01", 5, False, "2026-04-01", "best speaker ever amazing perfect excellent love it!!!"),
    ("F02", 5, False, "2026-04-01", "amazing perfect excellent best product ever love love!!!"),
    ("F03", 5, False, "2026-04-02", "perfect perfect perfect great great amazing!!!"),
    ("F04", 5, False, "2026-04-02", "best ever amazing perfect love love love!!!"),
    # 可疑：未验证 + 极端负面（疑似恶意差评）
    ("F05", 1, False, "2026-04-02", "worst product terrible awful broken bad bad bad!!!"),
]


def tokens(text):
    t = re.findall(r"[a-z']+", text.lower())
    return [x.replace("'", "") for x in t]


def split_sentences(text):
    return [s.strip() for s in re.split(r"[.,;!?]", text) if s.strip()]


# ---------------------------------------------------------------- A) ABSA
def aspect_sentiment(text):
    """句级：先找句子里的方面词，再在句子内找情感词并判定极性。

    抽取式 ABSA 的简化版：方面由词典命中，情感由句内情感词加权求和。
    真实系统会把（方面词, 情感词）作为序列标注任务训练。
    """
    found = defaultdict(list)
    for sent in split_sentences(text):
        toks = tokens(sent)
        hits = []
        for asp, kws in ASPECTS.items():
            if any(k in toks for k in kws):
                hits.append(asp)
        if not hits:
            continue
        # 句内情感
        score = 0.0
        for i, t in enumerate(toks):
            s = ASPECT_POS.get(t, 0) + ASPECT_NEG.get(t, 0)
            if s == 0:
                continue
            if any(w in NEGATORS for w in toks[max(0, i - 2):i]):
                s = -s * 0.7
            score += s
        for asp in hits:
            found[asp].append((round(score, 2), sent))
    return dict(found)


def demo_absa():
    print("=" * 78)
    print("A) 方面级情感（ABSA）：把「好不好」拆成「哪方面好、哪方面差」")
    print("=" * 78)

    table = defaultdict(lambda: {"正": 0, "负": 0, "中": 0, "句子": []})
    for rid, star, verified, date, text in REVIEWS:
        if rid.startswith("F"):
            continue
        for asp, items in aspect_sentiment(text).items():
            for score, sent in items:
                lab = "正" if score > 0.4 else ("负" if score < -0.4 else "中")
                table[asp][lab] += 1
                table[asp]["句子"].append((rid, star, score, sent))

    print(f"\n{'方面':<6}{'提及':<6}{'正':<5}{'负':<5}{'中':<5}{'负向占比':<10}{'改进优先级'}")
    rows = []
    for asp, d in table.items():
        total = d["正"] + d["负"] + d["中"]
        neg_ratio = d["负"] / total if total else 0
        priority = total * neg_ratio
        rows.append((asp, total, d["正"], d["负"], d["中"], neg_ratio, priority))

    for asp, total, pos, neg, mid, ratio, pri in sorted(rows, key=lambda x: -x[6]):
        bar = "#" * max(0, int(pri * 4))
        print(f"{asp:<6}{total:<6}{pos:<5}{neg:<5}{mid:<5}{ratio:<10.0%}{pri:>6.1f} {bar}")

    print("\n改进优先级 = 提及次数 × 负向占比（课件里的公式）")
    print("读法：优先级高的方面，就是下一版产品最该改的地方。")
    top = max(rows, key=lambda x: x[6])
    print(f"\n本例最该改的是「{top[0]}」——找几条对应的差评原句：")
    for rid, star, score, sent in table[top[0]]["句子"]:
        if score < -0.4:
            print(f"  [{rid} {star}星 {score:+.1f}] {sent}")


# ---------------------------------------------------------------- B) 虚假检测
def char_bigrams(s):
    s = re.sub(r"[^a-z ]", "", s.lower())
    return set(s[i:i + 2] for i in range(len(s) - 1))


def jaccard(a, b):
    A, B = char_bigrams(a), char_bigrams(b)
    return len(A & B) / len(A | B) if (A | B) else 0.0


def fake_features(reviews):
    """为每条评论算可解释特征。每个特征对应一句人话解释。"""
    # 时间聚集：同一天评论数
    day_count = Counter(r[3] for r in reviews)
    # 近似重复：与另一条的字符二元组 Jaccard > 0.6
    dup_pairs = set()
    for (i, a), (j, b) in combinations(enumerate(reviews), 2):
        if jaccard(a[4], b[4]) > 0.6:
            dup_pairs.add(a[0]); dup_pairs.add(b[0])

    feats = {}
    for rid, star, verified, date, text in reviews:
        toks = tokens(text)
        n = len(toks)
        exclaims = text.count("!")
        pos = sum(ASPECT_POS.get(t, 0) for t in toks)
        neg = -sum(ASPECT_NEG.get(t, 0) for t in toks)
        text_pol = "正" if pos > neg else ("负" if neg > pos else "中")

        f = {}
        # 1) 星级与文本极性不一致
        if (star >= 4 and text_pol == "负") or (star <= 2 and text_pol == "正"):
            f["星级与文本情感矛盾"] = 25
        # 2) 未验证购买
        if not verified:
            f["未验证购买"] = 15
        # 3) 文本过短（阈值刻意设低：本演示数据的正常评论都在 8 词以上，
        #    现实中该阈值应按品类语料的长度分布来定，而不是拍脑袋）
        if n < 6:
            f["文本过短（信息量低）"] = 12
        # 4) 标点异常
        if exclaims >= 3:
            f["感叹号密集"] = 15
        # 5) 近义重复
        if rid in dup_pairs:
            f["与另一条评论高度相似"] = 30
        # 6) 时间聚集
        if day_count[date] >= 3 and not verified:
            f["同一天集中出现（未验证）"] = 20
        # 7) 极端星级 + 极短
        if star in (1, 5) and n < 6:
            f["极端星级 + 极短文本"] = 10

        feats[rid] = (f, sum(f.values()))
    return feats


def demo_fake():
    print("\n" + "=" * 78)
    print("B) 虚假评论识别：可解释特征打分")
    print("=" * 78)

    feats = fake_features(REVIEWS)
    print(f"\n{'评论':<6}{'星级':<6}{'验证':<6}{'可疑分':<8}{'命中特征'}")
    flagged = []
    for rid, star, verified, date, text in REVIEWS:
        f, score = feats[rid]
        mark = " ← 可疑" if score >= 30 else ""
        names = "；".join(f"{k}(+{v})" for k, v in f.items()) or "无"
        print(f"{rid:<6}{star:<6}{'是' if verified else '否':<6}{score:<8}{names}{mark}")
        if score >= 30:
            flagged.append((rid, score))

    print(f"\n判定为可疑：{len(flagged)} 条 —— " + "，".join(f"{r}({s})" for r, s in flagged))
    print("\n重要：")
    print("  · 这些是「可疑信号」，不是「定罪证据」。真实用户也可能写短评、也可能激动。")
    print("  · 实务处理：把可疑评论【降权】用于统计，而不是删除——避免误伤真实用户。")
    print("  · 换成 GBDT / 图神经网络时，输入仍是这些特征（加上账号关系、IP 等），只是权重由模型学。")


if __name__ == "__main__":
    demo_absa()
    demo_fake()
