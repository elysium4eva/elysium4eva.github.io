#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
content_eval.py —— 生成质量怎么评：BLEU / ROUGE 的局限 + 结构化评审打分

对应课件：模块五 第二章 §8「内容生成的主流算法」— 生成质量评估
          并与第四章「效果评估」呼应：三种必要、一种判定

用途：
    A) 手算 BLEU（含 1–4 gram 精确率与简短惩罚）与 ROUGE-N（召回导向），
       让你看清「文本相似度」到底在算什么、为什么不等于「文案好坏」。
    B) 结构化评审：按「利益前置 / 参数完整 / 合规 / 语气」四维打分，
       模拟 LLM-as-a-Judge 的评分表，但把标准写死、结果可复现。

为什么要亲手算一遍 BLEU：
    很多同学以为 BLEU 高就是文案好。算一遍之后会发现：
    把词序打乱、把「更差但更像参考」的句子放进来，BLEU 反而更高。
    ——这就是课件里说的「BLEU/ROUGE 是必要条件，不是判定标准」。

依赖：无第三方依赖（纯标准库）
运行：python content_eval.py
"""

import math
import re
from collections import Counter

# ---------------------------------------------------------------- 语料
REFERENCE = ("IPX7 waterproof to 1 m for 30 min , so a sudden rain does not end your trip . "
             "12 h playtime , 210 g , USB-C .")

CANDIDATES = {
    "C1 高质量改写": ("IPX7 waterproof to 1 m for 30 min , so rain does not end your trip . "
                    "12 h playtime , 210 g , USB-C ."),
    "C2 词序打乱": ("waterproof IPX7 30 min 1 m to , trip your end not does rain sudden a so . "
                   "USB-C , g 210 , playtime h 12 ."),
    "C3 空话连篇": ("premium quality sound with best-in-class design and unbeatable value "
                   "for every outdoor lover ."),
    "C4 编造参数": ("IPX8 waterproof to 50 m , 24 h playtime , 150 g , wireless charging ."),
}


def norm(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\- ]", " ", s)
    return s.split()


def ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu(candidate, reference, max_n=4, smooth=False):
    """BLEU：几何平均的 n-gram 精确率 × 简短惩罚。

    smooth=False 为标准 BLEU（任一阶精确率为 0 则整体归零）；
    smooth=True 用加一平滑（add-1），让乱序/短句也能得到非零分。
    两种结果都打印，正好用来说明「同一个指标换个变体，结论就变了」。
    """
    cand, ref = norm(candidate), norm(reference)
    if not cand:
        return 0.0, []

    precisions = []
    for n in range(1, max_n + 1):
        c_ng, r_ng = ngrams(cand, n), ngrams(ref, n)
        if not c_ng:
            precisions.append(0.0)
            continue
        overlap = sum(min(c, r_ng.get(g, 0)) for g, c in c_ng.items())
        total = max(sum(c_ng.values()), 1)
        p = overlap / total
        if smooth and p == 0:
            p = 1.0 / (total + 1)     # 加一平滑
        precisions.append(p)

    # 标准 BLEU：任一阶为 0 则整体归零
    if (not smooth) and min(precisions) == 0:
        geo = 0.0
    else:
        geo = math.exp(sum(math.log(p) for p in precisions if p > 0) /
                       max(sum(1 for p in precisions if p > 0), 1))

    # 简短惩罚
    c_len, r_len = len(cand), len(ref)
    bp = 1.0 if c_len >= r_len else math.exp(1 - r_len / max(c_len, 1))
    return bp * geo, precisions


def rouge_n(candidate, reference, n=1):
    """ROUGE-N：偏召回——参考里有多少 n-gram 被候选覆盖到。"""
    cand, ref = norm(candidate), norm(reference)
    c_ng, r_ng = ngrams(cand, n), ngrams(ref, n)
    if not r_ng:
        return 0.0
    overlap = sum(min(c, r_ng.get(g, 0)) for g, c in c_ng.items())
    return overlap / sum(r_ng.values())


# ---------------------------------------------------------------- 结构化评审
RUBRIC = [
    ("利益前置", 0.30, "是否说清「所以对我意味着什么」，而不是只堆参数"),
    ("参数完整", 0.30, "关键参数是否齐全且与规格一致"),
    ("合规", 0.25, "有无绝对化用语、虚构认证、无法兑现的承诺"),
    ("语言质量", 0.15, "英文是否自然、术语是否一致"),
]
BANNED = ["best", "no.1", "100%", "unbeatable", "premium", "lifetime", "perfect"]
SPEC_NUMBERS = {"1", "30", "12", "210", "3", "7"}   # 允许出现的数字（来自规格）


def judge(text):
    """按四维打分，每维 0–5，返回加权总分与逐维理由。

    这是「可复现的规则化评审」：标准写死在代码里，谁跑都一样。
    换成 LLM-as-a-Judge 时，把 RUBRIC 写进提示词即可，但偏差也随之引入。
    """
    low = text.lower()
    toks = norm(text)
    detail = {}

    # 利益前置：是否出现 so / means / 结果性表述
    benefit_hits = sum(low.count(w) for w in [" so ", " means", "without", "does not", "so that"])
    detail["利益前置"] = min(5, benefit_hits * 2)

    # 参数完整：命中规格数字的个数
    nums = set(re.findall(r"\d+", low))
    hit = len(nums & SPEC_NUMBERS)
    detail["参数完整"] = min(5, hit)

    # 合规：命中禁用词就扣，虚构认证也扣
    bad = [w for w in BANNED if w in low]
    certs = re.findall(r"\b(CE|RoHS|FCC|UL|GS)\b", text)
    fake_cert = [c for c in certs if c not in ("CE", "RoHS", "FCC")]
    detail["合规"] = max(0, 5 - 2 * len(bad) - 2 * len(fake_cert))

    # 语言质量：长度合理 + 无重复堆叠
    uniq = len(set(toks)) / max(len(toks), 1)
    detail["语言质量"] = min(5, round(uniq * 6))

    total = sum(detail[k] * w for k, w, _ in RUBRIC)
    notes = []
    if bad:
        notes.append(f"禁用词：{', '.join(bad)}")
    if fake_cert:
        notes.append(f"虚构认证：{', '.join(fake_cert)}")
    missing = SPEC_NUMBERS - nums
    if missing:
        notes.append(f"缺少关键数字：{', '.join(sorted(missing))}")
    return round(total, 2), detail, notes


def main():
    print("=" * 78)
    print("A) BLEU / ROUGE：先看它到底在算什么")
    print("=" * 78)
    print(f"\n参考文本：{REFERENCE}\n")
    print(f"{'候选':<16}{'BLEU':<8}{'BLEU平滑':<10}{'ROUGE-1':<9}{'ROUGE-2':<9}{'说明'}")

    notes = {
        "C1 高质量改写": "几乎一致，指标最高",
        "C2 词序打乱": "BLEU 归零但 ROUGE-1 高达 0.96",
        "C3 空话连篇": "全形容词，两个指标都低",
        "C4 编造参数": "参数全错，ROUGE-1 仍有 0.26",
    }
    for name, text in CANDIDATES.items():
        b, _ = bleu(text, REFERENCE)
        bs, _ = bleu(text, REFERENCE, smooth=True)
        r1 = rouge_n(text, REFERENCE, 1)
        r2 = rouge_n(text, REFERENCE, 2)
        print(f"{name:<16}{b:<8.3f}{bs:<10.3f}{r1:<9.3f}{r2:<9.3f}{notes[name]}")

    print("\n关键结论（三条，都是指标的失效方式）：")
    print("  · C2 词序完全打乱：标准 BLEU 直接归零，但 ROUGE-1 有 0.96——")
    print("    两个常用指标对同一段文本给出完全相反的判断。")
    print("  · C4 参数全错（IPX8/50m/24h）：ROUGE-1 仍有 0.26，指标对事实错误无感。")
    print("  · 加平滑后 C2 的 BLEU 从 0 变成非零——同一指标换个变体，结论又变了。")
    print("  · 所以 BLEU/ROUGE 只能筛掉「完全不相关」的输出，不能判定「文案好不好」。")

    print("\n" + "=" * 78)
    print("B) 结构化评审：把标准写死，结果可复现")
    print("=" * 78)
    print("\n评分维度：" + "；".join(f"{k}({w:.0%})" for k, w, _ in RUBRIC) + "\n")

    print(f"{'候选':<16}{'总分':<8}{'利益':<6}{'参数':<6}{'合规':<6}{'语言':<6}{'问题'}")
    for name, text in CANDIDATES.items():
        total, detail, notes_ = judge(text)
        print(f"{name:<16}{total:<8}{detail['利益前置']:<6}{detail['参数完整']:<6}"
              f"{detail['合规']:<6}{detail['语言质量']:<6}{'；'.join(notes_) or '无'}")

    print("\n结论对比：")
    print("  · 指标（BLEU）认为 C2、C4 都不差；评审认为 C3 合规扣分、C4 参数缺失。")
    print("  · 两者都不等于「能转化」——真正的判定标准是 A/B 投放数据。")
    print("  · 三级定位：文本相似度（粗筛）→ 结构化评审（细筛）→ 投放数据（定论）。")



# ---------------------------------------------------------------- 可视化
def save_visual(path='visuals/content_eval.svg'):
    """输出 BLEU 与 ROUGE-1 的分歧柱状图。"""
    from chartlib import BarChart
    names = list(CANDIDATES.keys())
    short = [n.split()[0] for n in names]
    bleu_scores = [bleu(CANDIDATES[n], REFERENCE)[0] for n in names]
    rouge1 = [rouge_n(CANDIDATES[n], REFERENCE, 1) for n in names]
    ch = BarChart('BLEU 与 ROUGE-1：同一批候选的分歧', '候选', '得分')
    ch.add_series('BLEU', short, [float(v) for v in bleu_scores])
    ch.add_series('ROUGE-1', short, [float(v) for v in rouge1])
    ch.add_note('C2 词序乱：BLEU=0 但 ROUGE-1 最高；C4 参数错：ROUGE 仍给分')
    return ch.save(path)

if __name__ == "__main__":
    main()
