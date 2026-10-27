#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
content_generate.py —— 内容生成三件套：模板化生成、提示词构造、三层校验

对应课件：模块五 第二章 §8「内容生成的主流算法」
          — 阶段一 模板化（规则 + 变量填充）
          — 阶段三/四 提示词工程（结构化输出 + 给禁区）
          — 多智能体三角色（撰写 / 审校 / 合规）

用途：
    A) 模板化生成：用「利益短语 + 证据句」的结构批量生成五点描述
    B) 提示词构造：把事实、结构、禁区、来源标注四件事组装成一段可直接用的提示词
    C) 三层校验：事实层（数字是否与规格一致）、合规层（禁用词）、风格层（是否利益前置）

为什么先讲模板再做提示词：
    模板把「结构」固定下来，大模型只负责组织语言——这是最稳的分工。
    直接让模型「写个详情页」而不给结构，输出通常通顺但无用；
    给了结构再让它填，产出可以直接进入校验环节。

依赖：无第三方依赖（纯标准库），可直接运行
运行：python content_generate.py

数据说明：BS-520 的参数为教学示例，不对应真实产品。
"""

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------- 输入事实
# 铁律：所有事实必须来自这份规格，模型不得补充
SPEC = {
    "model": "BS-520",
    "waterproof": "IPX7 (1 m / 30 min)",
    "battery": "5000mAh, 12 h playtime, 2.5 h charge",
    "bluetooth": "Bluetooth 5.3, up to 30 m",
    "weight": "210 g",
    "power": "10 W",
    "certifications": ["CE", "RoHS", "FCC"],
    "in_box": ["speaker x1", "USB-C cable x1", "strap x1", "manual x1"],
}

# 禁用词（合规层）
BANNED = ["best", "no.1", "no. 1", "100%", "lifetime", "cure", "guaranteed",
          "permanent", "perfect", "unbeatable"]

# 五点结构：每点承担一个任务（五问五答）
BULLET_PLAN = [
    ("TRULY WATERPROOF", "waterproof",
     "eliminates the #1 return reason in this category"),
    ("ALL-DAY PLAYTIME", "battery",
     "a full day outdoors without mid-hike charging"),
    ("STABLE CONNECTION", "bluetooth",
     "your phone can stay in your bag"),
    ("LIGHT & PACKABLE", "weight",
     "clips to a backpack without weighing it down"),
    ("READY TO USE", "in_box",
     "gift-ready, no extra accessories to buy"),
]


# ---------------------------------------------------------------- A) 模板化
@dataclass
class Bullet:
    headline: str
    body: str
    source_fields: list = field(default_factory=list)


def render_bullets(spec):
    """模板化：利益短语（大写）+ 冒号 + 事实作证据 + 一句「所以意味着什么」。"""
    out = []
    for headline, key, benefit in BULLET_PLAN:
        raw = spec[key]
        if isinstance(raw, list):
            evidence = ", ".join(raw)
        else:
            evidence = raw
        body = f"{evidence} — {benefit}"
        out.append(Bullet(headline=headline, body=body, source_fields=[key]))
    return out


def render_title(spec):
    """标题三段式：品牌/型号 + 核心品类词 + 关键属性 + 场景。"""
    return (f"{spec['model']} Bluetooth Speaker, IPX7 Waterproof, 12H Playtime, "
            f"210g Ultra-Light — for Camping & Shower")


# ---------------------------------------------------------------- B) 提示词
def build_prompt(spec, bullets, audience="outdoor campers and commuters"):
    """把四件事组装进提示词：事实、结构、禁区、来源标注。"""
    facts = "\n".join(f"  - {k}: {v}" for k, v in spec.items())
    struct = "\n".join(
        f"  {i+1}. {b.headline}: {b.body}" for i, b in enumerate(bullets))
    banned = ", ".join(BANNED)
    return f"""你是一名跨境电商文案。请严格按以下约束完成任务。

【事实】只能使用下面这些信息，不得补充或推测任何未提供的内容：
{facts}

【结构】输出 5 条五点描述，每条格式为「大写利益短语: 具体证据 — 对客户意味着什么」，
保持下面这个骨架的卖点顺序：
{struct}

【受众】{audience}。语言：英文，简洁，手机端每条不超过两行。

【禁区】不得出现以下表述（合规风险）：{banned}。
不得声称任何未列在【事实】中的认证、参数或承诺。

【标注】输出末尾附一行「来源核对」：逐条说明每个数字来自【事实】的哪个字段；
凡不是来自【事实】的内容，标注「来自模型，需人工核实」。
"""


# ---------------------------------------------------------------- C) 三层校验
def validate(bullets, title, spec):
    """三层校验：事实 / 合规 / 风格。返回问题清单。"""
    issues = []
    all_text = title + " " + " ".join(b.headline + " " + b.body for b in bullets)
    low = all_text.lower()

    # ---- 事实层：检查文本里出现的数字是否都能在 SPEC 里找到
    spec_blob = " ".join(str(v) for v in spec.values()).lower()
    numbers = set(re.findall(r"\d+(?:\.\d+)?", low))
    for n in sorted(numbers):
        if n not in spec_blob:
            issues.append(("事实", f"数字 {n} 未在规格中出现，疑似编造"))

    # ---- 事实层：认证名称核对
    for cert in re.findall(r"\b(CE|RoHS|FCC|UL|GS|PSE)\b", all_text):
        if cert not in spec["certifications"]:
            issues.append(("事实", f"认证 {cert} 未持有，属虚构"))

    # ---- 合规层：禁用词
    for w in BANNED:
        if w in low:
            issues.append(("合规", f"出现禁用词「{w}」"))

    # ---- 合规层：承诺类表述
    for pat, why in [
        (r"\bguarantee[ds]?\b", "不得承诺"),
        (r"\b(waterproof|lifetime)\s+warranty\b", "质保期未提供，不得声称"),
        (r"\bFDA\b|\bmedical\b|\bcure\b", "不得宣称医疗功效"),
    ]:
        if re.search(pat, low):
            issues.append(("合规", f"命中风险表述（{why}）"))

    # ---- 风格层：是否利益前置（大写短语开头）
    for i, b in enumerate(bullets, 1):
        if not b.headline.isupper():
            issues.append(("风格", f"第 {i} 点未用大写利益短语开头"))
        if "—" not in b.body and "-" not in b.body:
            issues.append(("风格", f"第 {i} 点缺少「所以意味着什么」"))

    return issues


def demo():
    print("=" * 78)
    print("A) 模板化生成")
    print("=" * 78)
    title = render_title(SPEC)
    bullets = render_bullets(SPEC)
    print(f"\n标题：{title}\n")
    for i, b in enumerate(bullets, 1):
        print(f"{i}. {b.headline}: {b.body}")

    print("\n" + "=" * 78)
    print("B) 提示词构造（这就是要喂给模型的完整输入）")
    print("=" * 78)
    prompt = build_prompt(SPEC, bullets)
    print()
    for line in prompt.strip().split("\n"):
        print("  " + line)

    print("\n" + "=" * 78)
    print("C) 三层校验")
    print("=" * 78)
    issues = validate(bullets, title, SPEC)
    if issues:
        for layer, msg in issues:
            print(f"  [{layer}] {msg}")
    else:
        print("  三层全部通过。")

    # 故意造一个不合格版本，演示校验能抓住什么
    print("\n" + "-" * 78)
    print("演示：把模型常见的三类错误塞进去，看校验能不能抓住")
    bad = [
        Bullet("BEST WATERPROOF", "IPX8 rated, 20 m depth, lifetime warranty — cures all water worries", ["waterproof"]),
        Bullet("ALL DAY", "8000mAh, 48 h playtime — never charge again", ["battery"]),
    ]
    bad_title = "BS-520 Speaker, 100% Waterproof, Best in Class, UL Certified"
    bad_issues = validate(bad, bad_title, SPEC)
    for layer, msg in bad_issues:
        print(f"  [{layer}] {msg}")
    print(f"\n共发现 {len(bad_issues)} 处问题——这正是「AI 起草 + 人工三层校验」的分工所在。")



# ---------------------------------------------------------------- 可视化
def save_visual(path='visuals/content_generate.svg'):
    """输出「三层校验抓出的问题数」分组柱状图（合格版 vs 造错版）。"""
    from collections import Counter
    from chartlib import BarChart
    good = validate(render_bullets(SPEC), render_title(SPEC), SPEC)
    bad = [
        Bullet('BEST WATERPROOF',
               'IPX8 rated, 20 m depth, lifetime warranty - cures all water worries',
               ['waterproof']),
        Bullet('ALL DAY', '8000mAh, 48 h playtime - never charge again', ['battery']),
    ]
    bad_title = 'BS-520 Speaker, 100% Waterproof, Best in Class, UL Certified'
    bad_issues = validate(bad, bad_title, SPEC)
    layers = ['事实', '合规', '风格']
    c1 = Counter(l for l, _ in good)
    c2 = Counter(l for l, _ in bad_issues)
    ch = BarChart('三层校验抓出的问题数', '校验层', '问题数')
    ch.add_series('合格版本', layers, [float(c1.get(l, 0)) for l in layers])
    ch.add_series('造错版本', layers, [float(c2.get(l, 0)) for l in layers])
    ch.add_note('合格版本应当三层全过；造错版本三类问题都被抓出')
    return ch.save(path)

if __name__ == "__main__":
    demo()
