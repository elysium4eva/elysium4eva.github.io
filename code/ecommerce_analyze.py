#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ecommerce_analyze.py —— 电商热销品数据的清洗与分析

重要前提（务必先读）：
    本脚本【不做任何网页抓取】。它的输入必须是【你已经合法获得】的数据文件：
      · 平台官方 API 返回的 JSON
      · 平台后台或第三方授权数据服务商导出的 CSV / Excel
      · 公开页面的低频人工核对结果
    未经授权绕过反爬抓取平台数据违反其服务条款，本课程不提供此类方法。

用途：
    读取上述文件，完成清洗、价格带分布、高口碑低竞争筛选、卖点词频统计，
    并导出可直接用于横向竞品分析的对比表。

准备：
    pip install pandas            # 可选：中文分词 pip install jieba

用法：
    python ecommerce_analyze.py --input raw/top_products.json
    python ecommerce_analyze.py --input raw/export.csv --lang auto
"""

import argparse
import json
import os
import re
from collections import Counter

import pandas as pd

CLEAN_DIR = "clean"
STOP_EN = {"with", "for", "and", "the", "a", "of", "to", "in", "on", "pro",
           "max", "new", "set", "pcs", "pack", "high", "quality", "type",
           "inch", "inches", "waterproof", "wireless"}   # 按品类自行增删


def parse_args():
    p = argparse.ArgumentParser(description="电商热销数据清洗与分析（输入须为合法获得的数据）")
    p.add_argument("--input", required=True, help="输入文件：JSON 或 CSV")
    p.add_argument("--title-field", default="title", help="标题字段名")
    p.add_argument("--price-field", default="price", help="价格字段名")
    p.add_argument("--rating-field", default="rating", help="评分字段名")
    p.add_argument("--review-field", default="reviews", help="评论数字段名")
    p.add_argument("--topn", type=int, default=15, help="词频统计取前 N 个词")
    return p.parse_args()


def load(path):
    """读取 JSON 或 CSV，并规范成一张表。字段名以实际文件为准。"""
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    items = raw.get("items", raw) if isinstance(raw, dict) else raw
    rows = []
    for it in items:
        rows.append({
            "sku":     it.get("asin") or it.get("id") or it.get("sku"),
            "title":   (it.get("title") or "")[:120],
            # 不同来源的价格字段可能是字典，这里做兼容处理
            "price":   float((it.get("price") or {}).get("value", 0)
                             if isinstance(it.get("price"), dict)
                             else (it.get("price") or 0)),
            "rating":  float(it.get("rating") or 0),
            "reviews": int(it.get("reviews") or 0),
            "rank":    int(it.get("bsr") or it.get("rank") or 0),
        })
    return pd.DataFrame(rows)


def clean(df, args):
    before = len(df)
    df = df.rename(columns={args.title_field: "title", args.price_field: "price",
                            args.rating_field: "rating", args.review_field: "reviews"})
    df = df[(df["price"] > 0) & (df["rating"] > 0)]
    print(f"清洗：{before} 行 -> {len(df)} 行（剔除价格或评分为空的记录）")
    return df


def price_band(df):
    bins = [0, 20, 40, 60, 100, 200, float("inf")]
    labels = ["<20", "20-40", "40-60", "60-100", "100-200", ">200"]
    df["band"] = pd.cut(df["price"], bins=bins, labels=labels)
    stat = df.groupby("band", observed=True).agg(
        竞品数=("sku", "count"), 均价=("price", "mean"),
        平均评分=("rating", "mean"), 平均评论数=("reviews", "mean")).round(2)
    print("\n=== 价格带分布 ===")
    print(stat.to_string())
    return df, stat


def niche(df):
    """低竞争（评论数低于中位数）且高口碑（评分 >= 4.3）的候选款。"""
    med = df["reviews"].median()
    cand = df[(df["reviews"] < med) & (df["rating"] >= 4.3)]
    print(f"\n=== 高口碑低竞争候选（评论数中位数 {med:.0f}）===")
    print(f"命中 {len(cand)} 款")
    if len(cand):
        print(cand[["sku", "price", "rating", "reviews"]]
              .sort_values("rating", ascending=False).head(10).to_string(index=False))
    hot = df[df["reviews"] >= df["reviews"].quantile(0.9)]
    if len(hot):
        print("\n对比：高竞争区（评论数前 10%）价格区间 "
              f"{hot['price'].min():.1f} - {hot['price'].max():.1f}")
    return cand


def keywords(df, topn):
    words = []
    for t in df["title"].fillna(""):
        words += [w.lower() for w in re.findall(r"[A-Za-z]{3,}", str(t))
                  if w.lower() not in STOP_EN]
    print(f"\n=== 标题卖点词频（前 {topn}）===")
    for w, c in Counter(words).most_common(topn):
        print(f"  {w:18s} {c:4d}")


def cn_keywords(df, topn):
    """中文标题/评论的粗分词：按 2-4 字连续汉字切分，仅作快速预览。"""
    try:
        import jieba                                            # noqa: F401
    except ImportError:
        print("\n（未安装 jieba，跳过中文词频统计：pip install jieba）")
        return
    import jieba
    words = []
    for t in df["title"].fillna(""):
        words += [w for w in jieba.lcut(str(t)) if len(w) >= 2 and re.match(r"[\u4e00-\u9fa5]+$", w)]
    print(f"\n=== 中文词频（前 {topn}）===")
    for w, c in Counter(words).most_common(topn):
        print(f"  {w:10s} {c:4d}")


def main():
    args = parse_args()
    os.makedirs(CLEAN_DIR, exist_ok=True)

    print(f"读取：{args.input}")
    df = load(args.input)
    df = clean(df, args)

    df, stat = price_band(df)
    niche(df)
    keywords(df, args.topn)
    cn_keywords(df, args.topn)

    out = df[["sku", "title", "price", "band", "rating", "reviews", "rank"]] \
        .sort_values("reviews", ascending=False)
    path = os.path.join(CLEAN_DIR, "competitor_matrix.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    stat.to_csv(os.path.join(CLEAN_DIR, "price_band.csv"), encoding="utf-8-sig")
    print(f"\n已导出：{path}（{len(out)} 条）")
    print("下一步：把这张表交给横向竞品分析（课件第 7 节）做聚类与主题归纳。")


if __name__ == "__main__":
    main()
