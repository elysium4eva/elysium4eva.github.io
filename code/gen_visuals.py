#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_visuals.py —— 运行各程序的可视化函数，生成课程页面用的 SVG 图

用法：python gen_visuals.py
输出：visuals/*.svg（8 张），文件名与程序同名
"""

import importlib
import os
import sys
import traceback

TARGETS = [
    ("sel_scorecard", "sel_scorecard.py"),
    ("sel_rules_cluster", "sel_rules_cluster.py"),
    ("sel_bsr_forecast", "sel_bsr_forecast.py"),
    ("sel_semantic", "sel_semantic.py"),
    ("rev_text_analysis", "rev_text_analysis.py"),
    ("rev_absa_fake", "rev_absa_fake.py"),
    ("content_generate", "content_generate.py"),
    ("content_eval", "content_eval.py"),
    ("ecommerce_analyze", "ecommerce_analyze.py"),
]


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visuals")
    os.makedirs(out, exist_ok=True)
    ok, fail = 0, 0
    for mod_name, fname in TARGETS:
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, "save_visual", None)
            if fn is None:
                print(f"  {fname:<28} 跳过（无 save_visual）")
                continue
            path = fn(os.path.join(out, mod_name + ".svg"))
            size = os.path.getsize(path)
            print(f"  {fname:<28} OK  {size/1024:.1f} KB")
            ok += 1
        except Exception as e:
            fail += 1
            print(f"  {fname:<28} FAIL  {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
    print(f"\n生成 {ok} 张，失败 {fail} 张 → {out}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
