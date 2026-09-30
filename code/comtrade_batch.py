#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comtrade_batch.py —— 用 UN Comtrade API 批量拉取海关贸易数据

用途：
    按「报告国 × 年份 × HS 编码」批量查询进口/出口数据，自动限速、断点续跑，
    并输出「原始返回 + 清洗后表格 + 运行日志」三份文件。

准备：
    pip install requests pandas

用法示例：
    # 1) 先小规模试跑（预览接口，无需密钥）
    python comtrade_batch.py --reporter 276 --years 2023 --codes 8518

    # 2) 正式批量（使用订阅密钥，参数写在 --key 或环境变量 COMTRADE_KEY）
    set COMTRADE_KEY=你的订阅密钥        # Windows
    export COMTRADE_KEY=你的订阅密钥     # macOS / Linux
    python comtrade_batch.py --reporter 276,392,410 --years 2022,2023,2024 \
                             --codes 8518,8504 --flow M

说明：
    本脚本只读取公开的官方统计接口，请遵守数据源的使用条款与配额限制。
    完整参数说明见课程主页「实操教程」栏目。
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
import requests

# ---------------------------------------------------------------- 配置
PREVIEW_API = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"
FULL_API = "https://comtradeapi.un.org/data/v1/get/C/A/HS"

RAW_DIR, CLEAN_DIR = "raw", "clean"
STATE_FILE = "raw/_state.json"          # 断点续跑：记录已完成的参数组合
SLEEP_SEC = 1.2                          # 每次请求间隔，避免触发配额限制
MAX_RETRY = 3                            # 单个请求的最大重试次数


def parse_args():
    p = argparse.ArgumentParser(description="批量拉取 UN Comtrade 贸易数据")
    p.add_argument("--reporter", default="276",
                   help="报告国代码，多个用逗号分隔，如 276,392,410")
    p.add_argument("--years", default="2023", help="年份，多个用逗号分隔")
    p.add_argument("--codes", default="8518", help="HS 编码，多个用逗号分隔")
    p.add_argument("--flow", default="M", choices=["M", "X"],
                   help="M=进口，X=出口")
    p.add_argument("--partner", default="0", help="伙伴国代码，0=全球合计")
    p.add_argument("--key", default=os.environ.get("COMTRADE_KEY", ""),
                   help="订阅密钥；留空则使用预览接口")
    p.add_argument("--out", default="imports_clean.csv", help="清洗后输出文件名")
    return p.parse_args()


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, encoding="utf-8") as f:
            return set(tuple(x) for x in json.load(f))
    return set()


def save_state(done):
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(list(done)), f, ensure_ascii=False)


def fetch(api, headers, reporter, year, code, flow, partner):
    """单次请求；失败自动重试，返回记录列表（失败返回 None）。"""
    params = {
        "reporterCode": reporter,
        "period": year,
        "cmdCode": code,
        "flowCode": flow,
        "partnerCode": partner,
    }
    for attempt in range(1, MAX_RETRY + 1):
        try:
            r = requests.get(api, params=params, headers=headers, timeout=30)
            if r.status_code == 429:                      # 配额超限：等待后重试
                time.sleep(5 * attempt)
                continue
            r.raise_for_status()
            return r.json().get("data", [])
        except Exception as exc:                          # noqa: BLE001
            if attempt == MAX_RETRY:
                print(f"    [失败] {reporter}/{year}/{code}：{exc}")
                return None
            time.sleep(2 * attempt)
    return None


def main():
    args = parse_args()
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(CLEAN_DIR, exist_ok=True)

    api = FULL_API if args.key else PREVIEW_API
    headers = {"Ocp-Apim-Subscription-Key": args.key} if args.key else {}
    print(f"接口：{'正式接口' if args.key else '预览接口（无密钥）'}")

    reporters = [x.strip() for x in args.reporter.split(",") if x.strip()]
    years = [x.strip() for x in args.years.split(",") if x.strip()]
    codes = [x.strip() for x in args.codes.split(",") if x.strip()]

    done = load_state()
    records, failed = [], []
    total = len(reporters) * len(years) * len(codes)
    n = 0

    for rep in reporters:
        for yr in years:
            for code in codes:
                n += 1
                key = (rep, yr, code, args.flow, args.partner)
                if key in done:
                    print(f"[{n}/{total}] 跳过（已完成）{rep}/{yr}/{code}")
                    continue
                print(f"[{n}/{total}] 拉取 {rep}/{yr}/{code} ...")
                data = fetch(api, headers, rep, yr, code, args.flow, args.partner)
                if data is None:
                    failed.append(key)
                else:
                    records += data
                    done.add(key)
                    save_state(done)                      # 即时落盘，可断点续跑
                time.sleep(SLEEP_SEC)

    # ------------------------------------------------ 落盘与清洗
    with open(os.path.join(RAW_DIR, "comtrade_all.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    print(f"\n原始记录：{len(records)} 条；失败组合：{len(failed)} 个")

    if not records:
        print("没有取到数据：请检查参数、密钥或配额。")
        return 1

    df = pd.json_normalize(records)
    cols = ["refYear", "reporterCode", "reporterDesc", "cmdCode", "cmdDesc",
            "partnerDesc", "flowDesc", "primaryValue", "netWgt"]
    cols = [c for c in cols if c in df.columns]           # 兼容字段差异
    df = df[cols]

    # 去重：同一年可能同时存在月度与年度记录，必须按年份去重
    keys = [c for c in ["refYear", "reporterCode", "cmdCode"] if c in df.columns]
    df = df.sort_values("refYear").drop_duplicates(subset=keys, keep="last")

    out_path = os.path.join(CLEAN_DIR, args.out)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"清洗后：{len(df)} 行 -> {out_path}")

    if "reporterDesc" in df.columns and "primaryValue" in df.columns:
        print("\n按报告国汇总（金额）：")
        print(df.groupby("reporterDesc")["primaryValue"].sum().round(0).to_string())

    if failed:
        print("\n以下组合未取到数据，可再次运行本脚本重试：")
        for k in failed:
            print("   ", k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
