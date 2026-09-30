#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
places_importers.py —— 用地理信息平台按地区批量获取候选客户名单

两条数据路线（可分别使用）：
    A) Google Places API —— 覆盖最好，按量计费，条款对数据存储限制严格
    B) OpenStreetMap Overpass API —— 免费，条款宽松，欧美覆盖较好

⚠️ 合规提示（使用前务必阅读）：
    1. Google Maps Platform 服务条款通常只允许长期保存有限字段（如 place ID），
       并禁止用其数据构建或补充其他数据集、禁止未经授权的转售与公开披露。
       因此本脚本默认【只保存 place_id / name / website 三类字段】，
       其余字段请在得到结果后当场核对使用，不要落盘。
    2. OSM 数据采用 ODbL 许可：使用时须署名 "© OpenStreetMap contributors"，
       并遵守相同方式共享（Share-alike）要求。
    3. 本脚本仅调用官方接口，不包含任何绕过限制的做法。

准备：
    pip install requests pandas

用法：
    # A) Google（需要 API Key 与开通 Places API）
    set GOOGLE_MAPS_KEY=你的Key          # Windows
    export GOOGLE_MAPS_KEY=你的Key       # macOS / Linux
    python places_importers.py --mode google \
        --query "consumer electronics distributor in Hamburg"

    # B) OpenStreetMap（免费）
    python places_importers.py --mode osm --city Hamburg \
        --shop "electronics|hifi|mobile_phone"
"""

import argparse
import os
import time

import pandas as pd
import requests

OUT_DIR = "clean"
PLACES_URL = "https://places.googleapis.com/v1/places:searchText"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def parse_args():
    p = argparse.ArgumentParser(description="地理信息平台批量获取候选客户名单")
    p.add_argument("--mode", choices=["google", "osm"], default="osm")
    p.add_argument("--query", default="consumer electronics distributor",
                   help="Google 文本搜索关键词")
    p.add_argument("--city", default="Hamburg", help="OSM 查询的城市名")
    p.add_argument("--shop", default="electronics",
                   help="OSM 店铺类别标签，多个用竖线分隔，如 electronics|hifi")
    p.add_argument("--pages", type=int, default=3, help="Google 最多取几页")
    p.add_argument("--out", default="lead_list.csv", help="输出文件名")
    p.add_argument("--key", default=os.environ.get("GOOGLE_MAPS_KEY", ""))
    return p.parse_args()


# ============================================================ A) Google Places
def google_search(query, key, token=None):
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": key,
        # 字段掩码：只声明需要的字段，既省费用也降低合规风险
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,"
                            "places.websiteUri,places.id,nextPageToken",
    }
    body = {"textQuery": query, "languageCode": "en", "pageSize": 20}
    if token:
        body["pageToken"] = token
    r = requests.post(PLACES_URL, json=body, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def google_all(query, key, max_pages):
    out, token, seen = [], None, set()
    for i in range(max_pages):
        j = google_search(query, key, token)
        batch = j.get("places", [])
        for p in batch:
            if p["id"] not in seen:
                seen.add(p["id"])
                out.append(p)
        print(f"  第 {i + 1} 页：本页 {len(batch)} 条，累计 {len(out)} 条")
        token = j.get("nextPageToken")
        if not token:
            break
        time.sleep(2)                    # 分页需间隔，分页令牌才会生效
    return out


def run_google(args):
    if not args.key:
        print("缺少 API Key：请设置环境变量 GOOGLE_MAPS_KEY 或使用 --key 参数。")
        print("（也可先用 --mode osm 免费跑通流程）")
        return 1
    places = google_all(args.query, args.key, args.pages)
    rows = [{
        "name":     p["displayName"]["text"],
        "website":  p.get("websiteUri", ""),
        "place_id": p["id"],          # 条款允许长期保存的字段
    } for p in places]
    df = pd.DataFrame(rows).drop_duplicates("place_id")
    print(f"\n去重后 {len(df)} 家公司")
    print("提示：地址等其余字段请当场核对使用，不要长期落盘（见脚本头部合规说明）。")
    return df


# ============================================================ B) OpenStreetMap
def run_osm(args):
    query = f"""
[out:json][timeout:90];
area["name"="{args.city}"]["admin_level"="4"]->.a;
node["shop"~"{args.shop}"](area.a);
out center 300;
"""
    print("查询 Overpass API ...")
    r = requests.post(OVERPASS_URL, data={"data": query}, timeout=180)
    r.raise_for_status()
    els = r.json().get("elements", [])
    print(f"返回 {len(els)} 个地点")

    rows = [{
        "name":    e.get("tags", {}).get("name", ""),
        "shop":    e.get("tags", {}).get("shop", ""),
        "street":  e.get("tags", {}).get("addr:street", ""),
        "website": e.get("tags", {}).get("website", ""),
        "lat":     e.get("lat"),
        "lon":     e.get("lon"),
    } for e in els]
    df = pd.DataFrame(rows).dropna(subset=["name"]).drop_duplicates("name")
    print(f"去重后 {len(df)} 家商户")
    print("署名要求：使用本数据须注明 © OpenStreetMap contributors（ODbL 许可）。")
    return df


def enrich(df):
    """补上客户开发流程需要的字段，便于直接跟进。"""
    if "website" in df.columns:
        df["domain"] = (df["website"].fillna("")
                        .str.extract(r"https?://([^/]+)", expand=False)
                        .str.replace(r"^www\.", "", regex=True))
    type_map = {"electronics": "电子零售商", "hifi": "音响零售商",
                "mobile_phone": "手机零售商", "wholesale": "批发商"}
    if "shop" in df.columns:
        df["customer_type"] = df["shop"].map(type_map).fillna("待确认")
    df["lead_grade"] = ""                       # A/B/C 线索等级（手工填写）
    df["next_action"] = "核对官网产品线"
    df["next_date"] = ""
    return df


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    df = run_google(args) if args.mode == "google" else run_osm(args)
    if isinstance(df, int):
        return df

    df = enrich(df)
    path = os.path.join(OUT_DIR, args.out)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n已导出：{path}（{len(df)} 条）")
    print("下一步：按课件第 9 节的 ICP 条件筛选，再进入线索评分与跟进流程。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
