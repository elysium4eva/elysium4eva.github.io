#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sel_bsr_forecast.py —— 选品两件事：BSR 反推销量（回归）+ 季节性分解与备货预测

对应课件：模块五 第二章 §4「智能选品的典型算法」
          — 回归预测族（线性回归 / 树模型）
          — 时间序列族（ARIMA / Prophet / 经典分解）

用途：
    A) BSR → 销量：在 log-log 坐标下拟合「排名越靠后，销量越低」的关系，
       给出销量点估计与**预测区间**——强调工具里的月销数字是个区间，不是点值。
    B) 季节性分解：用移动平均把时间序列拆成「趋势 + 季节 + 残差」，
       算出季节指数，用于备货节奏判断（什么时候该压货、什么时候该清仓）。

为什么先取对数：
    BSR 与销量的关系接近幂律（销量 ∝ BSR^-k）。取对数后变成线性关系，
    斜率的绝对值就是 k，也便于看出「类目越靠后衰减越快」。

依赖：numpy、pandas（本课程环境自带）；不依赖 statsmodels / prophet
运行：python sel_bsr_forecast.py

数据说明：全部为教学用合成数据，用以演示方法，不代表任何真实 ASIN。
"""

import numpy as np
import pandas as pd


# ================================================================ A) BSR 回归
def fit_power_law(bsr, sales):
    """log(sales) = a + b * log(bsr)，最小二乘拟合，返回系数与残差标准差。"""
    x = np.log(np.asarray(bsr, dtype=float))
    y = np.log(np.asarray(sales, dtype=float))
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef
    resid = y - (a + b * x)
    dof = max(len(x) - 2, 1)
    sigma = float(np.sqrt((resid ** 2).sum() / dof))
    r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return a, b, sigma, r2


def predict_interval(a, b, sigma, bsr, level=0.80):
    """对数正态假设下的预测区间（近似）。

    level=0.80 表示 80% 预测区间——这就是我们要教学生看的「区间」。
    """
    z = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600}[level]
    log_mu = a + b * np.log(np.asarray(bsr, dtype=float))
    lo = np.exp(log_mu - z * sigma)
    hi = np.exp(log_mu + z * sigma)
    mid = np.exp(log_mu)
    return mid, lo, hi


def demo_bsr():
    print("=" * 72)
    print("A) BSR 反推销量：为什么工具给的月销永远是个区间")
    print("=" * 72)

    # 构造 40 个已知点（模拟某类目的基准对照表）
    rng = np.random.default_rng(11)
    bsr = np.array([20, 35, 50, 80, 120, 180, 260, 400, 600, 900,
                    1300, 1900, 2700, 4000, 5800, 8200, 12000, 17000, 24000, 34000,
                    48000, 68000, 96000, 135000, 190000, 270000, 380000, 530000,
                    750000, 1050000, 1500000, 2100000, 3000000, 4200000, 6000000,
                    8500000, 12000000, 17000000, 24000000, 34000000], dtype=float)
    # 真实关系：销量 ≈ 90000 * bsr^-0.72，再叠乘性噪声
    true_sales = 90000 * bsr ** -0.72
    sales = true_sales * rng.lognormal(mean=0.0, sigma=0.18, size=len(bsr))

    a, b, sigma, r2 = fit_power_law(bsr, sales)
    print(f"\n拟合结果：log(销量) = {a:.4f} + ({b:.4f}) × log(BSR)")
    print(f"  幂指数 k = {-b:.3f}（排名越靠后，销量按 BSR^k 衰减）")
    print(f"  R^2 = {r2:.4f}，残差标准差 sigma = {sigma:.4f}")

    print("\n用它反推几个排名的销量（80% 预测区间）：")
    test_bsr = [100, 500, 2000, 10000, 100000]
    mid, lo, hi = predict_interval(a, b, sigma, test_bsr, level=0.80)
    out = pd.DataFrame({
        "BSR": test_bsr,
        "点估计月销": mid.round(0).astype(int),
        "区间下限": lo.round(0).astype(int),
        "区间上限": hi.round(0).astype(int),
        "上下限倍数": (hi / lo).round(2),
    })
    print(out.to_string(index=False))

    print("\n关键结论：")
    print("  · 点估计只是「最可能的值」，真实值大概率落在区间里。")
    print("  · 上下限相差 2 倍以上是常态——所以课件强调「用区间做备货，不要按点值规划」。")
    print("  · 每个类目的基准表不同，跨类目不能直接套用同一条曲线。")


# ================================================================ B) 季节性
def seasonal_decompose(series, period=12):
    """经典加法分解：趋势（居中移动平均）+ 季节（同月均值去趋势）+ 残差。"""
    s = pd.Series(series, dtype=float)
    # 1) 趋势：居中移动平均
    if period % 2 == 0:
        trend = s.rolling(period, center=True).mean()
        # 偶数期用两段平均对齐
        trend = s.rolling(period // 2, center=True).mean().rolling(period // 2, center=True).mean()
    else:
        trend = s.rolling(period, center=True).mean()

    # 2) 去趋势
    detrended = s - trend

    # 3) 季节指数：按位置取均值
    idx = np.arange(len(s)) % period
    season_idx = detrended.groupby(idx).mean()
    season_idx = season_idx - season_idx.mean()          # 中心化
    seasonal = pd.Series([season_idx[i] for i in idx], index=s.index)

    # 4) 残差
    resid = s - trend - seasonal
    return pd.DataFrame({"原始": s, "趋势": trend, "季节": seasonal, "残差": resid})


def demo_seasonal():
    print("\n" + "=" * 72)
    print("B) 季节性分解：什么时候压货、什么时候清仓")
    print("=" * 72)

    rng = np.random.default_rng(5)
    months = pd.date_range("2024-01-01", periods=36, freq="MS")
    t = np.arange(36)
    trend = 1000 + 18 * t                                   # 缓慢上升
    season = 320 * np.sin(2 * np.pi * (t % 12) / 12 - 1.0)  # 12 个月周期
    noise = rng.normal(0, 45, 36)
    sales = trend + season + noise

    dec = seasonal_decompose(sales, period=12)
    dec.index = months

    print("\n季节指数（相对年均值的偏离，单位：件）：")
    se = dec["季节"].iloc[:12].to_numpy()
    month_names = [f"{m}月" for m in range(1, 13)]
    order = np.argsort(se)
    for i in order[::-1][:3]:
        print(f"  旺季 {month_names[i]:>4}：+{se[i]:.0f} 件")
    for i in order[:2]:
        print(f"  淡季 {month_names[i]:>4}：{se[i]:.0f} 件")

    peak, trough = int(order[-1]), int(order[0])
    print(f"\n最强月：{month_names[peak]}（季节指数 {se[peak]:+.0f}）")
    print(f"最弱月：{month_names[trough]}（季节指数 {se[trough]:+.0f}）")
    print(f"旺淡季差：{se[peak] - se[trough]:.0f} 件，约占年均销量的 "
          f"{(se[peak] - se[trough]) / dec['原始'].mean() * 100:.0f}%")

    print("\n备货含义：")
    print(f"  · 旺季前 2–3 个月开始备货（考虑生产 45 天 + 海运 30 天），")
    print(f"    即大约在 {month_names[(peak - 3) % 12]} 就要下单。")
    print(f"  · 淡季（{month_names[trough]}）避免压货，可安排清仓或做测款。")

    print("\n看最近 6 个月的残差（判断有没有异常）：")
    tail = dec.tail(6)[["原始", "趋势", "季节", "残差"]].round(1)
    print(tail.to_string())



# ---------------------------------------------------------------- 可视化
def save_visual(path='visuals/sel_bsr_forecast.svg'):
    """输出月销量的「实际 vs 趋势」折线图，展示季节波动幅度。"""
    from chartlib import LineChart
    rng = np.random.default_rng(5)
    t = np.arange(36)
    sales = 1000 + 18 * t + 320 * np.sin(2 * np.pi * (t % 12) / 12 - 1.0) + rng.normal(0, 45, 36)
    dec = seasonal_decompose(sales, period=12)
    trend = dec['趋势'].fillna(float(np.mean(sales))).to_numpy()
    ch = LineChart('月销量：实际波动与趋势项', '月份序号（1-36）', '销量（件）')
    ch.add_line('实际销量', list(range(1, 37)), [float(v) for v in sales])
    ch.add_line('趋势项', list(range(1, 37)), [float(v) for v in trend])
    se = dec['季节'].iloc[:12].to_numpy()
    ch.add_note(f'季节指数峰谷差约 {float(se.max() - se.min()):.0f} 件，约合年均的 '
                f'{(se.max() - se.min()) / float(np.mean(sales)) * 100:.0f}%')
    return ch.save(path)

if __name__ == "__main__":
    demo_bsr()
    demo_seasonal()
