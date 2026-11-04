#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aigc_diffusion.py -- 扩散模型一维演示：前向加噪与反向去噪

用途：
    在一个一维「信号」上演示扩散模型的两个过程：
    1) 前向：按 beta 调度逐步加高斯噪声，直到接近纯噪声
    2) 反向：用一个「学习到的」噪声预测器逐步去噪，还原信号
    并演示无分类器引导（CFG）如何放大条件的影响。

输入：
    合成的一维信号（无需外部数据）
输出：
    各时间步的信号与噪声水平、去噪后的重建误差、不同引导尺度 s 的效果
依赖：
    numpy（可选 chartlib 出 SVG 图）
对应课件：
    模块五第三章 §6 算法 6-1 扩散模型、算法 6-2 条件生成与 CFG

说明：
    真实扩散模型在图像空间、用 U-Net 预测噪声；这里降到一维、
    用「解析已知的噪声」代替学习，目的是让每一步都看得见。
"""

import numpy as np

T_STEPS = 100
SEED = 9


def make_signal(n=64, seed=SEED):
    """构造一条平滑的一维信号（正弦叠加 + 慢趋势）。"""
    x = np.linspace(0, 4 * np.pi, n)
    sig = np.sin(x) + 0.5 * np.sin(2.3 * x + 0.7) + 0.02 * x
    return sig


def beta_schedule(t_steps=T_STEPS, beta_start=1e-4, beta_end=0.16):
    """线性 beta 调度。"""
    return np.linspace(beta_start, beta_end, t_steps)


def forward_diffuse(x0, betas, seed=SEED):
    """前向过程：逐步加噪，返回每个时间步的样本。"""
    rng = np.random.default_rng(seed)
    xs = [x0.copy()]
    x = x0.copy()
    for b in betas:
        x = np.sqrt(1 - b) * x + np.sqrt(b) * rng.normal(0, 1, x0.shape)
        xs.append(x.copy())
    return xs


def alpha_bar(betas):
    """累积 alpha_bar = prod(1 - beta)。"""
    return np.cumprod(1 - betas)


def forward_closed_form(x0, abar_t, seed=SEED):
    """闭式加噪：q(x_t|x_0) = N(sqrt(abar_t) x_0, (1-abar_t) I)。"""
    rng = np.random.default_rng(seed)
    return np.sqrt(abar_t) * x0 + np.sqrt(1 - abar_t) * rng.normal(0, 1, x0.shape)


def denoise_demo(x0, betas, seed=SEED):
    """反向去噪演示。

    说明：真实采样时不知道 x0，噪声由网络预测。
    这里用「由 x0 反推出的真实噪声」作为理想预测器，
    目的是验证反向公式本身能还原信号（而不是演示网络能力）。
    """
    abar = alpha_bar(betas)
    rng = np.random.default_rng(seed)
    xT = forward_closed_form(x0, abar[-1], seed=seed)
    x = xT.copy()
    traj = [x.copy()]
    for t in range(len(betas) - 1, -1, -1):
        b = betas[t]
        ab = abar[t]
        eps_true = (x - np.sqrt(ab) * x0) / np.sqrt(max(1 - ab, 1e-12))
        x = (x - (b / np.sqrt(max(1 - ab, 1e-12))) * eps_true) / np.sqrt(1 - b)
        traj.append(x.copy())
    return traj


def cfg_demo(eps_uncond, eps_cond, scales=(1.0, 3.0, 7.0, 12.0)):
    """CFG：eps_hat = eps_uncond + s * (eps_cond - eps_uncond)。"""
    return {s: eps_uncond + s * (eps_cond - eps_uncond) for s in scales}


def demo():
    x0 = make_signal()
    betas = beta_schedule()
    abar = alpha_bar(betas)

    print("=== 扩散模型一维演示 ===")
    print(f"信号长度 {len(x0)} | 时间步 T={T_STEPS}")
    print(f"beta: {betas[0]:.2e} -> {betas[-1]:.2e} | abar: {abar[0]:.4f} -> {abar[-1]:.4f}")

    xs = forward_diffuse(x0, betas)
    print("\n=== 前向加噪：部分时间步的信噪比 ===")
    for t in (0, 25, 50, 75, 100):
        x = xs[t]
        snr = np.var(x0) / max(np.var(x - x0), 1e-12)
        print(f"  t={t:2d}  与原信号的相关系数 {np.corrcoef(x0, x)[0,1]:+.3f}  "
              f"信噪比 {10*np.log10(snr):+6.1f} dB")

    traj = denoise_demo(x0, betas)
    err_start = float(np.mean((traj[0] - x0) ** 2))
    err_end = float(np.mean((traj[-1] - x0) ** 2))
    print("\n=== 反向去噪（用真实噪声作为理想预测器）===")
    print(f"  起始 MSE {err_start:.4f} -> 结束 MSE {err_end:.6f}")
    print("  说明：这里验证的是反向公式本身；真实采样用网络预测噪声，会有误差。")

    rng = np.random.default_rng(SEED)
    eps_u = rng.normal(0, 1, len(x0))
    eps_c = eps_u + 0.8 * np.sign(np.sin(np.linspace(0, 6, len(x0))))
    cfg = cfg_demo(eps_u, eps_c)
    print("\n=== 引导尺度 s 对条件偏离度的影响 ===")
    for s, eps in cfg.items():
        dev = float(np.mean(np.abs(eps - eps_u)))
        print(f"  s={s:5.1f}  平均偏离无条件预测 {dev:.3f}  （越大越贴合提示词，也越僵化）")

    print("\n结论：前向把结构逐步销毁，反向逐步重建；")
    print("      训练只学「这一步加了什么噪声」，所以比直接生成图像稳定得多。")
    return x0, xs, traj, cfg


# ---------------------------------------------------------------- 可视化
def save_visual(path='visuals/aigc_diffusion.svg'):
    """输出一条信号在四个时间步上的形态（加噪过程）。"""
    from chartlib import LineChart
    x0, xs, _, _ = demo()
    idx = [0, 33, 66, 100]
    names = [f't={i}' for i in idx]
    xs_pos = list(range(1, len(x0) + 1))
    ch = LineChart('一维扩散：信号如何被噪声吞没', '位置', '信号值')
    for name, i in zip(names, idx):
        ch.add_line(name, xs_pos, [round(float(v), 3) for v in xs[i]],
                    markers=(i == 0))
    ch.add_note('t=0 是原信号，t=100 接近纯噪声；生成时从 t=100 反向走到 t=0')
    return ch.save(path)


if __name__ == '__main__':
    demo()
