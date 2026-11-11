#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ch4_diffusion_2d.py -- 二维空间的扩散生成模型：把噪声还原成一个数字

用途：
    在二维平面上演示扩散模型的两个过程：
      前向：把一个「数字 3」的点集一步步加噪声，直到变成一团高斯噪声；
      反向：从纯噪声出发，一步步去噪，把一个 3 的形状重新「长」出来。
    关键是让你亲眼看到：生成 = 反复去噪，而不是一次性画出来。

输入：
    内置「数字 3」的二维点集（约 240 个点，参数曲线生成）
输出：
    前向噪声的 MSE 轨迹、反向去噪的 MSE 轨迹、五帧采样图
依赖：
    numpy、viz_png
对应课件：
    模块五第四章 §9 算法五（扩散生成模型）

说明：
    教学简化：真正的扩散模型用一个神经网络预测噪声 ε，这里为了「无依赖、能跑、
    结果稳定」，改用一个**非参数去噪器**——按高斯核把噪声点往训练点集拉。
    它演示的是同一个机制（逐步去噪 + 噪声调度），但不是神经网络实现。
"""

import numpy as np

T = 60                 # 扩散步数
BETA_START, BETA_END = 1e-4, 0.18
N_POINTS = 240
SEED = 11


def make_digit3(n=N_POINTS):
    """沿一条「3」的折线路径等距采样 n 个点。

    用折线而不是两段圆弧：圆弧会在中段相交，生成点容易堆在交点；
    折线是单笔画（sharp 但没有分叉），去噪时每个点都有唯一的最近落点。
    坐标范围约 x∈[-0.45, 0.55], y∈[-0.62, 0.68]。
    """
    ctrl = np.array([
        [-0.28, 0.58], [-0.10, 0.70], [0.14, 0.68], [0.32, 0.52],
        [0.36, 0.34], [0.24, 0.18], [0.06, 0.10],
        [0.26, 0.04], [0.40, -0.10], [0.40, -0.30],
        [0.28, -0.50], [0.06, -0.60], [-0.18, -0.54], [-0.34, -0.42],
    ])
    # 用 Catmull-Rom 插值把折线变平滑，再等距采样
    def catmull_rom(p0, p1, p2, p3, t):
        t2, t3 = t * t, t * t * t
        return 0.5 * ((2 * p1) + (-p0 + p2) * t +
                      (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                      (-p0 + 3 * p1 - 3 * p2 + p3) * t3)

    dense = []
    L = len(ctrl)
    for k in range(L):
        p0 = ctrl[(k - 1) % L]
        p1 = ctrl[k]
        p2 = ctrl[(k + 1) % L]
        p3 = ctrl[(k + 2) % L]
        for t in np.linspace(0, 1, 40, endpoint=False):
            dense.append(catmull_rom(p0, p1, p2, p3, t))
    dense = np.array(dense)

    # 等弧长采样
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    targets = np.linspace(0, total * 0.995, n)
    out = np.empty((n, 2))
    for m, tv in enumerate(targets):
        idx = int(np.searchsorted(cum, tv) - 1)
        idx = min(max(idx, 0), len(dense) - 2)
        r = (tv - cum[idx]) / max(seg[idx], 1e-9)
        out[m] = dense[idx] + r * (dense[idx + 1] - dense[idx])
    # 轻微抖动，避免点完全共线（更像手写）
    rng = np.random.default_rng(5)
    out = out + rng.normal(0, 0.008, out.shape)
    return out


def schedule(t_steps=T, b0=BETA_START, b1=BETA_END):
    betas = np.linspace(b0, b1, t_steps)
    alphas = 1.0 - betas
    abar = np.cumprod(alphas)
    return betas, alphas, abar


def forward(x0, abar, rng):
    """返回每一帧的 x_t（t 越大越像噪声）。"""
    frames = []
    for a in abar:
        eps = rng.standard_normal(x0.shape)
        frames.append(np.sqrt(a) * x0 + np.sqrt(1 - a) * eps)
    return frames


def score_kde(x, train, sigma):
    """核密度估计的分数函数 ∇log p_σ(x) = (Σ w_k μ_k − x) / σ²。

    这就是「扩散模型要学的那个东西」的无参数版本：
    真实模型用一个神经网络逼近它，这里因为数据只有 240 个点，可以直接算出来。
    """
    d2 = ((x[:, None, :] - train[None, :, :]) ** 2).sum(axis=2)
    w = np.exp(-d2 / (2 * sigma * sigma))
    w = w / (w.sum(axis=1, keepdims=True) + 1e-12)
    return (w @ train - x) / (sigma * sigma)


def annealed_langevin(x, train, sigma, s_max, rng, steps=40, eps=0.02):
    """退火朗之万采样（Song & Ermon 2019 的思路，用在核密度分数上）。

    从大方差（几乎全图）逐步退到小方差（贴住数据），每一步：
        x ← x + α·score(x, σ) + √(2α)·噪声
    α 与 σ² 成正比，保证每一步的「信噪比」大致恒定。
    """
    alpha = eps * (sigma * sigma) / (s_max * s_max)
    x = np.clip(x, -3.0, 3.0)
    for _ in range(steps):
        sc = np.clip(score_kde(x, train, sigma), -50.0, 50.0)
        x = x + alpha * sc + np.sqrt(2 * alpha) * rng.standard_normal(x.shape)
        x = np.clip(x, -3.0, 3.0)
    return x


def chamfer(a, b):
    """Chamfer 距离：a 到 b 的最近距离 + 反向，取平均（不受点顺序影响）。"""
    d_ab = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2).min(axis=1).mean()
    d_ba = ((b[:, None, :] - a[None, :, :]) ** 2).sum(axis=2).min(axis=1).mean()
    return float(np.sqrt(d_ab) + np.sqrt(d_ba)) / 2.0


def mse(a, b):
    return float(((a - b) ** 2).mean())


def demo():
    rng = np.random.default_rng(SEED)
    x0 = make_digit3()
    betas, alphas, abar = schedule()

    print('=== 二维扩散：把噪声还原成一个数字 ===')
    print('步数 T = %d，beta 从 %.0e 线性到 %.2f' % (T, BETA_START, BETA_END))
    print('训练点集：%d 个二维点组成的「3」\n' % len(x0))

    # 前向
    fwd = forward(x0, abar, rng)
    print('--- 前向：一步步加噪 ---')
    for t in (0, T // 4, T // 2, T - 1):
        a = abar[t]
        print('  t=%2d  abar=%.4f  Chamfer=%.4f  %s' % (
            t, a, chamfer(fwd[t], x0), '几乎原图' if a > 0.8 else ('半糊' if a > 0.2 else '纯噪声')))

    # 反向：退火朗之万采样（从纯噪声出发）
    print('\n--- 反向：退火朗之万采样（从纯噪声出发，25 个噪声级别 × 40 步）---')
    sigmas = np.geomspace(0.50, 0.02, 25)
    s_max = float(sigmas[0])
    x = rng.standard_normal(x0.shape) * sigmas[0]
    snaps = {T - 1: x.copy()}
    traj = []
    for stage, sigma in enumerate(sigmas):
        x = annealed_langevin(x, x0, float(sigma), s_max, rng, steps=40)
        frac = 1.0 - (stage + 1) / len(sigmas)
        traj.append((stage, chamfer(x, x0)))
        if stage in (0, 6, 12, 18, len(sigmas) - 1):
            snaps[stage] = x.copy()
    for stage in (0, 6, 12, 18, len(sigmas) - 1):
        print('  σ=%.3f  Chamfer=%.4f' % (sigmas[stage], chamfer(snaps[stage], x0)))

    first = traj[1][1] if len(traj) > 1 else traj[0][1]
    last = traj[-1][1]
    print('\n  Chamfer 距离从 %.3f 降到 %.3f（下降 %.0f%%）' % (
        first, last, (1 - last / max(first, 1e-9)) * 100))

    print('\n结论：')
    print('  1) 生成不是「一次画出 3」，而是「从噪声出发，去噪 60 次」。')
    print('  2) 前向过程只加噪，不需要训练；真正要学的是「反向的每一步该往哪去」。')
    print('  3) 本例用「核密度的分数函数 + 退火朗之万」代替神经网络——算法同一个，模型换成了无参的。')
    print('  4) 机制完全相同：噪声调度 + 逐步去噪 + 条件引导（CFG 就是在这里插进去的）。')
    return snaps, traj


def save_png(path='visuals/ch4_diffusion_2d.png'):
    from viz_png import Canvas, RED, RED_L, GOLD, GOLD_L, INK, MUTED, LINE, lerp

    snaps, traj = demo()
    x0 = make_digit3()
    cv = Canvas(1200, 700)
    cv.title('二维扩散：从一团噪声里「长」出一个 3')

    # 五帧
    order = [0, 6, 12, 18, 24]
    labels = ['σ = 0.50', 'σ = 0.16', 'σ = 0.06', 'σ = 0.03', 'σ = 0.02 完成']
    pw = 210
    for k, t in enumerate(order):
        x = 50 + k * (pw + 24)
        y = 100
        cv.rect(x, y, pw, pw, fill=(255, 255, 255), outline=LINE, width=2, r=10)
        pts = snaps[t]
        cx, cy = x + pw / 2, y + pw / 2
        s = pw * 0.40
        for p in pts:
            if not (-1.0 <= p[0] <= 1.0 and -1.0 <= p[1] <= 1.0):
                continue
            px = cx + p[0] * s
            py = cy - p[1] * s
            if not (x + 4 <= px <= x + pw - 4 and y + 4 <= py <= y + pw - 4):
                continue
            r = 2.6 if t > 0 else 3.2
            cv.d.ellipse([px - r, py - r, px + r, py + r],
                         fill=GOLD if t == 0 else RED)
        cv.text(x + pw / 2, y + pw + 14, labels[k], 14, INK, True, 'mm')
        cv.text(x + pw / 2, y + pw + 38, 'Chamfer %.3f' % chamfer(snaps[t], x0), 12, MUTED, False, 'mm')
        if k < 4:
            cv.arrow(x + pw + 3, y + pw / 2, x + pw + 21, y + pw / 2, color=MUTED, width=2)

    cv.text(50, 400, '从纯噪声出发，噪声级别从 σ=0.50 退到 σ=0.02：一个「3」逐步浮现', 14, INK, True)
    cv.text(50, 426, '每一帧都是采样得到的点集（不是画出来的）；Chamfer 越低表示形状越贴合真实数字', 12, MUTED)

    # 下方 MSE 曲线（手绘折线）
    bx, by, bw, bh = 50, 470, 640, 170
    cv.rect(bx, by, bw, bh, fill=(255, 255, 255), outline=LINE, width=1, r=8)
    cv.text(bx + 12, by + 8, '反向过程中的 Chamfer 距离（越接近 0，形状越对得上）', 13, INK, True)
    xs = [d[0] for d in traj]
    ys = [d[1] for d in traj]
    ymax = max(ys) or 1
    n = len(traj)
    for i in range(n - 1):
        x1 = bx + 20 + (bw - 40) * i / (n - 1)
        y1 = by + bh - 24 - (bh - 60) * ys[i] / ymax
        x2 = bx + 20 + (bw - 40) * (i + 1) / (n - 1)
        y2 = by + bh - 24 - (bh - 60) * ys[i + 1] / ymax
        cv.line(x1, y1, x2, y2, color=RED, width=2)
    cv.text(bx + 16, by + bh - 16, 'σ = 0.50', 11, MUTED)
    cv.text(bx + bw - 64, by + bh - 16, 'σ = 0.02', 11, MUTED)

    # 右侧说明
    px = 730
    notes = [
        ('前向（不用学）', 'x_t = √ā·x_0 + √(1-ā)·ε', '只加噪，无需训练'),
        ('反向（要学）', '每步预测噪声 ε，再减去它', '真实模型：神经网络'),
        ('本例的做法', '核密度分数 + 退火采样', '无参数版分数函数'),
        ('条件引导', 'CFG 在这一步加条件', '第 3 章已讲，不重复'),
    ]
    for k, (t, f, d) in enumerate(notes):
        y = 470 + k * 52
        cv.rect(px, y, 420, 44, fill=(255, 255, 255), outline=LINE, width=1, r=8)
        cv.text(px + 12, y + 6, t, 12, RED, True)
        cv.text(px + 12, y + 24, f, 11, INK)
        cv.text(px + 300, y + 14, d, 10, MUTED, False, 'ra')

    cv.text(30, 675, '数据：240 个二维点组成的「3」，T=60；程序 ch4_diffusion_2d.py（去噪器为非参数简化）', 11, MUTED)
    return cv.save(path)


if __name__ == '__main__':
    demo()
