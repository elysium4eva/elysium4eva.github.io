#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rec_bandit.py -- 冷启动与探索：ε-greedy / UCB / Thompson 采样

用途：
    在同一组「素材/商品」转化率环境下，对比三种探索策略的累计收益，
    说明为什么「只投已知最优」会错过真正的好素材，以及 UCB / TS 为何更优。

输入：
    合成的多臂老虎机环境（5 个臂，各自真实转化率未知）
输出：
    三种策略的累计收益曲线、各自最终选择次数
依赖：
    numpy（可选 chartlib 出 SVG 图）
对应课件：
    模块五第三章 §5 算法 5-6 冷启动与探索

说明：
    这是「新品冷启动」「素材轮换」的最小模型；真实业务中一个臂
    可能是某个素材、某个达人、某个推荐位。
"""

import numpy as np

N_ARMS = 10
TRUE_P = [0.030, 0.041, 0.052, 0.048, 0.038, 0.055, 0.033, 0.045, 0.050, 0.036]
N_ROUNDS = 3000
SEED = 17


class Env:
    """伯努利老虎机环境。"""

    def __init__(self, p, seed=SEED):
        self.p = np.array(p, dtype=float)
        self.rng = np.random.default_rng(seed)

    def pull(self, arm):
        return int(self.rng.random() < self.p[arm])


def run_eps_greedy(env, eps=0.1, rounds=N_ROUNDS, seed=SEED):
    rng = np.random.default_rng(seed)
    n = np.zeros(N_ARMS)
    s = np.zeros(N_ARMS)
    curve = []
    for t in range(rounds):
        if rng.random() < eps:
            a = int(rng.integers(0, N_ARMS))
        else:
            mean = np.where(n > 0, s / np.maximum(n, 1), 0)
            a = int(np.argmax(mean))
        r = env.pull(a)
        n[a] += 1
        s[a] += r
        curve.append(s.sum())
    return curve, n


def run_ucb(env, rounds=N_ROUNDS, c=0.35):
    n = np.zeros(N_ARMS)
    s = np.zeros(N_ARMS)
    curve = []
    for t in range(rounds):
        if (n == 0).any():
            a = int(np.argmax(n == 0))
        else:
            mean = s / n
            bonus = c * np.sqrt(2 * np.log(t + 1) / n)
            a = int(np.argmax(mean + bonus))
        r = env.pull(a)
        n[a] += 1
        s[a] += r
        curve.append(s.sum())
    return curve, n


def run_thompson(env, rounds=N_ROUNDS, seed=SEED):
    rng = np.random.default_rng(seed + 1)
    alpha = np.ones(N_ARMS)      # Beta 先验（成功次数 + 1）
    beta = np.ones(N_ARMS)       # 失败次数 + 1
    curve = []
    n = np.zeros(N_ARMS)
    for _ in range(rounds):
        theta = rng.beta(alpha, beta)
        a = int(np.argmax(theta))
        r = env.pull(a)
        if r:
            alpha[a] += 1
        else:
            beta[a] += 1
        n[a] += 1
        curve.append(int((alpha - 1).sum()))
    return curve, n


def run_pure_greedy(env, rounds=N_ROUNDS):
    """纯贪心（完全不探索）：锁定当前均值最高的臂，早先的偶然会被永久固化。"""
    n = np.zeros(N_ARMS)
    s = np.zeros(N_ARMS)
    curve = []
    for _ in range(rounds):
        mean = np.where(n > 0, s / np.maximum(n, 1), 1e9)   # 未试过的优先试一次
        a = int(np.argmax(mean))
        r = env.pull(a)
        n[a] += 1
        s[a] += r
        curve.append(s.sum())
    return curve, n


def demo():
    print("=== 冷启动与探索：三种策略对比 ===")
    print(f"臂数 {N_ARMS} | 轮次 {N_ROUNDS} | 真实转化率 {TRUE_P}")
    best = int(np.argmax(TRUE_P))
    print(f"最优臂是第 {best} 个（真实转化率 {TRUE_P[best]*100:.1f}%），但策略事先不知道\n")

    env1 = Env(TRUE_P, SEED)
    c1, n1 = run_eps_greedy(env1)
    env2 = Env(TRUE_P, SEED)
    c2, n2 = run_ucb(env2)
    env3 = Env(TRUE_P, SEED)
    c3, n3 = run_thompson(env3)
    env0 = Env(TRUE_P, SEED)
    c0, n0 = run_pure_greedy(env0)

    print(f"{'策略':<12}{'累计转化':>10}{'最优臂占比':>12}")
    for name, curve, n in [
        ("纯贪心(不探索)", c0, n0),
        ("ε-greedy", c1, n1),
        ("UCB", c2, n2),
        ("Thompson", c3, n3),
    ]:
        share = n[best] / n.sum()
        print(f"{name:<12}{curve[-1]:>10}{share*100:>11.1f}%")

    print("\n各策略的臂选择次数：")
    for name, n in [("纯贪心", n0), ("ε-greedy", n1), ("UCB", n2), ("Thompson", n3)]:
        print(f"  {name:<12}{[int(x) for x in n]}")

    print("\n结论：")
    print(f"  1) 纯贪心最差（{c0[-1]}）——它把早期的偶然当成结论，之后再不改主意。")
    print("     这就是「新品必须给探索预算」的原因。")
    print("  2) 三种探索策略都能明显改善；本例中 ε-greedy 反而最好，")
    print("     因为 10 个臂的转化率差异不大且轮次有限，过度探索是浪费。")
    print("  3) 臂更多、差异更小、轮次更长时，UCB 与 Thompson 的优势才会显现。")
    print("     选哪种要看你的场景，不要迷信「某个策略一定最好」。")
    return {"greedy0": c0, "eps": c1, "ucb": c2, "ts": c3}


# ---------------------------------------------------------------- 可视化
def save_visual(path='visuals/rec_bandit.svg'):
    """输出三种策略的累计收益曲线。"""
    from chartlib import LineChart
    r = demo()
    step = 500
    xs = list(range(0, N_ROUNDS + 1, step))
    def samp(c):
        return [float(c[min(i, len(c) - 1)]) for i in xs]
    ch = LineChart('三种探索策略的累计转化数', '轮次', '累计转化')
    ch.add_line('纯贪心(不探索)', xs, samp(r["greedy0"]), markers=False)
    ch.add_line('ε-greedy', xs, samp(r["eps"]), markers=False)
    ch.add_line('UCB', xs, samp(r["ucb"]), markers=False)
    ch.add_line('Thompson', xs, samp(r["ts"]), markers=False)
    ch.add_note('纯贪心最差：不探索就永远发现不了更好的素材')
    return ch.save(path)


if __name__ == '__main__':
    demo()
