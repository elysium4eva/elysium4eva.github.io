#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ch3_rec_graph.py -- 图推荐：两层传播 = 数「二跳路径」

用途：
    在「3 个用户 x 4 个商品」的二部图上，演示图推荐的本质：
    第一层看你自己（你买过什么），第二层看「和你买过同一批东西的人」买过什么。
    写成公式就是：给 u 推荐 i 的得分 = u 与 i 之间的二跳路径强度（带归一化）。

输入：
    内置用户-商品交互（0/1），无任何特征
输出：
    用户间两跳权重、各商品得分、目标商品的路径明细
依赖：
    numpy、viz_png
对应课件：
    模块五第三章 第三章 §5 算法 5-5（图推荐与 LightGCN）

说明：
    LightGCN 公式：e_u^(k) = Σ_i A[u,i]/√(|N(u)||N(i)|) · e_i^(k-1)。
    本程序把它展开成「路径计数」形式，好处是每一步都能手算、能解释。
    真实模型里初始向量是训练出来的；这里只用图结构，所以得分就是路径强度。
"""

import numpy as np

USERS = ['u1', 'u2', 'u3', 'u4']
ITEMS = ['耳机', '音箱', '支架', '贴膜', '数据线']

# 设计意图：u2 与 u1 共享两件（最像）；u4 与 u1 共享一件；u3 与 u1 没有共享
A = np.array([
    [1, 1, 0, 0, 0],   # u1: 耳机, 音箱
    [1, 1, 1, 0, 0],   # u2: 耳机, 音箱, 支架（与 u1 最像）
    [0, 0, 0, 1, 1],   # u3: 贴膜, 数据线（与 u1 无关）
    [1, 0, 1, 0, 0],   # u4: 耳机, 支架（与 u1 共享耳机）
], dtype=float)

TARGET_U, TARGET_I = 0, 2      # 给 u1 推荐「支架」吗？


def degrees(A):
    return A.sum(axis=1), A.sum(axis=0)


def user_user_weight(A):
    """两跳后用户间权重（LightGCN 对称归一化的展开式）。

    w(u,v) = Σ_i A[u,i]A[v,i] / ( √|N(u)| √|N(v)| |N(i)| )
    含义：u 与 v 通过共同商品 i 相连；共同商品越热门，这条边越弱。
    """
    du, di = degrees(A)
    n_u = A.shape[0]
    W = np.zeros((n_u, n_u))
    for u in range(n_u):
        for v in range(n_u):
            acc = 0.0
            for i in range(A.shape[1]):
                if A[u, i] and A[v, i]:
                    acc += 1.0 / (np.sqrt(du[u]) * np.sqrt(du[v]) * di[i])
            W[u, v] = acc
    return W


def score_2hop(A, u):
    """给 u 的每个商品打分 = Σ_{v≠u} W[u,v] · A[v,i]。"""
    W = user_user_weight(A)
    s = np.zeros(A.shape[1])
    for v in range(A.shape[0]):
        if v == u:
            continue
        s += W[u, v] * A[v]
    return s, W


def paths(A, u, i):
    """列出 u 到 i 的二跳路径（经过哪个用户、共同商品是什么）。"""
    out = []
    for v in range(A.shape[0]):
        if v == u:
            continue
        shared = [ITEMS[k] for k in range(A.shape[1]) if A[u, k] and A[v, k]]
        if shared and A[v, i]:
            out.append((USERS[v], shared))
    return out


def demo():
    print('=== 图推荐：两层传播 = 数二跳路径 ===')
    print('用户 %d 个，商品 %d 个，交互 %d 条\n' % (
        A.shape[0], A.shape[1], int(A.sum())))

    print('--- 交互矩阵（1 = 买过）---')
    print('        ' + '  '.join('%4s' % x for x in ITEMS))
    for u, name in enumerate(USERS):
        print('%4s    ' % name + '  '.join('%4d' % v for v in A[u]))
    print()

    W = user_user_weight(A)
    print('--- 两跳后用户间权重（共同商品越多、越不热门，权重越大）---')
    print('        ' + '  '.join('%6s' % x for x in USERS))
    for u, name in enumerate(USERS):
        print('%4s    ' % name + '  '.join('%6.3f' % W[u, v] for v in range(len(USERS))))
    print()

    s, _ = score_2hop(A, TARGET_U)
    print('--- 给 %s 的商品打分（越高越该推）---' % USERS[TARGET_U])
    order = np.argsort(-s)
    for i in order:
        tag = '（已买过）' if A[TARGET_U, i] else ('（目标：待推荐）' if i == TARGET_I else '')
        print('  %-4s %.3f %s' % (ITEMS[i], s[i], tag))
    print()

    print('--- 目标《%s》的二跳路径明细 ---' % ITEMS[TARGET_I])
    ps = paths(A, TARGET_U, TARGET_I)
    if ps:
        for v, shared in ps:
            print('  %s --（共同买过 %s）-- %s --（买过 %s）--> 推荐' % (
                USERS[TARGET_U], '、'.join(shared), v, ITEMS[TARGET_I]))
        print('  共 %d 条路径（每个中间用户算 1 条）' % len(ps))
    else:
        print('  没有二跳路径——图推荐就无从下手（这正是冷启动问题）')
    print()

    print('--- 一层 vs 两层在解决不同问题 ---')
    print('  第一层（看你自己）：你买过耳机、音箱 → 只会推「和它们像的」')
    print('  第二层（看同好）：u2 也买过耳机，他还买了支架 → 支架被带进来')
    print('  这就是协同信号：你自己没买过支架，但和你像的人买了\n')

    print('结论：')
    print('  1) 图推荐不用任何商品特征，只用「谁买过谁」这张图。')
    print('  2) 二层传播的实质是「数二跳路径」——每条路径就是一个同好投的一票。')
    print('  3) LightGCN 的全部魔法 = 归一化邻居求和 + 多层叠加，没有激活函数。')
    print('  4) 层数不是越多越好：太深会让所有节点向量趋同（过平滑）。')
    return s, W


def save_png(path='visuals/ch4_rec_graph.png'):
    from viz_png import Canvas, RED, RED_L, GOLD, GOLD_L, INK, MUTED, LINE, lerp

    s, W = demo()
    cv = Canvas(1200, 740)
    cv.title('图推荐：两层传播 = 数「二跳路径」')

    # ---- 二分图：用户左列、商品右列 ----
    n_u, n_i = len(USERS), len(ITEMS)
    ux, ix = 130, 520
    uy0, iy0, dyy = 150, 130, 78
    uy = [uy0 + k * dyy for k in range(n_u)]
    iy = [iy0 + k * dyy for k in range(n_i)]

    for u in range(n_u):
        for i in range(n_i):
            if A[u, i]:
                cv.line(ux + 26, uy[u], ix - 30, iy[i], color=LINE, width=2)
    # 高亮：u1 -耳机- u2 -支架-
    cv.line(ux + 26, uy[0] - 6, ix - 30, iy[0] - 6, color=RED, width=3, dash=True)
    cv.line(ix - 30, iy[0] - 6, ux + 26, uy[1] - 6, color=RED, width=3, dash=True)
    cv.line(ux + 26, uy[1] + 6, ix - 30, iy[2] + 6, color=RED, width=3, dash=True)

    for u in range(n_u):
        cv.node(ux, uy[u], 26, GOLD_L if u == 0 else RED_L, GOLD if u == 0 else RED, 3 if u == 0 else 2)
        cv.text(ux, uy[u], USERS[u], 15, INK, True, 'mm')
    for i in range(n_i):
        tgt = (i == TARGET_I)
        cv.node(ix, iy[i], 30, GOLD_L if tgt else RED_L, GOLD if tgt else RED, 3 if tgt else 2)
        cv.text(ix, iy[i], ITEMS[i], 13, INK, True, 'mm')

    cv.text(60, 505, '左 = 用户，右 = 商品；虚线 = 一条二跳路径', 13, RED, True)
    cv.text(60, 528, 'u1 —（耳机）— u2 —（支架）— 推荐给 u1', 12, MUTED)
    cv.text(60, 551, 'u1 自己没买过支架，但同好 u2、u4 都买了', 12, MUTED)

    # ---- 左下：用户间权重 —— 只画非零项，避免密集表格压住其他元素 ----
    bx, by = 620, 120
    cv.text(bx, by - 16, '两跳后用户间权重（只列非零）', 14, INK, True)
    pairs = [(u, v, W[u, v]) for u in range(n_u) for v in range(u + 1, n_u) if W[u, v] > 0]
    for k, (u, v, w) in enumerate(pairs):
        y = by + 14 + k * 38
        cv.rect(bx, y, 250, 30, fill=lerp((250, 247, 246), RED, min(w, 1) * 0.5),
                outline=LINE, width=1, r=6)
        cv.text(bx + 10, y + 7, '%s ↔ %s' % (USERS[u], USERS[v]), 12, INK, True)
        cv.text(bx + 236, y + 7, '%.3f' % w, 13, RED, True, 'ra')
    cv.text(bx, by + 14 + len(pairs) * 38 + 4, '共同商品越热门，这条边越弱（÷|N(i)|）', 11, MUTED)

    # ---- 右下：推荐排序 ----
    ry = by + 14 + len(pairs) * 38 + 34
    cv.text(bx, ry - 16, '给 u1 的推荐排序', 14, INK, True)
    order = np.argsort(-s)
    for k, i in enumerate(order):
        y = ry + k * 36
        seen = A[0, i] == 1
        tgt = (i == TARGET_I)
        col = GOLD if tgt else (MUTED if seen else RED)
        cv.rect(bx, y, 250, 30, fill=GOLD_L if tgt else (255, 255, 255),
                outline=col, width=2 if tgt else 1, r=6)
        cv.text(bx + 10, y + 7, ITEMS[i], 12, INK, True)
        cv.text(bx + 236, y + 7, '%.3f' % s[i], 13, col, True, 'ra')
        tag = '已买过' if seen else ('待推荐 ★' if tgt else '可推荐')
        cv.text(bx + 130, y + 8, tag, 10, col, True)

    cv.text(60, 640, '第一层看你自己买过什么；第二层看「和你买过同样东西的人」买什么', 13, INK, True)
    cv.text(60, 668, '支架自己没买过，但同好买了 → 这就是「协同信号」', 12, GOLD, True)
    cv.text(620, 668, 'LightGCN 的魔法 = 归一化邻居求和 + 多层叠加，无激活函数', 11, MUTED)

    cv.text(60, 706, '数据：4 用户 x 5 商品、9 条交互；程序 ch3_rec_graph.py（路径计数写法，便于手算）', 11, MUTED)
    return cv.save(path)


if __name__ == '__main__':
    demo()
