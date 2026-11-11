#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ch3_rec_netflix.py -- 推荐算法：协同过滤（极简版）

用途：
    用 6 个用户 x 6 部电影的极小评分矩阵，演示协同过滤怎么「补空格」。
    同时用这个例子讲算法的提出背景：Netflix Prize（2006-2009）。

输入：
    内置评分矩阵（1-5 分，0 表示没看过）
输出：
    UserCF 与 ItemCF 对两个空格子的预测分，以及留一法 RMSE
依赖：
    numpy、viz_png
对应课件：
    模块五第三章 第三章 §5 算法 5-1（推荐算法与 Netflix Prize）

背景提示：
    2006 年 Netflix 悬赏 100 万美元，要求把推荐准确率（RMSE）提升 10%。
    这场比赛让「矩阵分解」成为主流，也让推荐系统从「规则」变成「模型」。
"""

import numpy as np

USERS = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6']
ITEMS = ['星战', '教父', '玩具', '爱乐', '谍影', '魔戒']

# 0 = 未评分
R = np.array([
    [5, 4, 1, 0, 4, 5],   # u1  科幻/动作爱好者
    [4, 5, 0, 1, 5, 4],   # u2
    [1, 0, 5, 4, 1, 2],   # u3  动画/音乐剧爱好者
    [0, 1, 4, 5, 0, 1],   # u4
    [5, 4, 1, 0, 4, 0],   # u5
    [2, 1, 4, 5, 1, 3],   # u6
], dtype=float)

TARGETS = [(0, 3), (2, 1)]   # 要预测的两个空格子


def cosine(a, b):
    """只按两人都评过的维度算余弦相似度。"""
    mask = (a > 0) & (b > 0)
    if mask.sum() < 2:
        return 0.0
    av, bv = a[mask], b[mask]
    d = np.linalg.norm(av) * np.linalg.norm(bv)
    return float(av @ bv / d) if d else 0.0


def user_cf(R, u, i, k=2):
    """UserCF：找最像的 k 个用户，按相似度加权平均他们的评分。"""
    sims = []
    for v in range(R.shape[0]):
        if v == u or R[v, i] == 0:
            continue
        sims.append((cosine(R[u], R[v]), v))
    sims.sort(reverse=True)
    top = sims[:k]
    if not top:
        return None, []
    num = sum(s * R[v, i] for s, v in top)
    den = sum(abs(s) for s, _ in top)
    return (num / den if den else None), top


def item_cf(R, u, i, k=2):
    """ItemCF：找最像的 k 个电影（按用户评分向量），加权平均目标用户对它们的评分。"""
    sims = []
    for j in range(R.shape[1]):
        if j == i or R[u, j] == 0:
            continue
        sims.append((cosine(R[:, i], R[:, j]), j))
    sims.sort(reverse=True)
    top = sims[:k]
    if not top:
        return None, []
    num = sum(s * R[u, j] for s, j in top)
    den = sum(abs(s) for s, _ in top)
    return (num / den if den else None), top


def holdout_rmse(R, method):
    """留一法：把每个已知评分当未知预测一次，算 RMSE。"""
    errs = []
    for u in range(R.shape[0]):
        for i in range(R.shape[1]):
            if R[u, i] == 0:
                continue
            truth = R[u, i]
            R2 = R.copy()
            R2[u, i] = 0
            pred, _ = method(R2, u, i)
            if pred is not None:
                errs.append((pred - truth) ** 2)
    return float(np.sqrt(np.mean(errs))) if errs else float('nan')


def demo():
    print('=== 推荐算法：协同过滤（6 用户 x 6 电影）===')
    print('背景：2006 年 Netflix Prize 悬赏 100 万美元，要求 RMSE 提升 10%\\n')
    print('评分矩阵（0 = 没看过）：')
    print('        ' + '  '.join('%4s' % it for it in ITEMS))
    for u, name in enumerate(USERS):
        print('%4s    ' % name + '  '.join('%4d' % v for v in R[u]))
    print()

    print('--- 对两个空格子的预测 ---')
    rows = []
    for (u, i) in TARGETS:
        pu, nu = user_cf(R, u, i)
        pi, ni = item_cf(R, u, i)
        rows.append((USERS[u], ITEMS[i], pu, pi))
        print('  %s 对《%s》：UserCF %.2f ／ ItemCF %.2f' % (USERS[u], ITEMS[i], pu, pi))
        print('     UserCF 参考：' + '、'.join('%s(相似度%.2f)' % (USERS[v], s) for s, v in nu))
        print('     ItemCF 参考：' + '、'.join('《%s》(相似度%.2f)' % (ITEMS[j], s) for s, j in ni))
    print()

    r_u = holdout_rmse(R, user_cf)
    r_i = holdout_rmse(R, item_cf)
    print('--- 留一法 RMSE（越小越好）---')
    print('  UserCF %.3f   ItemCF %.3f' % (r_u, r_i))
    print()

    print('结论：')
    print('  1) 协同过滤只靠评分矩阵，不需要知道电影内容——这是它最大的优势。')
    print('  2) 但矩阵极其稀疏时，找相似用户/相似物品都很难（本表有 11 个空格子）。')
    print('  3) Netflix Prize 的真正遗产不是某个算法，而是「用 RMSE 量化推荐质量」这件事。')
    print('  4) 后来的教训：RMSE 更准不等于用户更爱看——离线指标与线上效果会背离。')
    return rows, r_u, r_i


def save_png(path='visuals/ch4_rec_netflix.png'):
    from viz_png import Canvas, RED, RED_L, GOLD, GOLD_L, INK, MUTED, LINE, PAPER, lerp

    rows, r_u, r_i = demo()
    cv = Canvas(1200, 720)
    cv.title('推荐算法：协同过滤怎么「补空格」（Netflix Prize 的背景）')

    # 左：评分矩阵热力图
    ox, oy, cw, ch = 150, 130, 92, 62
    cv.text(ox - 10, oy - 34, '评分矩阵（空格 = 没看过）', 16, INK, True)
    for j, it in enumerate(ITEMS):
        cv.text(ox + j * cw + cw / 2, oy - 12, it, 13, MUTED, False, 'mm')
    for u, name in enumerate(USERS):
        cv.text(ox - 18, oy + u * ch + ch / 2, name, 14, INK, True, 'rm')
        for j in range(len(ITEMS)):
            x, y = ox + j * cw, oy + u * ch
            v = R[u, j]
            if v == 0:
                cv.rect(x + 2, y + 2, cw - 4, ch - 4, fill=(243, 240, 238), outline=LINE, width=1, r=4)
                cv.text(x + cw / 2, y + ch / 2, '—', 16, LINE, False, 'mm')
            else:
                t = min(v / 5.0, 1.0)
                cv.rect(x + 2, y + 2, cw - 4, ch - 4, fill=lerp((250, 247, 246), RED, t * 0.85),
                        outline=None, width=0, r=4)
                cv.text(x + cw / 2, y + ch / 2, str(int(v)), 17, (255, 255, 255) if t > 0.5 else INK,
                        True, 'mm')

    # 标注要预测的格子
    for (u, i) in TARGETS:
        x, y = ox + i * cw, oy + u * ch
        cv.rect(x, y, cw, ch, fill=None, outline=GOLD, width=3, r=0)

    cv.text(ox, oy + 6 * ch + 20, '金框 = 要预测的两个空格子', 14, GOLD, True)
    cv.text(ox, oy + 6 * ch + 46, 'u1 看不看《爱乐》？u3 看不看《教父》？', 13, MUTED)
    cv.text(ox, oy + 6 * ch + 70, '注：矩阵只有 6x6，真实场景是百万级且 99% 是空格', 12, MUTED)

    # 右：预测结果
    px = 790
    cv.text(px, 130, '两种协同过滤的预测', 16, INK, True)
    yy = 170
    for (un, item, pu, pi) in rows:
        cv.rect(px, yy, 370, 96, fill=(255, 255, 255), outline=LINE, width=1, r=8)
        cv.text(px + 16, yy + 14, '%s 对《%s》' % (un, item), 15, INK, True)
        cv.text(px + 16, yy + 42, 'UserCF（找相似的人）', 13, MUTED)
        cv.text(px + 300, yy + 42, '%.2f' % pu, 17, RED, True, 'ra')
        cv.text(px + 16, yy + 68, 'ItemCF（找相似的片）', 13, MUTED)
        cv.text(px + 300, yy + 68, '%.2f' % pi, 17, GOLD, True, 'ra')
        yy += 112

    cv.text(px, yy + 6, '留一法 RMSE（越小越好）', 15, INK, True)
    cv.text(px, yy + 34, 'UserCF %.3f    ItemCF %.3f' % (r_u, r_i), 16, RED, True)
    cv.text(px, yy + 66, 'RMSE 更准 ≠ 用户更爱看', 14, GOLD, True)
    cv.text(px, yy + 90, '这是 Netflix Prize 之后最重要的教训', 12, MUTED)

    cv.text(30, 680, '数据：6x6 评分矩阵（教学用极小规模）；程序 ch3_rec_netflix.py', 12, MUTED)
    return cv.save(path)


if __name__ == '__main__':
    demo()
