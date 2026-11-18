#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs_dialog_state.py -- 多轮对话状态机（退货流程）

用途：
    用 4 个槽位、5 个状态、3 个分支模拟一次退货对话，
    演示「多轮怎么记住说到哪了」，以及客户不配合时怎么兜底。

输入：
    内置的 6 轮客户话语（含一次不配合）
输出：
    每轮的状态、槽位填充情况、以及最终走向（完成 / 升级）
依赖：
    viz_png
对应课件：
    模块六 §6 自动回复与多轮对话设计 / §10 类一算法
"""

SLOTS = ['订单号', '退货原因', '凭证', '期望方案']
BRANCHES = ['期限内可退', '超期需审批', '已使用不可退']
UPGRADE = ['情绪激烈', '金额超限', '法律风险', '反复未解决', '要求人工']

# 客户话语：(文本, 提供了哪个槽位或触发什么)
TURNS = [
    ('我要退货', None),
    ('订单号 A12345', '订单号'),
    ('东西坏了', '退货原因'),
    ('照片我发你', '凭证'),
    ('退款就行', '期望方案'),
    ('你们是不是不想退，我要投诉', '!升级'),
]


def parse(turn):
    """极简槽位抽取：用关键词判断这一轮提供了什么。"""
    text, hint = turn
    if hint and hint.startswith('!'):
        return None, hint[1:]
    return hint, None


def run():
    state = 'S1_识别意图'
    filled = {s: False for s in SLOTS}
    trace = []
    branch = None
    escalated = None

    for i, turn in enumerate(TURNS, 1):
        slot, trig = parse(turn)
        if trig:
            escalated = trig
            state = 'S5_转人工'
            trace.append((i, turn[0], state, dict(filled), '触发升级：' + trig))
            break
        if slot:
            filled[slot] = True
        # 状态推进
        if not filled['订单号']:
            state = 'S2_收订单号'
        elif not filled['退货原因']:
            state = 'S3_问原因'
        elif not filled['凭证']:
            state = 'S3_问原因'
        elif not filled['期望方案']:
            state = 'S4_判定方案'
        else:
            state = 'S4_判定方案'
            branch = '期限内可退'
        trace.append((i, turn[0], state, dict(filled), ''))

    done = all(filled.values()) and escalated is None
    return trace, filled, branch, escalated, done


def demo():
    print('=== 多轮对话状态机（退货流程）===')
    print('槽位 %d 个：%s' % (len(SLOTS), '、'.join(SLOTS)))
    print('升级条件 %d 个：%s\n' % (len(UPGRADE), '、'.join(UPGRADE)))

    trace, filled, branch, esc, done = run()
    print('--- 逐轮状态 ---')
    for i, text, state, fl, note in trace:
        got = [k for k, v in fl.items() if v]
        print('  T%d %-18s -> %-14s 已填 %d/%d %s' % (
            i, text[:16], state, len(got), len(SLOTS), note))
    print()

    print('--- 最终槽位 ---')
    for s in SLOTS:
        print('  %-6s %s' % (s, 'OK' if filled[s] else 'MISSING'))
    print()

    print('--- 走向 ---')
    if esc:
        print('  转人工（触发：%s）——这是设计内的结果，不是失败' % esc)
        print('  必须交接：完整对话 + 已填槽位 + 情绪判断')
    elif done:
        print('  判定分支：%s' % branch)
        print('  下一步：登记 + 给退货地址 + 承诺退款时效')
    else:
        print('  槽位不全，继续追问或转人工')
    print()

    print('结论：')
    print('  1) 状态机的作用是「不重复问、不丢信息」——每轮只推进一格。')
    print('  2) 客户在第 6 轮情绪爆发时，正确动作是立刻升级，而不是继续讲政策。')
    print('  3) 升级时要交接已填槽位，否则客户要把前面说过的话再说一遍。')
    print('  4) 兜底分支必须存在：没有它，对话会在「客户不配合」时断掉。')
    return trace, filled, esc


def save_png(path='visuals/cs_dialog_state.png'):
    from viz_png import Canvas, RED, GOLD, INK, MUTED, LINE, GOLD_L, RED_L
    trace, filled, esc = demo()
    cv = Canvas(1150, 640)
    cv.title('多轮对话状态机：每轮只推进一格')

    # 上：五个状态
    xs = [110, 330, 550, 770, 990]
    names = ['S1 识别意图', 'S2 收订单号', 'S3 问原因', 'S4 判定方案', 'S5 转人工']
    for i, (x, n) in enumerate(zip(xs, names)):
        col = GOLD if i == 4 else RED
        cv.rect(x - 85, 110, 170, 54, fill=GOLD_L if i == 4 else RED_L, outline=col, width=2, r=8)
        cv.text(x, 137, n, 12, INK, True, 'mm')
        if i < 4:
            cv.arrow(x + 88, 137, xs[i + 1] - 88, 137, color=MUTED, width=2)

    # 中：槽位填充进度
    cv.text(80, 200, '槽位填充进度', 14, INK, True)
    for i, s in enumerate(SLOTS):
        y = 232 + i * 46
        ok = filled[s]
        cv.text(80, y + 4, s, 12, INK, True)
        cv.rect(170, y, 260, 26, fill=(246, 243, 241), outline=LINE, width=1, r=5)
        cv.rect(170, y, 260 if ok else 0, 26, fill=GOLD if ok else RED, outline=None, width=0, r=5)
        cv.text(444, y + 5, 'OK' if ok else '缺', 12, GOLD if ok else RED, True)

    # 右：逐轮轨迹
    cv.text(560, 200, '逐轮轨迹', 14, INK, True)
    for i, (turn, text, state, fl, note) in enumerate(trace):
        y = 228 + i * 56
        hi = '升级' in state
        cv.rect(560, y, 520, 46, fill=GOLD_L if hi else (255, 255, 255),
                outline=GOLD if hi else LINE, width=2 if hi else 1, r=8)
        cv.text(572, y + 6, 'T%d' % turn, 11, MUTED, True)
        cv.text(608, y + 5, text[:20], 11, INK)
        cv.text(608, y + 24, state, 10, GOLD if hi else MUTED, True)
        if note:
            cv.text(1000, y + 24, note[:12], 9, GOLD, True)

    cv.text(80, 470, '客户第 6 轮情绪爆发 → 立刻升级，不再讲政策', 12, GOLD, True)
    cv.text(80, 496, '升级时交接已填槽位，客户不必重讲', 11, MUTED)
    cv.text(60, 600, '数据：4 槽位 / 5 状态 / 6 轮对话；程序 cs_dialog_state.py', 11, MUTED)
    return cv.save(path)


if __name__ == '__main__':
    demo()
