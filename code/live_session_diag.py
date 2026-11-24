#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_session_diag.py -- 直播场次诊断：净转化、退货后毛利、达人 ROI

用途：
    用三场典型直播（头部达人 / 品牌自播 / 腰尾 KOC 组合）的真实财务结构，
    演示一件最容易被忽略的事：
      GMV 最高的那场，可能是唯一亏钱的那场。

    诊断三件事：
      1) 毛转化率 vs 净转化率（退货算进去之后）
      2) 退货后毛利（扣商品成本、坑位费、佣金、人力）
      3) 每千次观看毛利（GPM-gross / GPM-net），用于跨场次横向比较

输入：
    内置三场直播数据（示意数据，非真实企业数据）
输出：
    每场的转化、毛利、ROI、诊断结论与改进方向
依赖：
    viz_png
对应课件：
    模块五第三章 §8 跨境电商直播（8.3 直播类型 / 场次复盘）
"""

# 商品成本率（占成交额的比重）；跨境一般 35%–50%，这里取 40%
COGS_RATE = 0.40

# 每场：名称、场观、成交单数、客单价、退货单数、坑位费、佣金率、人力成本
SESSIONS = [
    {'name': 'A 头部达人', 'views': 120000, 'orders': 2400, 'aov': 89.0,
     'returns': 720, 'slot_fee': 60000.0, 'commission': 0.20, 'labor': 5000.0},
    {'name': 'B 品牌自播', 'views': 30000, 'orders': 900, 'aov': 79.0,
     'returns': 180, 'slot_fee': 0.0, 'commission': 0.00, 'labor': 18000.0},
    {'name': 'C 腰尾KOC组合', 'views': 60000, 'orders': 1200, 'aov': 69.0,
     'returns': 300, 'slot_fee': 8000.0, 'commission': 0.15, 'labor': 3000.0},
]


def diagnose(s):
    """算一场直播的完整财务结构。"""
    gmv = s['orders'] * s['aov']
    ret_amt = s['returns'] * s['aov']
    net_gmv = gmv - ret_amt                      # 退货后成交额
    commission_fee = gmv * s['commission']       # 佣金按成交额算（退货一般不退佣）
    cogs = net_gmv * COGS_RATE                   # 商品成本按净成交额算
    cost_total = cogs + s['slot_fee'] + commission_fee + s['labor']
    profit = net_gmv - cost_total                # 退货后毛利
    invest = s['slot_fee'] + commission_fee + s['labor']   # 除了货之外的投入
    roi = profit / invest if invest else float('nan')

    return dict(
        name=s['name'], views=s['views'],
        gmv=gmv, ret_amt=ret_amt, net_gmv=net_gmv,
        cvr=s['orders'] / s['views'],            # 毛转化（按成交单 / 场观）
        net_cvr=(s['orders'] - s['returns']) / s['views'],   # 净转化
        ret_rate=s['returns'] / s['orders'],     # 退货率
        cogs=cogs, slot=s['slot_fee'], comm=commission_fee, labor=s['labor'],
        cost_total=cost_total, profit=profit, invest=invest, roi=roi,
        gpm_gross=gmv / s['views'] * 1000,       # 每千次观看成交额
        gpm_net=profit / s['views'] * 1000,      # 每千次观看毛利（真正可比的指标）
    )


def demo():
    print('=== 直播场次诊断：GMV 高不等于赚钱 ===')
    print('商品成本率 %.0f%%；佣金按成交额计（退货通常不返佣）\n' % (COGS_RATE * 100))

    rows = [diagnose(s) for s in SESSIONS]

    print('--- 一、转化 ---')
    print('%-14s %9s %8s %8s %8s' % ('场次', '场观', '毛转化', '退货率', '净转化'))
    for r in rows:
        print('%-14s %9s %7.2f%% %7.1f%% %7.2f%%' % (
            r['name'], '{:,}'.format(r['views']),
            r['cvr'] * 100, r['ret_rate'] * 100, r['net_cvr'] * 100))
    print()

    print('--- 二、钱去哪了 ---')
    print('%-14s %10s %10s %10s %10s %10s' % ('场次', 'GMV', '退货额', '坑位费', '佣金', '人力'))
    for r in rows:
        print('%-14s %10.0f %10.0f %10.0f %10.0f %10.0f' % (
            r['name'], r['gmv'], r['ret_amt'], r['slot'], r['comm'], r['labor']))
    print()

    print('--- 三、到底赚不赚 ---')
    print('%-14s %10s %10s %10s %9s' % ('场次', '净成交额', '退货后毛利', '总成本', 'ROI'))
    for r in rows:
        flag = '  <-- 亏' if r['profit'] < 0 else ''
        print('%-14s %10.0f %10.0f %10.0f %9.2f%s' % (
            r['name'], r['net_gmv'], r['profit'], r['cost_total'], r['roi'], flag))
    print()

    print('--- 四、每千次观看（跨场次可比） ---')
    print('%-14s %12s %12s' % ('场次', 'GPM-毛', 'GPM-净'))
    for r in rows:
        print('%-14s %12.1f %12.1f' % (r['name'], r['gpm_gross'], r['gpm_net']))
    print()

    # 排名与结论
    by_gmv = sorted(rows, key=lambda r: -r['gmv'])
    by_profit = sorted(rows, key=lambda r: -r['profit'])
    print('GMV 排名：' + ' > '.join('%s(%.0f)' % (r['name'], r['gmv']) for r in by_gmv))
    print('毛利排名：' + ' > '.join('%s(%.0f)' % (r['name'], r['profit']) for r in by_profit))
    print()

    print('结论：')
    print('  1) GMV 第一的 A（头部达人）毛利 %.0f，是唯一亏钱的场次——' % rows[0]['profit'])
    print('     坑位费 %.0f 是固定成本，卖不动也照付；退货率 %.0f%% 又把成交额吃掉一块。' % (
        rows[0]['slot'], rows[0]['ret_rate'] * 100))
    print('  2) GMV 最低的 B（品牌自播）毛利 %.0f 最高——没有坑位费与佣金，' % rows[1]['profit'])
    print('     成本是可控的人力，且客户与会员沉淀在自己手里。')
    print('  3) 只报 GMV 的复盘会，会得出与事实相反的结论。必须看「净转化 + 退货后毛利」。')
    print('  4) 跨场次比较用 GPM-净（每千次观看毛利），不要用 GPM-毛。')
    print('  5) 若退货率超过 30%，先修产品与描述，再谈加投放——否则每多卖一单可能多亏一单。')
    return rows


def save_png(path='visuals/live_session_diag.png'):
    from viz_png import Canvas, RED, GOLD, INK, MUTED, LINE, GOLD_L, RED_L
    rows = demo()
    cv = Canvas(1180, 660)
    cv.title('直播场次诊断：GMV 最高的那场，是唯一亏钱的')

    # 左：毛利柱状（含负值）
    ox, oy, w, h = 120, 150, 480, 340
    maxabs = max(abs(r['profit']) for r in rows) * 1.15
    zero_y = oy + h / 2
    cv.rect(ox, oy, w, h, fill=(252, 250, 249), outline=LINE, width=1, r=6)
    cv.line(ox, zero_y, ox + w, zero_y, color=MUTED, width=2)
    bw = w / (len(rows) * 1.7)
    for i, r in enumerate(rows):
        x = ox + w * (i + 0.5) / len(rows) - bw / 2
        val = r['profit']
        bh = abs(val) / maxabs * (h / 2 - 18)
        y = zero_y - bh if val >= 0 else zero_y
        col = GOLD if val >= 0 else RED
        cv.rect(x, y, bw, max(bh, 2), fill=col, outline=None, width=0, r=4)
        cv.text(x + bw / 2, y - 18 if val >= 0 else y + bh + 6,
                '%.0f' % val, 12, col, True, 'ma')
        cv.text(x + bw / 2, oy + h + 12, r['name'].split(' ')[0], 11, INK, True, 'ma')
        cv.text(x + bw / 2, oy + h + 30, r['name'].split(' ')[1], 9.5, MUTED, False, 'ma')
    cv.text(ox, oy - 22, '退货后毛利（元）', 13, INK, True)

    # 右：净转化 vs 毛转化
    px = 680
    cv.text(px, oy - 22, '毛转化 vs 净转化', 13, INK, True)
    for i, r in enumerate(rows):
        y = oy + 16 + i * 92
        cv.text(px, y, r['name'], 11.5, INK, True)
        # 毛转化条
        cv.rect(px, y + 18, int(r['cvr'] * 9000), 15, fill=LINE, outline=None, width=0, r=3)
        cv.text(px + int(r['cvr'] * 9000) + 8, y + 19, '毛 %.2f%%' % (r['cvr'] * 100), 9.5, MUTED)
        # 净转化条
        cv.rect(px, y + 38, int(r['net_cvr'] * 9000), 15, fill=GOLD, outline=None, width=0, r=3)
        cv.text(px + int(r['net_cvr'] * 9000) + 8, y + 39, '净 %.2f%%' % (r['net_cvr'] * 100), 9.5, GOLD, True)
        cv.text(px + 230, y + 58, '退货率 %.0f%%' % (r['ret_rate'] * 100), 9.5, RED)

    cv.text(60, 552, '只报 GMV 的复盘会得出相反结论：A 的 GMV 最高（%.0f），毛利最低（%.0f）' % (
        rows[0]['gmv'], rows[0]['profit']), 11.5, RED, True)
    cv.text(60, 578, '坑位费是固定成本，退货率把成交额再吃一刀——两个因素叠起来就是亏损', 10.5, MUTED)
    cv.text(60, 616, '数据：三场示意场次；商品成本率 40%%；程序 live_session_diag.py', 10.5, MUTED)
    return cv.save(path)


if __name__ == '__main__':
    demo()
