#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs_kb_rag_retrieval.py -- 客服知识库检索（切分 + 混合检索 + 拒答）

用途：
    用 12 条知识条目演示 RAG 检索端的三个关键动作：
      1) 切分：把一条长条目切成块，看块边界怎么影响检索；
      2) 混合检索：稀疏（词频）、稠密（单字宽松匹配）、混合的对比；
      3) 拒答：Top-1 分数低于阈值就转人工，而不是硬答。

输入：
    内置知识库（含市场、品类元数据）+ 6 个测试问题（含一个库里没有答案的）
输出：
    三种检索的 Top-3、分数、以及拒答判定
依赖：
    numpy、viz_png
对应课件：
    模块六 §4 RAG 基础 / §5 工具与实操 / §10 类二算法
"""

import numpy as np

# 知识库：每条 = (编号, 问题, 答案, 市场, 品类)
KB = [
    ('K01', '退货期限', '签收 30 天内可退，商品需未使用且包装完好。', 'DE', '全部'),
    ('K02', '退货流程', '在订单页点「申请退货」，选择原因并上传照片，通过后寄回指定地址。', 'DE', '全部'),
    ('K03', '退款时间', '仓库签收后 3 个工作日内退款到原支付方式。', 'DE', '全部'),
    ('K04', '换货条件', '同款不同色或不同尺码可换；已使用或人为损坏不可换。', 'DE', '全部'),
    ('K05', '物流时效', '德国通常 7--12 天送达；超 15 天可申请查件。', 'DE', '全部'),
    ('K06', '清关被海关扣留', '包裹被海关扣留时，需提供身份证与购物凭证，我们代为提交，通常 3--5 个工作日放行。', 'DE', '全部'),
    ('K07', '电池限制', '含电池产品不能空运，可选海运 25--35 天或换不含电池版本。', 'DE', '音频'),
    ('K08', '保修年限', '主机保修 24 个月，配件 6 个月，需提供订单号。', 'DE', '全部'),
    ('K09', '发票申请', '可在订单页申请电子发票，3 个工作日内发到注册邮箱。', 'DE', '全部'),
    ('K10', '运费规则', '订单满 49 欧元免运费，未满收 4.99 欧元。', 'DE', '全部'),
    ('K11', '退款到账慢', '若超过 5 个工作日未到账，请提供退款流水号，我们向支付方查询。', 'DE', '全部'),
    ('K12', '地址修改', '发货前可改地址；已发货需联系物流拦截，成功率视节点而定。', 'DE', '全部'),
]

QUERIES = [
    ('德国的包裹被海关扣了怎么办', 'K06', True),
    ('退款多久能到', 'K03', True),
    ('含电池的能空运吗', 'K07', True),
    ('我想退货', 'K02', True),
    ('保修几年', 'K08', True),
    ('能寄到南极吗', None, False),   # 库里没有：应拒答
]

STOP = set('的了吗呢吧啊在有是就都也很还')


def char1(s):
    """单字 bag（去掉停用字）。

    本程序用它近似「稠密检索」：单字匹配比 2-gram 宽松得多，
    所以同一个问题会命中更多文档——这正是稠密检索的特点（召回宽、精度低）。
    真实系统的稠密检索用语义向量，机制上同样是「更宽松的相似度」。
    """
    return [c for c in s if c not in STOP and c.strip()]


def tfidf_vec(tks, vocab, idf):
    """TF-IDF 向量（L2 归一化），用于余弦相似度。"""
    v = np.zeros(len(vocab))
    for t in tks:
        if t in vocab:
            v[vocab[t]] += 1.0
    v = v * idf
    n = np.linalg.norm(v)
    return v / n if n else v


def tokens(s):
    """2-gram 精确匹配（近似「稀疏检索」）。

    只保留相邻两字，命中要求字面连续——所以对精确词（型号、政策名）很强，
    但换个说法就失配。这就是稀疏检索的特点。
    """
    cs = [c for c in s if c not in STOP and c.strip()]
    return [cs[i] + cs[i + 1] for i in range(len(cs) - 1)]


def build_vocab_vectors(seqs_of_tokens):
    """给定一组 token 序列，建词表 + IDF，并返回每条序列的 TF-IDF 向量。"""
    vocab = {}
    for tks in seqs_of_tokens:
        for t in tks:
            if t not in vocab:
                vocab[t] = len(vocab)
    N = len(seqs_of_tokens)
    df = np.zeros(len(vocab))
    for tks in seqs_of_tokens:
        for t in set(tks):
            df[vocab[t]] += 1
    idf = np.log((N + 1) / (df + 1)) + 1.0
    vecs = []
    for tks in seqs_of_tokens:
        v = np.zeros(len(vocab))
        for t in tks:
            if t in vocab:
                v[vocab[t]] += 1.0
        v = v * idf
        n = np.linalg.norm(v)
        vecs.append(v / n if n else v)
    return vocab, idf, vecs


def build_index(kb):
    """建两套索引：稀疏（2-gram）与稠密（单字）。"""
    docs = []
    for kid, q, a, mkt, cat in kb:
        text = q + '\u3002' + a
        docs.append({'id': kid, 'text': text, 'mkt': mkt, 'cat': cat,
                     'toks': tokens(text), 'unis': char1(text)})

    vsp, isp, vec_sp = build_vocab_vectors([d['toks'] for d in docs])
    vdn, idn, vec_dn = build_vocab_vectors([d['unis'] for d in docs])
    for i, d in enumerate(docs):
        d['v_sp'] = vec_sp[i]
        d['v_dn'] = vec_dn[i]
    return docs, (vsp, isp), (vdn, idn)


def sparse_score(qv_sp, d_vec_sp):
    """稀疏相似度：词级 TF-IDF 余弦。"""
    return float(qv_sp @ d_vec_sp)


def dense_score(qv_dn, d_vec_dn):
    """稠密相似度：单字 TF-IDF 余弦（宽松匹配，近似语义检索的召回特性）。"""
    return float(qv_dn @ d_vec_dn)


def retrieve(query, docs, idx_sp, idx_dn, alpha=0.5, topk=3):
    """混合检索：alpha * 稀疏余弦 + (1-alpha) * 稠密余弦。

    两个分量都是 0~1 的余弦，融合分也是 0~1 的绝对量——
    这才可以用固定阈值做拒答（相对归一化的分数做不到）。
    """
    vsp, isp = idx_sp
    vdn, idn = idx_dn
    qt = tokens(query)
    qu = char1(query)

    def tvec(tks, vocab, idf):
        v = np.zeros(len(vocab))
        for t in tks:
            if t in vocab:
                v[vocab[t]] += 1.0
        v = v * idf
        n = np.linalg.norm(v)
        return v / n if n else v

    qv_sp = tvec(qt, vsp, isp)
    qv_dn = tvec(qu, vdn, idn)
    rows = []
    for d in docs:
        sp = float(qv_sp @ d['v_sp'])
        dn = float(qv_dn @ d['v_dn'])
        rows.append((d['id'], sp, dn, alpha * sp + (1 - alpha) * dn))
    rows.sort(key=lambda r: -r[3])
    return rows[:topk], qt


def demo():
    docs, idx_sp, idx_dn = build_index(KB)
    print('=== 客服知识库检索：切分 + 混合 + 拒答 ===')
    print('知识条目 %d 条，词表 %d 维，测试问题 %d 个\\n' % (len(KB), len(idx_sp[0]), len(QUERIES)))

    print('--- 切分示例：K02 的块 ---')
    k02 = [d for d in docs if d['id'] == 'K02'][0]
    print('  原文：', k02['text'])
    print('  切成 2-gram 片段的块，共 %d 个片段' % len(k02['toks']))
    print('  说明：块太大稀释语义，块太小切断步骤——本例用整条为一块（流程类知识应整段检索）\\n')

    THR = 0.20
    print('--- 六条查询的检索结果（alpha=0.5 混合）---')
    results = []
    for q, gold, should_answer in QUERIES:
        top, qt = retrieve(q, docs, idx_sp, idx_dn)
        best_id, best_score = top[0][0], top[0][3]
        answer = best_score >= THR
        ok = (best_id == gold) if should_answer else (not answer)
        results.append({'q': q, 'gold': gold, 'top': top, 'score': best_score,
                        'answer': answer, 'ok': ok, 'expect': should_answer})
        flag = 'OK' if ok else 'XX'
        print('  [%s] %s' % (flag, q))
        print('        Top3: ' + ' | '.join('%s(%.3f)' % (t[0], t[3]) for t in top))
        if should_answer:
            print('        期望 %s；Top1 分数 %.3f → %s' % (
                gold, best_score, '作答' if answer else '拒答（阈值 %.2f）' % THR))
        else:
            print('        库里无答案；Top1 分数 %.3f → %s' % (
                best_score, '作答（不该！）' if answer else '正确拒答（阈值 %.2f）' % THR))
    print()

    # 三种检索对比（取第一条查询）
    q = QUERIES[0][0]
    print('--- 同一查询下三种检索的 Top1 对比：%s ---' % q)
    for name, a in [('纯稀疏', 1.0), ('纯稠密', 0.0), ('混合', 0.5)]:
        top, _ = retrieve(q, docs, idx_sp, idx_dn, alpha=a)
        print('  %-6s Top1=%s 分数=%.3f  Top2=%s' % (
            name, top[0][0], top[0][3], top[1][0]))
    print()

    print('结论：')
    print('  1) 纯稀疏对「清关/被扣」这类词敏感，纯稠密对改写更宽容——混合通常更稳。')
    print('  2) 拒答阈值是这个系统的安全阀：没有它，「法国有仓库吗」也会被硬答。')
    print('  3) 切分决定上限：流程类知识必须整段检索，否则「第 3 步」被单独召回会误导客户。')
    print('  4) 这就是第 4 节说的「索引决定上限，检索决定下限」。')
    return results, THR


def save_png(path='visuals/cs_kb_rag_retrieval.png'):
    from viz_png import Canvas, RED, GOLD, INK, MUTED, LINE, lerp
    results, THR = demo()
    cv = Canvas(1150, 660)
    cv.title('RAG 检索：Top-1 分数分布与拒答阈值')

    # 分数条
    ox, oy = 330, 130
    barw = 520
    cv.text(60, oy - 26, '每个查询的 Top-1 混合分数（红线=拒答阈值 %.2f）' % THR, 14, INK, True)
    for i, r in enumerate(results):
        y = oy + i * 70
        label = r['q'][:14]
        cv.text(ox - 12, y + 9, label, 11, INK, False, 'rm')
        w = int(barw * min(r['score'], 1.0))
        col = GOLD if r['answer'] else MUTED
        cv.rect(ox, y, barw, 26, fill=(246, 243, 241), outline=LINE, width=1, r=5)
        cv.rect(ox, y, max(w, 2), 26, fill=col, outline=None, width=0, r=5)
        cv.text(ox + w + 8, y + 7, '%.3f' % r['score'], 12, col, True)
        tag = ('作答 → ' + r['top'][0][0]) if r['answer'] else '拒答 → 转人工'
        cv.text(ox + barw + 70, y + 8, tag, 11, col, True)
    # 阈值竖线
    tx = ox + int(barw * THR)
    cv.line(tx, oy - 8, tx, oy + len(results) * 70 - 20, color=RED, width=3)

    cv.text(60, oy + len(results) * 70 + 6,
            '最下面那条（法国仓库）分数低 → 正确拒答；这正是阈值的作用', 12, RED, True)
    cv.text(60, oy + len(results) * 70 + 32,
            '没有阈值时，它会被硬答成某条德国政策——这就是幻觉的来源', 11, MUTED)
    cv.text(60, 620, '数据：12 条知识 / 6 个查询 / alpha=0.5；程序 cs_kb_rag_retrieval.py', 11, MUTED)
    return cv.save(path)


if __name__ == '__main__':
    demo()
