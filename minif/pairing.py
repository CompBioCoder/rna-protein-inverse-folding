# -*- coding: utf-8 -*-
"""碱基配对：检测 + RNA 专属评价指标。

为什么需要这个：
序列恢复率在 RNA 上有个尴尬——字母表只有 4 个，瞎猜就 25%，
按最常见碱基猜大概 27-30%。这条基准线太高，恢复率涨几个点看不出名堂。

配对有效率不一样。它问的是：在天然结构里配对的那些位置上，
你设计出来的两个碱基还能不能配上？
一个"每个位置独立预测"的模型在这上面会很难看——它根本不知道
第 7 位选了 G 之后，第 40 位就必须是 C。
自回归解码器一上，这个数字会跳。这是自回归到底买到了什么的最直观证据，
在蛋白上反而看不到这么干净的现象。
"""
import numpy as np

CANONICAL = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}
WATSON_CRICK = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")}


def detect_pairs(wc, wc2, cutoff=3.4, min_sep=3):
    """用氢键原子间距判断配对。返回 partner[L]，没配对的是 -1。

    wc  [L,3]：主要 WC 边氢键原子。嘌呤(A,G) 用 N1，嘧啶(C,U) 用 N3。
    wc2 [L,3]：次要氢键原子。A 用 N6，G 用 O6，C 用 N4，U 用 O4。
    缺失的原子用 NaN 表示，会被自动跳过。

    判据：三组原子距离里任意一组小于 cutoff 就算配对。
      A-U / G-C  靠 wc-wc    （N1···N3，约 2.8-2.9 Å）
      G-U 摇摆    靠 wc2-wc   （O6···N3，约 2.8 Å）
    再加两条约束：序列上至少隔 min_sep 个位置（挨着的碱基不可能配对），
    一个碱基最多一个配偶（有多个候选就取最近的）。

    这是个粗判据，不是 DSSR。够不够用，看 pair_composition 的输出：
    检测出来的配对如果 95% 以上是 AU/GC/GU，就说明它在干正事。
    """
    L = wc.shape[0]
    big = 1e9

    def pdist(a, b):
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
        return np.where(np.isfinite(d), d, big)

    d = np.minimum(np.minimum(pdist(wc, wc), pdist(wc2, wc)), pdist(wc, wc2))
    sep = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])
    d = np.where(sep >= min_sep, d, big)

    partner = np.full(L, -1, dtype=np.int64)
    # 按距离从近到远贪心配对，保证一对一
    order = np.dstack(np.unravel_index(np.argsort(d, axis=None), d.shape))[0]
    for i, j in order:
        if d[i, j] >= cutoff:
            break
        if partner[i] == -1 and partner[j] == -1 and i != j:
            partner[i] = j
            partner[j] = i
    return partner


def pair_list(partner):
    """partner 数组 -> [(i,j)] 列表，i<j，每对只出现一次。"""
    return [(i, int(partner[i])) for i in range(len(partner))
            if partner[i] > i]


def pair_composition(seq, partner):
    """检测出来的配对都是些什么组合。用来验证检测器靠不靠谱。

    返回 dict，比如 {'GC': 12, 'AU': 8, 'GU': 2, '其他': 1}
    """
    out = {}
    for i, j in pair_list(partner):
        t = seq[i] + seq[j]
        key = t if (seq[i], seq[j]) in CANONICAL else "其他"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def pair_validity(seq, partner, wobble_ok=True):
    """配对有效率：天然配对的位置上，这条序列还能不能配上。

    seq 是字符串。返回 (有效比例, 配对总数)。没有配对就返回 (nan, 0)。
    """
    pairs = pair_list(partner)
    if not pairs:
        return float("nan"), 0
    allow = CANONICAL if wobble_ok else WATSON_CRICK
    good = sum(1 for i, j in pairs if (seq[i], seq[j]) in allow)
    return good / len(pairs), len(pairs)


def summarize(seq, partner, label=""):
    v, n = pair_validity(seq, partner)
    comp = pair_composition(seq, partner)
    s = " ".join("%s=%d" % (k, v2) for k, v2 in comp.items())
    print("%s配对 %d 对，有效 %.0f%%   组成：%s"
          % (label, n, 100 * v if n else float("nan"), s))
    return v
