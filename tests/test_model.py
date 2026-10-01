# -*- coding: utf-8 -*-
"""模型自测。第 4、5 关写完模型就跑：python tests_model.py

8/8 是第 5 关 BOSS 的条件之一。

第 3、4 项最关键，它们检查自回归有没有作弊。
解码器如果偷看到了自己要预测的那个碱基，训练 loss 会好看、恢复率会虚高，
但采样生成时全盘崩掉——而且从训练曲线上完全看不出来。
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from minif import features as F, scoreboard
from minif.model import MiniMPNN

torch.manual_seed(0)
np.random.seed(0)

MOL = "rna"
L, K = 30, 8
NL = F.n_letters(MOL)


def make_inputs(rot=False):
    rng = np.random.RandomState(7)
    c4 = np.cumsum(rng.randn(L, 3).astype(np.float32) * 2.5, axis=0)
    coords = np.stack([c4 + rng.randn(L, 3).astype(np.float32) * 0.8 for _ in range(4)], axis=1)
    if rot:
        Q, _ = np.linalg.qr(np.random.RandomState(3).randn(3, 3))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        coords = ((coords.reshape(-1, 3) @ Q.T) + np.array([5., -2., 9.])).reshape(coords.shape)
    coords = coords.astype(np.float32)
    f = F.featurize(coords, np.arange(L, dtype=np.int64), MOL, k=K)
    return (torch.tensor(f["V"]), torch.tensor(f["E"]), torch.tensor(f["idx"]),
            torch.randint(0, NL, (L,)))


def stepwise_logits(model, V, E, idx, true_seq, order):
    """逐个解码，每一步只把已经走过的位置的真实碱基填进去。"""
    h_enc, e = model.encode(V, E, idx)
    mask_bw = model.backward_mask(order, idx)
    out = torch.zeros(L, NL)
    seq = torch.zeros(L, dtype=torch.long)
    decided = torch.zeros(L, dtype=torch.bool)
    for t in range(L):
        i = int(order[t])
        m = mask_bw * decided[idx].unsqueeze(-1).to(torch.float32)
        s = model.embed_S(seq)
        h = h_enc
        for layer in model.dec:
            h = layer(h, h_enc, s, e, idx, m)
        out[i] = model.out(h[i])
        seq[i] = true_seq[i]
        decided[i] = True
    return out


def main():
    ok = []
    V, E, idx, seq = make_inputs()
    model = MiniMPNN(d_node=V.shape[-1], d_edge=E.shape[-1], n_letters=NL,
                     d_model=32, d_hidden=32, n_enc=2, n_dec=2, d_seq=8).eval()
    order = torch.randperm(L)
    rank = torch.empty(L, dtype=torch.long); rank[order] = torch.arange(L)

    with torch.no_grad():
        # 1. 形状
        h_enc, e = model.encode(V, E, idx)
        logits = model.decode(h_enc, e, idx, seq, order)
        print("1. 形状 V%s E%s idx%s -> h_enc%s logits%s"
              % (tuple(V.shape), tuple(E.shape), tuple(idx.shape),
                 tuple(h_enc.shape), tuple(logits.shape)))
        ok.append(h_enc.shape == (L, 32) and logits.shape == (L, NL))

        # 2. 编码器完全不碰序列
        h2, _ = model.encode(V, E, idx)
        l2 = model.decode(h_enc, e, idx, torch.randint(0, NL, (L,)), order)
        print("2. 编码器与序列无关：h_enc 差 %.1e；换序列后 logits 有变化 %s"
              % ((h_enc-h2).abs().max().item(), bool((logits-l2).abs().max() > 1e-5)))
        ok.append((h_enc-h2).abs().max().item() < 1e-7 and (logits-l2).abs().max() > 1e-5)

        # 3. 因果性
        i = int(order[L//2])
        later = [j for j in range(L) if rank[j] > rank[i]]
        nb_earlier = [int(j) for j in idx[i].tolist() if rank[j] < rank[i]]
        s1 = seq.clone()
        for j in later:
            s1[j] = (s1[j]+1) % NL
        d_later = (model.decode(h_enc, e, idx, s1, order)[i] - logits[i]).abs().max().item()
        d_earlier = 0.0
        if nb_earlier:
            s2 = seq.clone()
            for j in nb_earlier:
                s2[j] = (s2[j]+1) % NL
            d_earlier = (model.decode(h_enc, e, idx, s2, order)[i] - logits[i]).abs().max().item()
        print("3. 因果性：改排在后面的位置 logits[i] 变 %.1e（必须≈0）；改前面的邻居变 %.1e（必须>0）"
              % (d_later, d_earlier))
        ok.append(d_later < 1e-6 and d_earlier > 1e-5)

        # 4. 教师强制 == 逐步解码
        d = (stepwise_logits(model, V, E, idx, seq, order) - logits).abs().max().item()
        print("4. 教师强制 vs 逐步解码，最大差 %.1e（必须≈0）" % d)
        ok.append(d < 1e-5)

        # 5. 邻居顺序无关（mean 聚合）
        perm = torch.randperm(K)
        hp, _ = model.encode(V, E[:, perm], idx[:, perm])
        print("5. 打乱邻居顺序：h_enc 差 %.1e（应≈0）" % (hp-h_enc).abs().max().item())
        ok.append((hp-h_enc).abs().max().item() < 1e-5)

        # 6. 旋转平移不变
        Vr, Er, idxr, _ = make_inputs(rot=True)
        hr, er = model.encode(Vr, Er, idxr)
        lr = model.decode(hr, er, idxr, seq, order)
        print("6. 旋转平移后 logits 差 %.1e（应很小）" % (lr-logits).abs().max().item())
        ok.append((lr-logits).abs().max().item() < 1e-3)

        # 7. 采样 + 固定位点
        fixed = {3: 1, 10: 2, 20: 0}
        od = torch.tensor(sorted(fixed) + [i for i in range(L) if i not in fixed])
        s = model.sample(V, E, idx, od, temperature=0.5, fixed=fixed)
        held = all(int(s[p]) == a for p, a in fixed.items())
        print("7. 采样长度 %d，固定位点保住 %s，用到 %d 种碱基" % (len(s), held, len(set(s.tolist()))))
        ok.append(len(s) == L and held)

        # 8. logits 偏置能真的禁用一个碱基
        bias = torch.zeros(NL); bias[2] = -1e4          # 禁用 G
        sb = model.sample(V, E, idx, torch.randperm(L), temperature=1.0, bias=bias)
        print("8. 禁用 G 后序列里还有 G 吗：%s（应为 False）" % (2 in sb.tolist()))
        ok.append(2 not in sb.tolist())

    n = sum(ok)
    print("\n%s  %d/8" % ("全部通过" if n == 8 else "有不通过的项——先别往下走", n))
    scoreboard.record(5, model_tests=n)
    return 0 if n == 8 else 1


if __name__ == "__main__":
    sys.exit(main())
