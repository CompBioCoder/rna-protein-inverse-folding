# -*- coding: utf-8 -*-
"""第 3 关以后共用的训练 / 评估循环。

第 1、2 关的训练循环是手写在关卡脚本里的——那是要学的内容，不能藏在这里。
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn

from . import features as F
from . import pairing, scoreboard

CE = nn.CrossEntropyLoss()


def prep(rec, mol="rna", k=16, noise=0.0, rng=None, device="cpu"):
    """一条链 -> 可以直接喂模型的张量。"""
    f = F.featurize(rec["coords"], rec["resnum"], mol=mol, k=k, noise=noise, rng=rng,
                    partner=rec.get("partner"), chi_atoms=rec.get("chi"))
    t = lambda x, d: torch.as_tensor(x, dtype=d, device=device)
    return (t(f["V"], torch.float32), t(f["E"], torch.float32), t(f["idx"], torch.long),
            t(F.seq_to_idx(rec["seq"], mol), torch.long))


def random_order(L, device="cpu", first=None):
    """随机解码顺序。first 里的位置排到最前面（第 7 关固定位点要用）。"""
    first = list(first or [])
    rest = [i for i in range(L) if i not in set(first)]
    np.random.shuffle(rest)
    return torch.as_tensor(first + rest, dtype=torch.long, device=device)


def run_epoch(model, recs, opt=None, mol="rna", k=16, noise=0.0,
              device="cpu", autoregressive=True):
    """跑一遍数据。opt=None 就是只评估不更新。返回 (平均损失, 恢复率)。"""
    train = opt is not None
    model.train(train)
    seq_order = np.random.permutation(len(recs)) if train else np.arange(len(recs))
    tot_loss = tot_hit = tot_n = 0
    rng = np.random.RandomState()
    for ri in seq_order:
        V, E, idx, seq = prep(recs[ri], mol, k, noise if train else 0.0, rng, device)
        L = seq.shape[0]
        with torch.set_grad_enabled(train):
            logits = (model(V, E, idx, seq, random_order(L, device))
                      if autoregressive else model(V, E, idx))
            loss = CE(logits, seq)
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        tot_loss += loss.item() * L
        tot_hit += (logits.argmax(-1) == seq).sum().item()
        tot_n += L
    return tot_loss / tot_n, tot_hit / tot_n


@torch.no_grad()
def design(model, rec, mol="rna", temperature=0.3, autoregressive=True,
           k=16, device="cpu", fixed=None, bias=None):
    """生成一条序列，返回字符串。

    自回归模型逐位生成，每一步看得到已经定下来的邻居。
    非自回归模型（第 4 关）只能每个位置各自独立抽样——
    这正是它在配对上吃亏的原因。
    """
    model.eval()
    V, E, idx, seq = prep(rec, mol, k, device=device)
    L = seq.shape[0]
    if autoregressive:
        order = random_order(L, device, first=sorted(fixed) if fixed else None)
        out = model.sample(V, E, idx, order, temperature=temperature, fixed=fixed, bias=bias)
    else:
        logits = model(V, E, idx)
        if bias is not None:
            logits = logits + bias
        out = torch.multinomial(torch.softmax(logits / max(temperature, 1e-6), dim=-1), 1).squeeze(-1)
        for p, a in (fixed or {}).items():
            out[p] = a
    return F.idx_to_seq(out.tolist(), mol)


def eval_pair_validity(model, recs, mol="rna", temperature=0.3,
                       autoregressive=True, n_rep=2, k=16, device="cpu"):
    """配对有效率：天然配对的位置上，设计出来的序列还配不配得上。

    只对有配对的链算。返回 (平均有效率, 用到的链数)。
    """
    if mol != "rna":
        return float("nan"), 0
    vals = []
    for rec in recs:
        if (rec["partner"] >= 0).sum() == 0:
            continue
        for _ in range(n_rep):
            s = design(model, rec, mol, temperature, autoregressive, k, device)
            v, n = pairing.pair_validity(s, rec["partner"])
            if n:
                vals.append(v)
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


def fit(model, tr, va, mol="rna", epochs=60, lr=1e-3, k=16, noise=0.02, device="cpu",
        autoregressive=True, log_every=5, save=None, level=None, eval_pairs=True):
    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = 0.0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        trl, tra = run_epoch(model, tr, opt, mol, k, noise, device, autoregressive)
        vll, vaa = run_epoch(model, va, None, mol, k, 0.0, device, autoregressive)
        if vaa > best:
            best = vaa
            if save:
                torch.save(model.state_dict(), save)
        if ep % log_every == 0 or ep == 1 or ep == epochs:
            print("  ep%3d  训练 loss %.3f 恢复 %.1f%%   验证 loss %.3f 恢复 %.1f%%   最好 %.1f%%  [%.0fs]"
                  % (ep, trl, 100*tra, vll, 100*vaa, 100*best, time.time()-t0))

    if save and os.path.exists(save):
        model.load_state_dict(torch.load(save))
    print("\n验证集最好恢复率：%.1f%%" % (100*best))

    pv = None
    if eval_pairs and mol == "rna":
        pv, n = eval_pair_validity(model, va, mol, autoregressive=autoregressive, k=k, device=device)
        if n:
            print("配对有效率：%.1f%%（%d 次设计）" % (100*pv, n))

    if level is not None:
        scoreboard.record(level, recovery=best, pair_validity=pv)
    return best
