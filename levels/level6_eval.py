# -*- coding: utf-8 -*-
"""第 6 关 · 验尸官 —— 长训练 + 把结果拆开看

恢复率是一个数，它掩盖了很多东西。今天拆三刀：
  按配对状态拆 —— 茎区和环区，哪一边好做？
  按碱基种类拆 —— 哪几种它学不会？最常被错认成什么？
  按温度拆     —— 采样温度怎么在"像天然"和"有多样性"之间换

外加一个消融：把点特征全清零（ProteinMPNN 原文就是这么干的，信息全在边上），
看掉几个点。
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import numpy as np
import torch

from minif import data, engine, features as F, scoreboard
from minif.paths import run_file
from minif.model import MiniMPNN

MOL = "rna"
CKPT = run_file("level6.pt")
tr = data.load(MOL, "train"); va = data.load(MOL, "val")
data.describe(tr, "训练集："); data.describe(va, "验证集：")

mk = lambda **kw: MiniMPNN(d_node=F.node_dim(MOL), d_edge=F.edge_dim(MOL),
                           n_letters=F.n_letters(MOL), d_model=128, d_hidden=128,
                           n_enc=3, n_dec=3, **kw)

if not os.path.exists(CKPT):
    print("\n开始长训练 150 轮（预计半小时上下，跑着的时候去读 level6 这个文件）\n")
    engine.fit(mk(), tr, va, MOL, epochs=150, lr=1e-3, noise=0.02,
               autoregressive=True, log_every=10, save=CKPT, level=6, eval_pairs=True)

model = mk(); model.load_state_dict(torch.load(CKPT)); model.eval()

# 已经有权重时也补记一次成绩，免得 scores.json 被清掉后 BOSS 判不了
if not scoreboard.read()["levels"].get("6", {}).get("recovery"):
    _, acc = engine.run_epoch(model, va, None, MOL, autoregressive=True)
    pv, _ = engine.eval_pair_validity(model, va, MOL)
    scoreboard.record(6, recovery=acc, pair_validity=pv)

# ---- 第一刀：茎区 vs 环区
hit_s = n_s = hit_l = n_l = 0
conf = np.zeros((4, 4), dtype=int)
with torch.no_grad():
    for rec in va:
        V, E, idx, seq = engine.prep(rec, MOL)
        pred = model(V, E, idx, seq, engine.random_order(len(seq))).argmax(-1)
        hit = (pred == seq).numpy()
        paired = rec["partner"] >= 0
        hit_s += hit[paired].sum(); n_s += paired.sum()
        hit_l += hit[~paired].sum(); n_l += (~paired).sum()
        for t, p in zip(seq.tolist(), pred.tolist()):
            conf[t, p] += 1
print("\n茎区（配对）恢复 %.1f%%（n=%d）" % (100*hit_s/max(n_s, 1), n_s))
print("环区（不配对）恢复 %.1f%%（n=%d）" % (100*hit_l/max(n_l, 1), n_l))
print("（茎区应该明显好做：配对把可选项压到了几种组合。环区几乎什么都能放）")

# ---- 第二刀：按碱基
print("\n每种碱基：")
A = F.alphabet(MOL)
for i in range(4):
    tot = conf[i].sum()
    if not tot:
        continue
    wrong = conf[i].copy(); wrong[i] = -1
    print("  %s  n=%5d  恢复 %4.1f%%   最常被错认成 %s" % (A[i], tot, 100*conf[i, i]/tot, A[int(wrong.argmax())]))

# ---- 第三刀：温度
print("\n温度扫描（验证集前 6 条，每条采 3 次）：")
from minif import pairing
for T in (0.1, 0.3, 0.5, 1.0):
    sims, divs, pvs = [], [], []
    for rec in va[:6]:
        outs = [engine.design(model, rec, MOL, temperature=T) for _ in range(3)]
        sims += [sum(a == b for a, b in zip(rec["seq"], o))/len(o) for o in outs]
        divs += [1 - sum(a == b for a, b in zip(outs[x], outs[y]))/len(outs[0])
                 for x in range(3) for y in range(x+1, 3)]
        if (rec["partner"] >= 0).any():
            pvs += [pairing.pair_validity(o, rec["partner"])[0] for o in outs]
    print("  T=%.1f  与天然相同 %.1f%%   互相不同 %.1f%%   配对有效 %.1f%%"
          % (T, 100*np.mean(sims), 100*np.mean(divs), 100*np.nanmean(pvs) if pvs else float("nan")))
print("（设计要的不是最像天然，而是在能折回目标骨架的前提下尽量多样）")

# ---- 消融：点特征清零
AB = run_file("level6_ablation.pt")
print("\n消融实验：点特征全部清零，重训 60 轮")
ab = engine.fit(mk(use_node_feats=False), tr, va, MOL, epochs=60, lr=1e-3, noise=0.02,
                autoregressive=True, log_every=20, save=AB, eval_pairs=False)
scoreboard.record(6, ablation_recovery=ab)
print("（掉得少说明几何信息本来就几乎全在边上——ProteinMPNN 原文正是这么设计的）")

print("""
自测三问：
  1. 恢复率高是不是就说明设计成功了？用今天的温度表给答案补一句实测数据。
  2. 自洽性验证是什么？为什么它才是真正的判据？
  3. 点特征清零掉几个点？这说明几何信息主要在哪？
""")
