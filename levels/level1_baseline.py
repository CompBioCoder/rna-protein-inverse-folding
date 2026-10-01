# -*- coding: utf-8 -*-
"""第 1 关 · 第一滴血 —— 用几个最简单的结构特征预测碱基

这是整个项目最蠢的模型，但它已经是一个逆向设计模型了：
输入是结构信息（这个核苷酸周围挤不挤），输出是碱基。
后面六关做的全部事情，是把"输入"这一端换得越来越好。

RNA 有个尴尬：字母表只有 4 个，瞎猜就 25%，按最常见碱基猜接近 30%。
所以今天的第一件事是把基准线算出来——不知道底线在哪，涨多少都没意义。

跑：python levels/level1_baseline.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import numpy as np
import torch
import torch.nn as nn

from minif import data, features as F, scoreboard

MOL = "rna"
RADII = (10.0, 14.0, 18.0)

def flatten(recs, radii=RADII):
    """把所有链的所有核苷酸倒进一个大桶里。

    四个特征：三个半径的邻居数（埋藏程度）+ 这个位置配不配对。
    配对状态是单个特征里最强的——RNA 的碱基身份主要由它决定，
    而不是由埋藏程度决定。这一点和蛋白正好相反。
    """
    xs, ys = [], []
    for r in recs:
        nb = F.neighbor_counts(r["coords"][:, F.MOL[MOL]["center"]], radii)   # [L, 3]
        paired = (r["partner"] >= 0).astype("float32")[:, None]               # [L, 1]
        xs.append(np.concatenate([nb, paired], axis=1))
        ys.append(F.seq_to_idx(r["seq"], MOL))
    return (torch.tensor(np.concatenate(xs), dtype=torch.float32),
            torch.tensor(np.concatenate(ys), dtype=torch.long))


xtr, ytr = flatten(tr); xva, yva = flatten(va)
print("\nx 形状 %s（%d 个核苷酸，每个 %d 个特征）" % (tuple(xtr.shape), xtr.shape[0], xtr.shape[1]))
print("y 形状 %s（每个是 0-3 的整数）" % (tuple(ytr.shape),))

# ------------------------------------------------- 两条基准线，先立规矩
top, base = data.baseline(tr, MOL)
base_va = (yva == F.seq_to_idx(top, MOL)[0]).float().mean().item()
print("\n基准线：瞎猜 25.0%%；永远猜 %s（训练集最常见）在验证集上 %.1f%%" % (top, 100*base_va))

# ------------------------------------------------- 模型
# nn.Linear(a, b)：要求输入最后一维是 a，把它换成 b，前面的维度原样不动。
# 这里 [N, 1] -> [N, 4]。4 个输出就是 A C G U 各自的分数（logits）。
model = nn.Linear(xtr.shape[1], F.n_letters(MOL))
lossfn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.05)
print("参数量", sum(p.numel() for p in model.parameters()))

hist = []
for ep_i in range(1, 301):
    model.train()
    loss = lossfn(model(xtr), ytr)
    opt.zero_grad(); loss.backward(); opt.step()
    if ep_i == 1 or ep_i % 5 == 0:
        model.eval()
        with torch.no_grad():
            hist.append((ep_i,
                         lossfn(model(xtr), ytr).item(),
                         lossfn(model(xva), yva).item(),
                         (model(xtr).argmax(-1) == ytr).float().mean().item(),
                         (model(xva).argmax(-1) == yva).float().mean().item()))

ep, ltr, lva, atr, ava = (np.array(c) for c in zip(*hist))
best_i = int(ava.argmax())
acc = float(ava[best_i])          # 记最好的一轮，和第 3 关之后 engine.fit 的口径一致
print("验证恢复率：最好 %.1f%%（第 %d 轮），最后一轮 %.1f%%"
      % (100*acc, ep[best_i], 100*ava[-1]))

# ------------------------------------------------- 它学到了什么
with torch.no_grad():
    stem = model(torch.tensor([[2.0, 2.5, 3.0, 1.0]]))[0]    # 挤 + 配对 = 茎区
    loop = model(torch.tensor([[1.0, 1.3, 1.6, 0.0]]))[0]    # 松 + 不配对 = 环区
rank = lambda v: " > ".join(F.alphabet(MOL)[i] for i in v.argsort(descending=True).tolist())
print("茎区偏好：", rank(stem))
print("环区偏好：", rank(loop))
print("（茎区是配对且堆叠紧密的，G/C 应该排在前面）")

scoreboard.record(1, recovery=acc, baseline=base_va)
print("""
练习：把 flatten 里的 paired 那一列去掉（只留邻居数），重跑看掉多少。
       这能让你直观看到「配对状态」一个二值特征值多少个百分点。
自测三问（对着空气说一遍，说不顺就是没懂）：
  1. 逆向设计和正向预测的区别是什么？
  2. nn.Linear(768, 2) 吃 [3, 768] 出什么形状？吃 [2, 50, 768] 呢？
  3. opt.zero_grad() 不写会怎样？
""")
