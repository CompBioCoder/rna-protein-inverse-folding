# -*- coding: utf-8 -*-
"""第 2 关 · 几何入门 —— 赝扭转角 + 多层感知机

两个变化：
  1. 特征从 1 个变成 9 个：eta / theta / 赝chi 各拆成 sin+cos 共 6 个，
     加上三个半径的邻居数 3 个。
  2. 模型从 Linear 变成 Linear -> GELU -> Linear -> GELU -> Linear。
     这就是"深度学习"里"深"的全部含义。

RNA 主链有 6 个真扭转角（alpha 到 zeta），维度高又彼此相关，不好用。
Duarte 和 Pyle 提出用两个赝扭转角概括，只需要 P 和 C4'：
    eta   = C4'(i-1) - P(i)   - C4'(i)  - P(i+1)
    theta = P(i)     - C4'(i) - P(i+1)  - C4'(i+1)
第 3 关你会亲眼看到 A 型螺旋在 eta-theta 平面上聚成很紧的一团。

今天还要学会看一件事：训练恢复率和验证恢复率拉开的那个口子，就是过拟合。
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import numpy as np
import torch
import torch.nn as nn

from minif import data, features as F, scoreboard

MOL = "rna"
tr = data.load(MOL, "train"); va = data.load(MOL, "val")
data.describe(tr, "训练集："); data.describe(va, "验证集：")


def flatten(recs):
    xs = [F.node_features(r["coords"], MOL, r["partner"]) for r in recs]
    ys = [F.seq_to_idx(r["seq"], MOL) for r in recs]
    return (torch.tensor(np.concatenate(xs), dtype=torch.float32),
            torch.tensor(np.concatenate(ys), dtype=torch.long))


xtr, ytr = flatten(tr); xva, yva = flatten(va)
print("\n特征 %d 维：三个赝扭转角的 sin+cos（6）+ 三个半径的邻居数（3）" % xtr.shape[1])

# 形状链：[N,9] -> [N,64] -> [N,64] -> [N,4]
model = nn.Sequential(
    nn.Linear(xtr.shape[1], 64), nn.GELU(),
    nn.Linear(64, 64), nn.GELU(),
    nn.Linear(64, F.n_letters(MOL)),
)
print("参数量 %d" % sum(p.numel() for p in model.parameters()))

lossfn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# 小批量：一次只拿一部分数据算梯度。比全量更新更快也更稳。
N = xtr.shape[0]
best = 0.0
for ep in range(1, 81):
    model.train()
    perm = torch.randperm(N)
    for b in range(0, N, 512):
        sel = perm[b:b+512]
        loss = lossfn(model(xtr[sel]), ytr[sel])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        atr = (model(xtr).argmax(-1) == ytr).float().mean().item()
        ava = (model(xva).argmax(-1) == yva).float().mean().item()
    best = max(best, ava)
    if ep % 10 == 0 or ep == 1:
        print("  ep%2d  训练 %.1f%%  验证 %.1f%%  口子 %.1f 个百分点"
              % (ep, 100*atr, 100*ava, 100*(atr-ava)))

print("\n验证集最好 %.1f%%" % (100*best))
scoreboard.record(2, recovery=best)
print("""
自测三问：
  1. 训练和验证差出来的那几个百分点是什么？数据只有几百条链时怎么压住它？
  2. 为什么训练集和验证集是按序列簇划分的，不是随机划分？
     （prepare_data.py 里那段 cluster。这是你的主场，答案要比我写的详细）
  3. 二面角为什么要拆成 sin/cos？
""")
