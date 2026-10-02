# -*- coding: utf-8 -*-
"""第 3 关 · 结图者 —— kNN 图 + 边特征

今天不加任何新的网络结构，只把"输入"换成图。
每个核苷酸现在能看到周围 16 个最近邻（按 C4' 距离），
以及和每个邻居之间 4×4=16 个原子间距离。

今天真正要搞懂的是 gRNAde 五问第 1 问：结构怎么变成模型输入，为什么只用距离？
答案在 tests_geometry.py 第 5 项：距离在旋转平移下不变。
RNA 在空间里转一下还是同一条 RNA，模型输出就必须一样。
直接喂坐标 xyz 的话，模型得自己从数据里学会这件事，纯属浪费容量。
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import numpy as np
import torch
import torch.nn as nn

from minif import data, engine, features as F, pairing
from minif.model import mlp

MOL = "rna"
tr = data.load(MOL, "train"); va = data.load(MOL, "val")
data.describe(tr, "训练集："); data.describe(va, "验证集：")

# ---------------------------------------------- 先看看数据长什么样
# A 型螺旋在 eta-theta 平面上应该聚成很紧的一团。这是 RNA 结构最基本的事实。
ang = np.concatenate([F.torsions_rna(r["coords"], r.get("chi")) for r in tr])
eta = np.degrees(np.arctan2(ang[:, 0], ang[:, 3])) % 360
theta = np.degrees(np.arctan2(ang[:, 1], ang[:, 4])) % 360
ok = (eta > 0) & (theta > 0)
helical = ((eta[ok] > 140) & (eta[ok] < 220) & (theta[ok] > 170) & (theta[ok] < 260)).mean()
print("\neta 中位数 %.0f°  theta 中位数 %.0f°" % (np.median(eta[ok]), np.median(theta[ok])))
print("落在 A 型螺旋区的核苷酸占 %.0f%%（结构化 RNA 里这一团应该很显眼）" % (100*helical))

# 配对检测靠不靠谱：检测出来的应该绝大多数是 AU / GC / GU
comp = {}
for r in tr:
    for k, v in pairing.pair_composition(r["seq"], r["partner"]).items():
        comp[k] = comp.get(k, 0) + v
tot = sum(comp.values())
canon = sum(v for k, v in comp.items() if k != "其他")
print("配对检测自检：%d 对，规范配对 %.1f%%   %s"
      % (tot, 100*canon/max(tot, 1), "  ".join("%s=%d" % kv for kv in
                                               sorted(comp.items(), key=lambda kv: -kv[1]))))
print("（低于 90%% 就说明配对检测有问题，第 5 关的指标会不可信，回来说一声）")


class MeanPoolGraph(nn.Module):
    """点特征 + 邻居边特征的平均。没有可学习的消息，纯粹看"周围长什么样"。"""

    def __init__(self, d_node, d_edge, n_out, d=128):
        super().__init__()
        self.embed_E = nn.Linear(d_edge, d)
        self.head = mlp(d_node + d, d, n_out)

    def forward(self, V, E, idx):
        return self.head(torch.cat([V, self.embed_E(E).mean(dim=1)], dim=-1))


model = MeanPoolGraph(F.node_dim(MOL), F.edge_dim(MOL), F.n_letters(MOL))
print("\n参数量 %d" % sum(p.numel() for p in model.parameters()))
print("边特征 %d 维 = 4×4 个原子对距离 × 16 个 RBF 基 + 33 维相对序列位置\n" % F.edge_dim(MOL))

engine.fit(model, tr, va, MOL, epochs=40, lr=1e-3,
           autoregressive=False, log_every=5, level=3, eval_pairs=False)

print("""
今天必须跑：python tests/test_geometry.py   （8/8 才能过 BOSS）
练习：
  1. k 从 16 改成 4，再改成 32，各跑 10 轮。k 大一定更好吗？
  2. 把 features.py 里的相对位置 one-hot 关掉，掉多少？
     想一想：空间上挨着的两个核苷酸，可能是序列邻居，也可能是茎区里配对的对家。
自测三问：
  1. 结构输入为什么只用距离，不用坐标？
  2. 相对序列位置这个特征在 RNA 上为什么格外重要？
  3. A 型螺旋在 eta-theta 图上聚成一团，这说明什么？
""")
