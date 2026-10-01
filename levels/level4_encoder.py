# -*- coding: utf-8 -*-
"""第 4 关 · 传令兵 —— 消息传递编码器

和昨天唯一的区别：邻居信息不再是简单平均，而是
    每个邻居 j 根据（我自己、邻居 j、我俩之间的几何）算出一条消息，
    我把收到的所有消息平均起来，加到自己身上。
叠 3 层，信息就能传 3 跳远。

这一层代码只有十几行（minif/model.py 的 EncLayer），但它是整个图神经网络的全部。

今天还没有自回归——每个位置独立预测，互相不知道对方选了什么。
这个洞在 RNA 上特别致命：茎区里第 7 位和第 40 位必须配对，
但这个模型算第 7 位的时候根本不知道第 40 位会选什么。
所以今天除了恢复率，还要量一个配对有效率——明天那个数字会跳。
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import torch
import torch.nn as nn

from minif import data, engine, features as F
from minif.paths import run_file
from minif.model import EncLayer

MOL = "rna"
tr = data.load(MOL, "train"); va = data.load(MOL, "val")
data.describe(tr, "训练集："); data.describe(va, "验证集：")


class EncoderOnly(nn.Module):
    def __init__(self, d_node, d_edge, n_out, d=128, n_layers=3):
        super().__init__()
        self.embed_V = nn.Linear(d_node, d)
        self.embed_E = nn.Linear(d_edge, d)
        self.layers = nn.ModuleList([EncLayer(d, d, d) for _ in range(n_layers)])
        self.out = nn.Linear(d, n_out)

    def forward(self, V, E, idx):
        h, e = self.embed_V(V), self.embed_E(E)
        for layer in self.layers:
            h = layer(h, e, idx)
        return self.out(h)


model = EncoderOnly(F.node_dim(MOL), F.edge_dim(MOL), F.n_letters(MOL))
print("\n参数量 %d\n" % sum(p.numel() for p in model.parameters()))

engine.fit(model, tr, va, MOL, epochs=60, lr=1e-3, noise=0.02,
           autoregressive=False, log_every=5, save=run_file("level4.pt"),
           level=4, eval_pairs=True)

print("""
把配对有效率这个数记下来，明天要和它比。

练习：
  1. n_layers 改成 1 和 6，各跑 20 轮。层数和恢复率是什么关系？为什么不是越多越好？
  2. 拿张纸画：3 层消息传递后，一个核苷酸的信息能影响到多远？
  3. 训练时加的 0.02Å 坐标噪声是干什么的？设 noise=0 看验证恢复率怎么变。
自测：
  这个模型设计出来的序列，在茎区里为什么会配不上？用一句话说清机制。
""")
