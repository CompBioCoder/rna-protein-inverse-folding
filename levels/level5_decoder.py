# -*- coding: utf-8 -*-
"""第 5 关 · 自回归 —— 随机顺序解码器　★最难的一关

昨天的问题：每个位置独立预测，彼此不知道对方选了什么。
今天的解法：给核苷酸排一个顺序，一个一个定；
            定第 i 个的时候，允许看见"已经定好"的邻居选了什么碱基。

核心机关只有一行（minif/model.py 的 DecLayer）：

    sj = s_emb[idx] * mask_bw      # 还没轮到的邻居，碱基信息被乘成 0

mask_bw[i, j] = 1 当且仅当邻居 j 的解码次序排在 i 前面。整个自回归就这一个乘法。

为什么顺序是随机的，不是从 5' 到 3'？
  从左到右只能"前面决定后面"。随机顺序下模型见过各种
  "已知一部分、补剩下一部分"的局面——这正好是 RNA 设计的真实使用场景：
  固定几个关键位点，让模型补其余。第 7 关直接吃这个红利。

训练时不真的一步步跑（太慢），用教师强制：把真实序列整条喂进去，
靠 mask 保证第 i 位看不见自己、也看不见排在自己后面的位置。

★ 先跑 python tests/test_model.py，第 3、4 项必须过，再开始训练。
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import torch

from minif import data, engine, features as F, pairing, scoreboard
from minif.paths import run_file
from minif.model import MiniMPNN

MOL = "rna"
tr = data.load(MOL, "train"); va = data.load(MOL, "val")
data.describe(tr, "训练集："); data.describe(va, "验证集：")

prev = scoreboard.read()["levels"].get("4", {})
if prev.get("pair_validity"):
    print("第 4 关的配对有效率是 %.1f%%，今天要比它高至少 5 个百分点" % (100*prev["pair_validity"]))

model = MiniMPNN(d_node=F.node_dim(MOL), d_edge=F.edge_dim(MOL),
                 n_letters=F.n_letters(MOL), d_model=128, d_hidden=128, n_enc=3, n_dec=3)
print("参数量 %d\n" % sum(p.numel() for p in model.parameters()))

engine.fit(model, tr, va, MOL, epochs=60, lr=1e-3, noise=0.02,
           autoregressive=True, log_every=5, save=run_file("level5.pt"),
           level=5, eval_pairs=True)

# 生成一条看看，重点看茎区配没配上
rec = max(va, key=lambda r: (r["partner"] >= 0).sum())
designed = engine.design(model, rec, MOL, temperature=0.1)
native = rec["seq"]
same = sum(a == b for a, b in zip(native, designed)) / len(native)
mark = "".join("(" if 0 <= rec["partner"][i] and rec["partner"][i] > i else
               (")" if rec["partner"][i] >= 0 else ".") for i in range(len(native)))
print("\n%s" % rec["id"])
print("配对  %s" % mark)
print("天然  %s" % native)
print("设计  %s   逐位相同 %.0f%%" % (designed, 100*same))
pairing.summarize(native, rec["partner"], "天然：")
pairing.summarize(designed, rec["partner"], "设计：")

print("""
自测三问：
  1. gRNAde / ProteinMPNN 是怎么生成序列的？要能讲出随机顺序这一点。
  2. 教师强制是什么？它和真正采样生成的区别在哪？
  3. 如果解码器漏看了答案，你怎么发现？（tests/test_model.py 第 3、4 项在查什么）

卡住的降级方案：到今天结束还没过 tests_model.py，不要顺延。
用 python game.py hint 5 开参考实现，跑出结果继续往前，第 7 关收口时再回头。
保住主线比补齐每一关重要。
""")
