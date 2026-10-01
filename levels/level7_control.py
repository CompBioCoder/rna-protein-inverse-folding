# -*- coding: utf-8 -*-
"""第 7 关 · 控制者 —— 固定位点 / 温度 / 跨分子

这是 gRNAde 五问第 3 问，也是逆向设计真正值钱的地方。

三个旋钮：
  固定位点    锁住若干位置不动，让模型重新设计其余
  温度        调多样性
  logits 偏置  全局禁用或偏好某些碱基

固定位点为什么能直接做？因为训练时用的是随机解码顺序。
模型见过各种"已知一部分补其余"的局面，固定位点只是把已知那部分指定成你要的。
注意固定位点要排在解码顺序最前面，否则它们的信息传不给后面的位置。

这一条正好对上你的核糖开关问题：保住配体结合口袋的关键碱基，重设计其余。

最后半小时把同一套代码跑到蛋白上——这就是 gRNAde 和 ProteinMPNN
"同一范式换分子"的具体含义。
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import numpy as np
import torch

from minif import data, engine, features as F, pairing, scoreboard
from minif.paths import run_file
from minif.model import MiniMPNN

MOL = "rna"
if not os.path.exists(run_file("level6.pt")):
    raise SystemExit("找不到 %s —— 先跑 python levels/level6_eval.py" % run_file("level6.pt"))

va = data.load(MOL, "val")
model = MiniMPNN(d_node=F.node_dim(MOL), d_edge=F.edge_dim(MOL),
                 n_letters=F.n_letters(MOL), d_model=128, d_hidden=128, n_enc=3, n_dec=3)
model.load_state_dict(torch.load(run_file("level6.pt"))); model.eval()

rec = max(va, key=lambda r: (r["partner"] >= 0).sum())
seq, partner, L = rec["seq"], rec["partner"], len(rec["seq"])
dots = "".join("(" if partner[i] > i else (")" if partner[i] >= 0 else ".") for i in range(L))
print("%s（%d nt，%d 对配对）" % (rec["id"], L, int((partner >= 0).sum()) // 2))
print("配对  %s" % dots)
print("天然  %s" % seq)

# ---- 1. 自由设计
free = engine.design(model, rec, MOL, temperature=0.3)
print("自由  %s   与天然 %.0f%%" % (free, 100*sum(a == b for a, b in zip(seq, free))/L))
pairing.summarize(free, partner, "      ")

# ---- 2. 固定位点：锁住环区（功能位点通常在环上），重设计茎区
loop = [i for i in range(L) if partner[i] < 0]
fixed = {int(p): int(F.seq_to_idx(seq, MOL)[p]) for p in loop}
out = engine.design(model, rec, MOL, temperature=0.3, fixed=fixed)
held = sum(1 for p, a in fixed.items() if F.seq_to_idx(out, MOL)[p] == a) / max(len(fixed), 1)
print("\n锁住 %d 个环区位置（功能位点一般在这里），重设计茎区：" % len(fixed))
print("锁定  %s" % "".join("^" if i in fixed else " " for i in range(L)))
print("结果  %s   固定位点保住 %.0f%%" % (out, 100*held))
pairing.summarize(out, partner, "      ")
stem = [i for i in range(L) if i not in fixed]
print("非固定位置与天然相同 %.0f%%"
      % (100*np.mean([out[i] == seq[i] for i in stem]) if stem else float("nan")))

# ---- 3. 反过来：锁住茎区，重设计环区
fixed2 = {i: int(F.seq_to_idx(seq, MOL)[i]) for i in range(L) if partner[i] >= 0}
out2 = engine.design(model, rec, MOL, temperature=0.3, fixed=fixed2)
print("\n反过来锁住 %d 个茎区位置，重设计环区：" % len(fixed2))
print("结果  %s" % out2)

# ---- 4. logits 偏置：禁用 G（看模型怎么重排配对）
bias = torch.zeros(F.n_letters(MOL)); bias[F.alphabet(MOL).index("G")] = -1e4
nog = engine.design(model, rec, MOL, temperature=0.3, bias=bias)
print("\n禁用 G 之后：%s" % nog)
print("序列里还有 G 吗：%s" % ("G" in nog))
pairing.summarize(nog, partner, "      ")
print("（G 没了，GC 对只能变成 AU/UA。看配对有效率掉多少——这是约束的代价）")

# ---- 5. 同一骨架多条设计
print("\n同一骨架 T=0.5 采 5 条：")
outs = [engine.design(model, rec, MOL, temperature=0.5) for _ in range(5)]
for o in outs:
    v, _ = pairing.pair_validity(o, partner)
    print("  %s  与天然 %.0f%%  配对有效 %.0f%%"
          % (o, 100*sum(a == b for a, b in zip(seq, o))/L, 100*v))
pw = [1 - sum(a == b for a, b in zip(outs[x], outs[y]))/L
      for x in range(5) for y in range(x+1, 5)]
print("5 条之间平均差异 %.0f%%" % (100*np.mean(pw)))

scoreboard.record(7, fixed_held=held)

# ---- 6. 跨分子：同一套代码跑蛋白
print("\n" + "="*60)
print("跨分子对照：同一套代码，mol 换成 protein")
print("="*60)
if not os.path.exists(data.path_for("protein")):
    print("先跑：python prepare_data.py --mol protein")
    print("（下完再跑一次本脚本，就能拿到第 7 关 BOSS 的第二个条件）")
else:
    ptr = data.load("protein", "train"); pva = data.load("protein", "val")
    data.describe(ptr, "蛋白训练集："); data.describe(pva, "蛋白验证集：")
    pm = MiniMPNN(d_node=F.node_dim("protein"), d_edge=F.edge_dim("protein"),
                  n_letters=F.n_letters("protein"), d_model=128, d_hidden=128,
                  n_enc=3, n_dec=3)
    pr = engine.fit(pm, ptr, pva, "protein", epochs=40, lr=1e-3, noise=0.02,
                    autoregressive=True, log_every=10, save=run_file("level7_protein.pt"),
                    eval_pairs=False)
    scoreboard.record(7, protein_recovery=pr)
    print("\n改了什么？一个参数：mol='rna' -> 'protein'。")
    print("背后自动跟着变的只有三样：字母表 4->20，骨架原子，赝扭转角。")
    print("kNN、RBF、相对位置、消息传递、随机顺序自回归、固定位点——一行没动。")

print("""
=========== 收口 ===========

技术部分写进 notes/level7.md（这个仓库是公开的，只写技术）：

  五问的答案
  「我这个版本和官方差在哪」——README 里列了五条，用自己跑出来的数字补充
  「换一种分子要改哪几行」：
      字母表 4 -> 20
      骨架原子 P/C4'/C1'/N糖苷 -> N/CA/C/O(+虚拟Cβ)
      赝扭转角 eta/theta/赝chi -> phi/psi/omega
      边特征的原子对数跟着变（16 -> 25）
      其余一行没动

研究方向相关的东西——核糖开关双构象切换那个空白、你打算怎么做——
写进你自己的私有 vault，不要放进这个公开仓库。那是还没发表的想法。
""")
