# -*- coding: utf-8 -*-
"""第 2 关 · 几何入门 —— 三个扭转角 + 多层感知机

和 notebooks/level2_mlp.ipynb 内容一致，这里是无图的命令行版。

两个变化：
  1. 点特征从 4 维扩到 10 维：
       eta / theta / chi 各拆成 sin+cos        6 维（本关新增）
       三个半径（10/14/18 A）的邻居数          3 维（第 1 关就有）
       配不配对                                1 维（第 1 关就有）
     列顺序是 sin sin sin / cos cos cos，不是交替 —— 取列的时候最容易错。
  2. 模型从单个 Linear 变成 Linear -> GELU -> Linear -> GELU -> Linear。
     这就是"深度学习"里"深"的全部含义。中间那层叫隐藏层，
     它让模型能表达"配对【并且】chi 在某区间"这类组合条件，线性层做不到。

三个角的定义：
    eta   = C4'(i-1) - P(i)   - C4'(i)  - P(i+1)     赝扭转角 (Duarte & Pyle 1998)
    theta = P(i)     - C4'(i) - P(i+1)  - C4'(i+1)   赝扭转角
    chi   = O4' - C1' - N9 - C4   (嘌呤)             真扭转角，IUPAC 标准
            O4' - C1' - N1 - C2   (嘧啶)
注意 eta/theta 是【赝】角、chi 是【真】角，不能统称"三个赝扭转角"。

本关跑完得到的结论（详见 notes/level2.md）：
  * BOSS 50.2%，第 1 关 40.9%，门槛 42.4% —— 过关
  * 三组特征都有增量，没有一组是白给的
  * chi 的全部增量都要靠碱基环上的原子，gRNAde 的三珠输入拿不到 —— 这是泄漏
  * oracle 只对离散特征成立，连续特征要用"训练建表 -> 验证评分"的查表基线

环境变量：
    SWEEP=1   额外跑隐藏层宽度扫描（12 次训练，几分钟）
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import numpy as np
import torch
import torch.nn as nn

from minif import data, features as F, scoreboard

MOL = "rna"
H = 128          # 隐藏层宽度，由下面的宽度扫描选出：两列成绩都不输给 32/64/256
SEED = 0         # 固定种子保证可复现。要比较两个方案谁好，再跑多个种子看抖动

tr = data.load(MOL, "train"); va = data.load(MOL, "val")
data.describe(tr, "训练集："); data.describe(va, "验证集：")
prev = scoreboard.read()["levels"].get("1", {}).get("recovery")


# ---------------------------------------------------------------- 特征

def flatten(recs):
    """把所有链摊平成两张大表。摊平之后链的边界就没了，每个残基是独立样本 ——
    残基之间传消息是第 3 关的事。"""
    xs = [F.node_features(r["coords"], MOL, r["partner"], r.get("chi")) for r in recs]
    ys = [F.seq_to_idx(r["seq"], MOL) for r in recs]
    return (torch.tensor(np.concatenate(xs), dtype=torch.float32),
            torch.tensor(np.concatenate(ys), dtype=torch.long))


xtr, ytr = flatten(tr); xva, yva = flatten(va)
BASELINE = torch.bincount(yva).max().item() / len(yva)
print("\n点特征 %d 维   训练 %d 个残基   验证 %d 个残基" % (xtr.shape[1], len(ytr), len(yva)))
print("基准线（全猜最常见的碱基）%.1f%%" % (100*BASELINE))


# ---------------------------------------------------------------- 小工具

def pad(s, w):
    """按显示宽度补空格：中文字符占两格，不然表格会歪。"""
    s = str(s)
    return s + " " * max(0, w - sum(2 if ord(c) > 0x2E80 else 1 for c in s))


def train(a, b, seed=SEED, epochs=80, h=H, track_train=False):
    """用特征矩阵 a(训练) / b(验证) 训一个 MLP。

    返回 (最好验证恢复率, 最后一轮验证, 最好那轮预测用到几类, 最后一轮训练)。

    训练循环五步：前向 -> 算损失 -> zero_grad -> backward -> step。
    漏掉 zero_grad 梯度会累加，是最常见的 bug。

    注意"最好一轮"是用验证集挑出来的，系统性偏乐观 1.5~2 个百分点。
    报数的时候两个都报。
    """
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(a.shape[1], h), nn.GELU(),
                      nn.Linear(h, h), nn.GELU(),
                      nn.Linear(h, F.n_letters(MOL)))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    lf = nn.CrossEntropyLoss()
    n = a.shape[0]
    best, best_ep, last, ncls, last_tr = 0.0, 0, 0.0, 0, 0.0
    for ep in range(1, epochs + 1):
        m.train()
        perm = torch.randperm(n)
        for i in range(0, n, 512):
            s = perm[i:i+512]
            loss = lf(m(a[s]), ytr[s])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            p = m(b).argmax(-1)
            last = (p == yva).float().mean().item()
            if track_train:
                last_tr = (m(a).argmax(-1) == ytr).float().mean().item()
        if last > best:
            best, best_ep, ncls = last, ep, int(p.unique().numel())
    return best, best_ep, last, ncls, last_tr


# ---------------------------------------------------------------- 主模型

print("\n" + "=" * 62)
print("主模型：隐藏层 %d，80 轮，Adam lr=1e-3，batch 512，seed=%d" % (H, SEED))
print("=" * 62)
best, best_ep, last_va, _, last_tr = train(xtr, xva, track_train=True)
nparam = sum(p.numel() for p in nn.Sequential(
    nn.Linear(xtr.shape[1], H), nn.GELU(), nn.Linear(H, H), nn.GELU(),
    nn.Linear(H, F.n_letters(MOL))).parameters())
print("参数量 %d" % nparam)
print("最好 %.1f%%（第 %d 轮）   最后一轮 %.1f%%   训练集最后一轮 %.1f%%   口子 %+.1f"
      % (100*best, best_ep, 100*last_va, 100*last_tr, 100*(last_tr - last_va)))


# ---------------------------------------------------------------- 宽度扫描（可选）

if os.environ.get("SWEEP") == "1":
    print("\n" + "=" * 62)
    print("隐藏层宽度扫描（每个宽度 3 个种子）")
    print("=" * 62)
    print("单跑一次的 1 个百分点差异很可能只是初始权重的运气，所以要跑多个种子。")
    print("%5s %9s %13s %9s %10s" % ("宽度", "参数量", "最好(均±标)", "见顶轮", "最后一轮"))
    for h in (32, 64, 128, 256):
        rs = [train(xtr, xva, seed=s, h=h) for s in (0, 1, 2)]
        bb = [r[0] for r in rs]
        np_ = sum(p.numel() for p in nn.Sequential(
            nn.Linear(xtr.shape[1], h), nn.GELU(), nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, F.n_letters(MOL))).parameters())
        print("%5d %9d  %5.1f±%.1f%% %10.0f %9.1f%%"
              % (h, np_, 100*np.mean(bb), 100*np.std(bb),
                 np.mean([r[1] for r in rs]), 100*np.mean([r[2] for r in rs])))
    print("选在两列上都不输给别人的、最小的那个宽度。实测是 128。")


# ---------------------------------------------------------------- 表 1 消融

print("\n" + "=" * 62)
print("表 1 · 消融：一次只留一组特征，看谁在挣分")
print("=" * 62)
print("删列而不是置零：置零会让 cos=0 变成'90 度'这个真实取值，")
print("模型会当成一个真实的角去学，而不是'这里没有信息'。")
print()

GROUPS = {"eta/theta": [0, 1, 3, 4], "chi": [2, 5], "邻居数": [6, 7, 8], "配对": [9]}
COMBOS = [
    ("仅配对（第 1 关的核心）", ["配对"]),
    ("仅 chi",                  ["chi"]),
    ("仅 eta/theta",            ["eta/theta"]),
    ("配对 + chi",              ["配对", "chi"]),
    ("配对 + eta/theta",        ["配对", "eta/theta"]),
    ("全部 10 维",              ["配对", "chi", "eta/theta", "邻居数"]),
]

print(pad("特征组合", 26) + "%5s %9s %10s %9s" % ("维度", "最好", "最后一轮", "用到几类"))
print("-" * 62)
for name, keep in COMBOS:
    cols = sorted(sum((GROUPS[g] for g in keep), []))
    b_, _, l_, nc, _ = train(xtr[:, cols], xva[:, cols])
    print(pad(name, 26) + "%5d %8.1f%% %9.1f%% %9d" % (len(cols), 100*b_, 100*l_, nc))
print()
print("'用到几类'是退化检查：如果只输出 1 类，恢复率再高也只是在猜最常见的字母。")


# ---------------------------------------------------------------- 表 2 查表基线

print("\n" + "=" * 62)
print("表 2 · 查表基线（不是天花板）")
print("=" * 62)
print("oracle（同一份数据既建表又评分）只对【离散】特征成立。")
print("chi 是连续量，分桶越细 oracle 越高，极限是每个残基一个桶 = 100%。")
print("所以这里主看'查表基线'：训练集建表 -> 验证集评分，是下限参照不是上限。")
print("旁边保留 oracle 做对照，看它怎么随分桶被吹起来。")
print()


def chi_deg(x):
    """从 sin/cos 两列还原 chi 角度，映射到 [0,360)。"""
    return torch.rad2deg(torch.atan2(x[:, 2], x[:, 5])) % 360


KEYS = {
    "仅配对":                      lambda x: x[:, 9].long(),
    "配对 + chi（anti/syn 二分）": lambda x: x[:, 9].long() * 2 + (x[:, 5] >= 0).long(),
    "配对 + chi（30 度分桶）":     lambda x: x[:, 9].long() * 12 + (chi_deg(x) // 30).long().clamp(0, 11),
    "配对 + chi（10 度分桶）":     lambda x: x[:, 9].long() * 36 + (chi_deg(x) // 10).long().clamp(0, 35),
}


def lookup(key, xfit, yfit, xev, yev):
    """在 xfit/yfit 上建查找表（每组取最常见的碱基），到 xev/yev 上评分。
    xfit 和 xev 是同一份数据时，结果等于 oracle —— 这是个免费的自检。"""
    kf, ke = key(xfit), key(xev)
    tb = {int(g): int(torch.bincount(yfit[kf == g], minlength=F.n_letters(MOL)).argmax())
          for g in kf.unique()}
    fb = int(torch.bincount(yfit).argmax())
    pred = torch.tensor([tb.get(int(g), fb) for g in ke])
    unseen = sum(1 for g in ke.unique() if int(g) not in tb)
    return (pred == yev).float().mean().item(), int(ke.unique().numel()), unseen


def oracle(key, x, y):
    k = key(x)
    return sum(torch.bincount(y[k == g], minlength=F.n_letters(MOL)).max().item()
               for g in k.unique()) / len(y)


print(pad("特征", 30) + "%12s %12s %9s %8s" % ("查表基线", "验证oracle", "分组数", "没见过"))
print("-" * 74)
for name, key in KEYS.items():
    lb, ng, un = lookup(key, xtr, ytr, xva, yva)
    print(pad(name, 30) + "%11.1f%% %11.1f%% %9d %8d" % (100*lb, 100*oracle(key, xva, yva), ng, un))
print()
print("模型实际（全部 10 维）：最好 %.1f%%   最后一轮 %.1f%%" % (100*best, 100*last_va))
print("两列差距随分组数单调拉大，就是 oracle 在用验证集自己的标签给自己打分的证据。")


# ---------------------------------------------------------------- 表 2b 诊断

print("\n" + "=" * 62)
print("表 2b · 诊断：查表和 MLP 对 chi 的估值差 3~5 个百分点，差在哪")
print("=" * 62)
print("判据（跑之前就写死，不许事后改）：")
print("  MLP 训练集 >> 查表训练集       -> MLP 在用桶内细结构，或者在背数据")
print("  两者训练集接近、只在验证集分叉 -> 训练/验证分布迁移，不是 chi 的信息量问题")
print("  MLP 训练集 ≈ 同数据 oracle     -> MLP 榨干了这组特征，是查表那边低估")
print()

KEY10 = KEYS["配对 + chi（10 度分桶）"]
COLS = [2, 5, 9]                                  # sin(chi), cos(chi), 配对
_, _, mva, _, mtr = train(xtr[:, COLS], xva[:, COLS], track_train=True)

print(pad("方法", 30) + "%10s %10s" % ("训练集", "验证集"))
print("-" * 52)
print(pad("查表：训练建表 -> 评分", 30) + "%9.1f%% %9.1f%%"
      % (100*lookup(KEY10, xtr, ytr, xtr, ytr)[0], 100*lookup(KEY10, xtr, ytr, xva, yva)[0]))
print(pad("MLP：最后一轮", 30) + "%9.1f%% %9.1f%%" % (100*mtr, 100*mva))
print(pad("同数据 oracle（会虚高）", 30) + "%9.1f%% %9.1f%%"
      % (100*oracle(KEY10, xtr, ytr), 100*oracle(KEY10, xva, yva)))
print(pad("对照：仅配对", 30) + "%9.1f%% %9.1f%%"
      % (100*oracle(KEYS["仅配对"], xtr, ytr), 100*oracle(KEYS["仅配对"], xva, yva)))
print()
print("实测落在第二条：训练集只差 0.9，验证集差 3.0 -> 分布迁移。")
print("顺带一个免费自检：训练集那一列，查表和 oracle 必须【完全相等】。")
print("两个独立写的函数算出同一个数，说明实现都没错。")


# ---------------------------------------------------------------- 表 3 泄漏对照

print("\n" + "=" * 62)
print("表 3 · 泄漏对照：chi 到底在不在偷看答案")
print("=" * 62)
print("gRNAde 的输入只有 P / C4' / N1-N9 三个珠子，没有碱基环上的原子。")
print("而标准 chi 的第四个原子（嘌呤 C4 / 嘧啶 C2）就在碱基环上 ——")
print("要先知道碱基字母（= 标准答案）才知道取哪个。")
print("对照组用无标签赝扭转角 P-C4'-C1'-N：四个原子两种碱基都有，不需要知道身份。")
print("三行共用同一个底座（eta/theta 的 sin/cos + 配对），只换第三个角。")
print()


def pseudo_chi(coords):
    """P - C4' - C1' - 糖苷N 的二面角，拆成 sin/cos。coords 是 [L,4,3]。"""
    d = np.nan_to_num(F._dihedral(coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]))
    return np.stack([np.sin(d), np.cos(d)], axis=-1).astype(np.float32)


BASE = [0, 1, 3, 4, 9]


def build(recs, third):
    out = []
    for r in recs:
        v = F.node_features(r["coords"], MOL, r["partner"], r.get("chi"))
        base = v[:, BASE]
        if third == "none":
            out.append(base)
        elif third == "pseudo":
            out.append(np.concatenate([base, pseudo_chi(r["coords"])], axis=-1))
        else:
            out.append(np.concatenate([base, v[:, [2, 5]]], axis=-1))
    return torch.tensor(np.concatenate(out), dtype=torch.float32)


ROWS = [("none",   "不要第三个角（配对 + eta/theta）", "否"),
        ("pseudo", "无标签赝扭转角 P-C4'-C1'-N",       "否"),
        ("chi",    "标准 chi",                         "是")]

print(pad("第三个角", 34) + "%5s %9s %11s %9s" % ("维度", "最好", "最后一轮", "需碱基原子"))
print("-" * 72)
res = {}
for tag, name, need in ROWS:
    a, b = build(tr, tag), build(va, tag)
    b_, _, l_, _, _ = train(a, b)
    res[tag] = b_
    print(pad(name, 34) + "%5d %8.1f%% %10.1f%% %9s" % (a.shape[1], 100*b_, 100*l_, need))
print()
print("标准 chi 比无标签赝扭转角高 %+.1f 个百分点。" % (100*(res["chi"] - res["pseudo"])))
print("这部分在 gRNAde 的三珠输入下拿不到 —— 是泄漏，不是几何。")


# ---------------------------------------------------------------- 验收

print("\n" + "=" * 62)
print("验收结果（照抄进 notes/level2.md）")
print("=" * 62)
print("- 最终配置：隐藏层宽度 %d，80 轮，Adam lr=1e-3，batch 512，seed=%d" % (H, SEED))
print("- 恢复率（验证集最好一轮）：%.1f%%（第 %d 轮）" % (100*best, best_ep))
print("- 同一次运行的最后一轮：%.1f%%" % (100*last_va))
print("- 训练集最后一轮：%.1f%%（口子 %+.1f 个百分点）" % (100*last_tr, 100*(last_tr - last_va)))
if prev:
    thr = prev + 0.015
    print("- 第 1 关：%.1f%%（门槛 %.1f%%）" % (100*prev, 100*thr))
    print("- BOSS：%s，%+.1f 个百分点" % ("过了" if best >= thr else "没过", 100*(best - thr)))
print()
print("口径说明：'最好一轮'是用验证集挑的训练轮数，系统性偏乐观 1.5~2 个百分点。")
print("第 1、2 关都用这个口径所以关卡之间可比，但绝对值不可对外引用。第 3 关起加 test 划分。")

scoreboard.record(2, recovery=best)

print("""
自测六问（答案都在 notes/level2.md 的正文里）：
  1. 训练和验证差出来的几个百分点是什么？数据只有几百条链时怎么压住它？
  2. 为什么训练/验证是按序列簇划分的，不是随机划分？
     （看 prepare_data.py 里那段 cluster。这是你的主场，答案要比我写的详细）
  3. 二面角为什么要拆成 sin/cos？当前的 10 个维度分别是什么，顺序是怎样的？
  4. 本脚本的三个扭转角是怎么定义的？有没有文献支持？文献里还有哪些别的定义方式，
     分别有哪些算法在用？（能不能自己把那张对比表默出来）
  5. 欠拟合和过拟合的差别是什么？判据是"口子大"还是别的？
     有哪两个现象会被误判成过拟合？
  6. 第 2 关和第 1 关的差别是什么？两关共同的局限是什么，哪一关才解决？
""")
