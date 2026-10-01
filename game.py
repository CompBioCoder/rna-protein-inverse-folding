#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MiniMPNN 通关系统。

    python game.py              看地图和进度
    python game.py check 3      挑战第 3 关 BOSS
    python game.py hint 5       用一张提示卡（会记次数，不影响通关）
    python game.py achieve 捉虫人
    python game.py reset        清空进度

分数不用手动填：各关脚本跑完会自己写进 runs/scores.json。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minif import paths, scoreboard

LEVELS = [
    dict(n=1, name="第一滴血", sub="埋藏度 + 配对状态",
         file="levels/level1_baseline.py",
         goal="写出人生第一个训练循环，让模型比『永远猜最常见碱基』强",
         boss="恢复率 > 基准线 + 1.5 个百分点"),
    dict(n=2, name="几何入门", sub="赝扭转角 + 多层感知机",
         file="levels/level2_mlp.py",
         goal="把 eta / theta / 赝chi 喂进去，加上非线性",
         boss="恢复率 ≥ 第 1 关 + 1.5 个百分点"),
    dict(n=3, name="结图者", sub="kNN 图 + 边特征",
         file="levels/level3_graph.py",
         goal="建图，搞懂为什么只用距离不用坐标",
         boss="几何自测 8/8，且恢复率 ≥ 第 2 关 + 1.5 个百分点"),
    dict(n=4, name="传令兵", sub="消息传递编码器",
         file="levels/level4_encoder.py",
         goal="邻居之间开始互相传消息。今天模型第一次真的好用",
         boss="恢复率 ≥ 第 3 关 + 3 个百分点"),
    dict(n=5, name="自回归", sub="随机顺序解码器　★最难的一关",
         file="levels/level5_decoder.py",
         goal="让碱基一个一个定，后定的看得见先定的。配对有效率会跳",
         boss="模型自测 8/8（含因果性），且配对有效率 ≥ 第 4 关 + 5 个百分点"),
    dict(n=6, name="验尸官", sub="长训练 + 把结果拆开看",
         file="levels/level6_eval.py",
         goal="恢复率是一个数，它掩盖了很多东西。拆三刀",
         boss="恢复率 ≥ 第 5 关，且完成点特征消融实验"),
    dict(n=7, name="控制者", sub="固定位点 / 温度 / 跨分子",
         file="levels/level7_control.py",
         goal="逆向设计真正值钱的地方。顺手把同一套代码跑到蛋白上",
         boss="固定位点 100% 保住，且蛋白对照跑通"),
]

def _w(t):
    """显示宽度：中日韩字符占两列。"""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in t)


def pad(t, n):
    return t + " " * max(0, n - _w(t))


GREEN, YELLOW, GREY, RED, BOLD, OFF = "\033[32m", "\033[33m", "\033[90m", "\033[31m", "\033[1m", "\033[0m"
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    GREEN = YELLOW = GREY = RED = BOLD = OFF = ""


def get(st, n, key):
    v = st["levels"].get(str(n), {}).get(key)
    return v if isinstance(v, (int, float)) else None


def judge(st, n):
    """返回 (是否通过, [说明行])。"""
    L = lambda k: get(st, n, k)
    P = lambda m, k: get(st, m, k)
    out, ok = [], True

    def need(cond, text):
        nonlocal ok
        ok = ok and bool(cond)
        out.append(("  %s %s" % (GREEN + "✔" + OFF if cond else RED + "✘" + OFF, text)))

    if n == 1:
        r, b = L("recovery"), L("baseline")
        if r is None or b is None:
            return False, ["  还没有成绩。先跑 python levels/level1_baseline.py"]
        need(r > b + 0.015, "恢复率 %.1f%% > 基准线 %.1f%% + 1.5（现在超出 %+.1f 个百分点）"
             % (100*r, 100*b, 100*(r-b)))
    elif n in (2, 3, 4):
        gap = {2: 0.015, 3: 0.015, 4: 0.03}[n]
        r, p = L("recovery"), P(n-1, "recovery")
        if r is None:
            return False, ["  还没有成绩。先跑 python %s" % LEVELS[n-1]["file"]]
        if p is None:
            return False, ["  第 %d 关还没通，先把上一关打了" % (n-1)]
        need(r >= p + gap, "恢复率 %.1f%% ≥ 上一关 %.1f%% + %.1f（现在超出 %+.1f 个百分点）"
             % (100*r, 100*p, 100*gap, 100*(r-p)))
        if n == 3:
            g = L("geometry_tests")
            need(g == 8, "几何自测 %s/8" % (int(g) if g is not None else "未跑"))
    elif n == 5:
        m, v, pv = L("model_tests"), L("pair_validity"), P(4, "pair_validity")
        need(m == 8, "模型自测 %s/8（第 3、4 项查自回归有没有作弊）"
             % (int(m) if m is not None else "未跑"))
        if v is None:
            need(False, "配对有效率：还没跑出来")
        elif pv is None:
            need(False, "第 4 关的配对有效率没记上，回去重跑一次 level4")
        else:
            need(v >= pv + 0.05, "配对有效率 %.1f%% ≥ 第 4 关 %.1f%% + 5（现在 %+.1f 个百分点）"
                 % (100*v, 100*pv, 100*(v-pv)))
    elif n == 6:
        r, p, ab = L("recovery"), P(5, "recovery"), L("ablation_recovery")
        if r is None:
            return False, ["  还没有成绩。先跑 python levels/level6_eval.py"]
        need(p is not None and r >= p, "恢复率 %.1f%% ≥ 第 5 关 %s"
             % (100*r, "%.1f%%" % (100*p) if p else "（未记录）"))
        need(ab is not None, "点特征消融实验%s" % ("已完成，掉到 %.1f%%" % (100*ab) if ab is not None else "还没做"))
    elif n == 7:
        fh, pr = L("fixed_held"), L("protein_recovery")
        need(fh == 1.0, "固定位点保住率 %s" % ("100%" if fh == 1.0 else ("%.0f%%" % (100*fh) if fh is not None else "未跑")))
        need(pr is not None, "蛋白对照%s" % ("跑通，恢复率 %.1f%%" % (100*pr) if pr is not None else "还没跑"))
    return ok, out


def cleared(st, n):
    return judge(st, n)[0]


def bar(frac, width=28, lo=0.0, hi=1.0):
    f = max(0.0, min(1.0, (frac - lo) / (hi - lo))) if hi > lo else 0.0
    k = int(round(f * width))
    return "[" + "█" * k + "·" * (width - k) + "]"


def board():
    st = scoreboard.read()
    done = [n for n in range(1, 8) if cleared(st, n)]
    print()
    title = "RNA / 蛋白 逆向折叠  ·  七关"
    inner = 58
    print(BOLD + "  ╔" + "═" * inner + "╗" + OFF)
    print(BOLD + "  ║" + pad("  " + title, inner) + "║" + OFF)
    print(BOLD + "  ╚" + "═" * inner + "╝" + OFF)
    print("   进度 %s %d/7 关\n" % (bar(len(done)/7.0, 24), len(done)))

    unlocked = 1
    for n in range(1, 8):
        if cleared(st, n):
            unlocked = n + 1
        else:
            break

    for lv in LEVELS:
        n = lv["n"]
        got = cleared(st, n)
        if got:
            tag, col = "已通关", GREEN
        elif n <= unlocked:
            tag, col = "进行中", YELLOW
        else:
            tag, col = "未解锁", GREY
        rec = get(st, n, "recovery")
        pv = get(st, n, "pair_validity")
        hints = st["hints"].get(str(n), 0)
        extra = []
        if rec is not None:
            extra.append("恢复 %.1f%%" % (100*rec))
        if pv is not None:
            extra.append("配对 %.1f%%" % (100*pv))
        if hints:
            extra.append("提示卡 ×%d" % hints)
        print("%s  第%d关  %s%s %s%s" % (col, n, pad(lv["name"], 10), pad(lv["sub"], 26), tag, OFF))
        print("%s         %s%s" % (col, lv["goal"], OFF))
        print("%s         BOSS：%s%s" % (col, lv["boss"], OFF))
        if extra:
            print("%s         成绩：%s%s" % (col, "  ".join(extra), OFF))
        if n <= unlocked and not got:
            print("           跑 python %s" % lv["file"])
            print("           判定 python game.py check %d" % n)
        print()

    # 恢复率进度条
    b = get(st, 1, "baseline")
    best = max([get(st, n, "recovery") or 0 for n in range(1, 8)] + [0])
    if best:
        print("  恢复率 %s %.1f%%" % (bar(best, 28, 0.20, 0.60), 100*best))
        if b:
            print("         基准线 %.1f%%（永远猜最常见碱基）" % (100*b))
        print("         参考：gRNAde 论文在大得多的数据集上到 50%+。")
        print("         你用几百条链和一个小模型，别去追那个数。")
    pvs = [get(st, n, "pair_validity") for n in (4, 5, 6, 7)]
    pvs = [v for v in pvs if v is not None]
    if pvs:
        print("\n  配对有效率 %s %.1f%%" % (bar(max(pvs), 28, 0.25, 1.0), 100*max(pvs)))
        print("         这个指标比恢复率诚实。RNA 瞎猜就有 25% 恢复率，")
        print("         但瞎猜的序列配对几乎全错。")

    print("\n  " + paths.describe().replace("\n", "\n  "))
    if st["achievements"]:
        print("\n  成就：" + "  ".join("★" + a for a in st["achievements"]))
    print()


def check(n):
    st = scoreboard.read()
    if n > 1 and not cleared(st, n-1):
        print("\n  第 %d 关还没通，先打上一关。\n" % (n-1))
        return 1
    ok, lines = judge(st, n)
    lv = LEVELS[n-1]
    print("\n  ── 第 %d 关 BOSS：%s ──" % (n, lv["name"]))
    print("  要求：%s\n" % lv["boss"])
    for l in lines:
        print(l)
    if ok:
        print("\n  " + GREEN + BOLD + "通过。第 %d 关已通关。" % n + OFF)
        if n == 1:
            scoreboard.unlock("首杀")
        if st["hints"].get(str(n), 0) == 0:
            if scoreboard.unlock("不看攻略·第%d关" % n):
                print("  " + YELLOW + "★ 解锁成就：不看攻略·第%d关" % n + OFF)
        if n == 7:
            scoreboard.unlock("通关")
            print("\n  " + BOLD + "七关全通。把 notes/level7.md 写完，五问就齐了。" + OFF)
        print()
    else:
        print("\n  " + RED + "还没过。" + OFF + "卡住超过 25 分钟就用提示卡：python game.py hint %d\n" % n)
    return 0 if ok else 1


def hint(n):
    cnt = scoreboard.use_hint(n)
    f = LEVELS[n-1]["file"]
    print("\n  第 %d 关提示卡（第 %d 次）" % (n, cnt))
    print("  参考实现在 %s" % f)
    print("  只看卡住的那一段，看完关掉自己重写一遍。")
    print("  用提示卡不影响通关，只是拿不到'不看攻略'成就。\n")


def main():
    a = sys.argv[1:]
    if not a:
        board(); return 0
    cmd = a[0]
    if cmd == "check" and len(a) > 1:
        return check(int(a[1]))
    if cmd == "hint" and len(a) > 1:
        hint(int(a[1])); return 0
    if cmd == "achieve" and len(a) > 1:
        print("★ 解锁成就：%s" % a[1] if scoreboard.unlock(a[1]) else "已经有了")
        return 0
    if cmd == "reset":
        if input("清空全部进度？输 yes 确认：").strip() == "yes":
            scoreboard.write({"levels": {}, "hints": {}, "achievements": []})
            print("已清空")
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
