# -*- coding: utf-8 -*-
"""读 prepare_data.py 产出的 npz。"""
import os
import numpy as np

from .paths import data_file


def path_for(mol="rna"):
    return data_file(mol)


def load(mol="rna", split=None, path=None):
    """返回 list，每项 dict(id, seq, coords[L,4,3], resnum[L], partner[L])。"""
    path = path or path_for(mol)
    if not os.path.exists(path):
        raise SystemExit("找不到 %s\n先跑：python prepare_data.py --mol %s"
                         % (os.path.normpath(path), mol))
    z = np.load(path, allow_pickle=True)
    has_chi = "chi" in z.files
    if not has_chi and mol == "rna":
        print("[data] 这个数据集是旧版的，没有 chi 原子，糖苷扭转角会被填 0。\n"
              "       重跑一次 python prepare_data.py 即可。")
    out = []
    for i in range(len(z["ids"])):
        if split is not None and z["split"][i] != split:
            continue
        rec = {"id": str(z["ids"][i]), "seq": str(z["seqs"][i]),
               "coords": z["coords"][i].astype(np.float32),
               "resnum": z["resnum"][i].astype(np.int64),
               "partner": z["partner"][i].astype(np.int64)}
        if has_chi:
            rec["chi"] = z["chi"][i].astype(np.float32)
        out.append(rec)
    if not out:
        raise SystemExit("split=%r 一条都没有" % split)
    return out


def describe(recs, name=""):
    ls = [len(r["seq"]) for r in recs]
    npair = sum(int((r["partner"] >= 0).sum()) // 2 for r in recs)
    extra = "，配对 %d 对" % npair if npair else ""
    print("%s%d 条，长度 %d-%d（平均 %.0f），总残基 %d%s"
          % (name, len(recs), min(ls), max(ls), np.mean(ls), sum(ls), extra))


def baseline(recs, mol="rna"):
    """最常见字母基准线：永远猜出现最多的那个碱基/氨基酸能对多少。

    这是判断模型有没有学到东西的那条底线。RNA 上这条线很高（约 27-30%），
    所以别被 30% 的恢复率唬住。
    """
    from . import features as F
    from collections import Counter
    c = Counter("".join(r["seq"] for r in recs))
    top, n = c.most_common(1)[0]
    return top, n / sum(c.values())
