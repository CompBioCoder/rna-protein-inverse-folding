#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备数据：下载结构 -> 解析骨架 -> 检测碱基配对 -> 去冗余划分 -> 存成一个 npz。

只用标准库 + numpy。
用法：
    python prepare_data.py                      # RNA（默认）
    python prepare_data.py --mol protein        # 蛋白，第 7 关对照用
    python prepare_data.py --ids ids.txt        # 自备 PDB ID 列表

产出：data/rna.npz（或 protein.npz）
    ids     (M,)  "1EHZ_A"
    seqs    (M,)  序列字符串
    coords  (M,)  float32 [L, 4, 3]
              RNA  四个原子依次 P, C4', C1', N糖苷(嘌呤N9/嘧啶N1)
              蛋白 四个原子依次 N, CA, C, O
    resnum  (M,)  int32 [L]，PDB 残基编号，用来判断链断裂
    partner (M,)  int32 [L]，配对对象下标，-1 表示没配对（蛋白全是 -1）
    split   (M,)  "train" / "val"
"""
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minif import pairing
from minif import paths
from minif.paths import DATA_DIR as OUTDIR

VAL_FRAC = 0.15

# ---------------------------------------------------------------- RNA 定义
RNA_PARENT = {
    "A": "A", "C": "C", "G": "G", "U": "U",
    # 常见修饰碱基，映射回母核。tRNA 结构里很多，不处理会丢掉大半条链
    "PSU": "U", "5MU": "U", "H2U": "U", "4SU": "U", "UR3": "U", "5BU": "U",
    "1MA": "A", "2MA": "A", "MIA": "A", "T6A": "A", "I6A": "A", "6MZ": "A",
    "1MG": "G", "2MG": "G", "M2G": "G", "7MG": "G", "OMG": "G", "G7M": "G",
    "YG": "G", "QUO": "G", "GDP": "G",
    "5MC": "C", "OMC": "C", "4OC": "C", "CBR": "C",
}
PURINES = set("AG")
RNA_GLYC = {"A": "N9", "G": "N9", "C": "N1", "U": "N1"}   # 糖苷氮
RNA_WC = {"A": "N1", "G": "N1", "C": "N3", "U": "N3"}     # WC 边主氢键原子
RNA_WC2 = {"A": "N6", "G": "O6", "C": "N4", "U": "O4"}    # 次要氢键原子

# 这批 ID 我不保证每个都有效，脚本会自动跳过下不到或解析不出的。
# 只是搜索接口不通时的应急，数据会偏少。
RNA_FALLBACK = """
1EHZ 1EVV 1FIR 2TRA 1YFG 1ASY 1SER 4TNA 6TNA 1TRA
2GIS 1Y26 1U8D 3IRW 1Q9A 1MME 1NBS 1L2X 1ZIH 2KOC
1SCL 1XJR 1KXK 1GID 1DUQ 1F27 1MFQ 1E8O 1A9N 1QC0
1F7Y 1HMH 437D 1ZZN 2OIU 3D2G 1P5O 1KH6 2A43 1I9V
""".split()

PROTEIN_PARENT = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}
PROTEIN_FALLBACK = """
1UBQ 1CRN 2GB1 1PGB 1IGD 1ENH 1VII 2CI2 1SHG 1BDD 1L2Y 1MJC 1CSP 1C9O
1BPI 5PTI 1TEN 1TIT 1NYF 1SRL 1AEY 1BTA 2PTL 1HZ6 1POH 1HDN 1E0L 1PIN
1DIV 1RIS 1QYS 1FKB 1URN 3CHY 2ACY 1APS 1SSO 2ABD 1YCC 1HRC 1CYO 2F4K
""".split()

CFG = {
    "rna": dict(parent=RNA_PARENT, backbone=["P", "C4'", "C1'"],
                min_len=30, max_len=120, max_res=3.2, fallback=RNA_FALLBACK),
    "protein": dict(parent=PROTEIN_PARENT, backbone=["N", "CA", "C", "O"],
                    min_len=40, max_len=130, max_res=2.0, fallback=PROTEIN_FALLBACK),
}


def http_get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "rna-protein-inverse-folding/1.0 (learning project)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_pdb(pid):
    """取一个 PDB 文件。先看外置盘缓存，没有再下载并存下来。

    这样重跑 prepare_data 不用再下一遍，换参数重新筛选时特别省时间。
    外置盘没挂上就直接走网络，不缓存。
    """
    cache = None
    if paths.available(paths.PDB_DIR):
        cache = os.path.join(paths.PDB_DIR, "%s.pdb" % pid.upper())
        if os.path.exists(cache):
            try:
                with open(cache, encoding="utf-8", errors="replace") as f:
                    return f.read()
            except Exception:
                pass
    try:
        txt = http_get("https://files.rcsb.org/download/%s.pdb" % pid,
                       timeout=30).decode("utf-8", "replace")
    except Exception:
        return None
    if cache:
        try:
            with open(cache, "w", encoding="utf-8") as f:
                f.write(txt)
        except Exception:
            pass
    return txt


def search_ids(mol, n_want=600):
    c = CFG[mol]
    nodes = [
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.resolution_combined",
            "operator": "less_or_equal", "value": c["max_res"]}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "entity_poly.rcsb_sample_sequence_length",
            "operator": "range",
            "value": {"from": c["min_len"], "to": c["max_len"]}}},
    ]
    if mol == "rna":
        nodes += [
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "entity_poly.rcsb_entity_polymer_type",
                "operator": "exact_match", "value": "RNA"}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                "operator": "equals", "value": 0}},
        ]
    else:
        nodes += [
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                "operator": "equals", "value": 1}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_entry_info.selected_polymer_entity_types",
                "operator": "exact_match", "value": "Protein (only)"}},
        ]
    q = {"query": {"type": "group", "logical_operator": "and", "nodes": nodes},
         "return_type": "entry",
         "request_options": {"paginate": {"start": 0, "rows": n_want},
                             "results_content_type": ["experimental"]}}
    url = ("https://search.rcsb.org/rcsbsearch/v2/query?json="
           + urllib.parse.quote(json.dumps(q)))
    try:
        data = json.loads(http_get(url, timeout=60))
        ids = [x["identifier"] for x in data.get("result_set", [])]
        return ids or None
    except Exception as e:
        print("  搜索接口没通：%s" % e)
        return None


def collect_atoms(text, parent):
    """把 PDB 文本读成 {chain: [(key, resname1, {atom: xyz}), ...]}，保持出现顺序。"""
    res, names, order = {}, {}, []
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if line[:6] not in ("ATOM  ", "HETATM"):
            continue
        rn = line[17:20].strip()
        if rn not in parent:
            continue
        if line[16] not in (" ", "A"):          # altloc
            continue
        key = (line[21], line[22:27])
        if key not in res:
            res[key] = {}
            names[key] = parent[rn]
            order.append(key)
        try:
            res[key][line[12:16].strip()] = (
                float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            pass
    by_chain = {}
    for k in order:
        by_chain.setdefault(k[0], []).append((k, names[k], res[k]))
    return by_chain


def parse_entry(text, mol):
    """返回 (chain, seq, coords[L,4,3], resnum[L], partner[L]) 或 None。

    只取原子最全的那一条链。RNA 的发夹、假结这类链内配对都在同一条链上，
    跨链的双链体会被丢掉——对学习没影响。
    """
    c = CFG[mol]
    by_chain = collect_atoms(text, c["parent"])
    if not by_chain:
        return None
    chain = max(by_chain, key=lambda ch: len(by_chain[ch]))

    seq, coords, resnum, wc, wc2 = [], [], [], [], []
    nan3 = (float("nan"),) * 3
    for key, one, atoms in by_chain[chain]:
        if mol == "rna":
            need = c["backbone"] + [RNA_GLYC[one]]
        else:
            need = c["backbone"]
        if not all(a in atoms for a in need):
            continue
        seq.append(one)
        coords.append([atoms[a] for a in need])
        try:
            resnum.append(int(key[1][:4]))
        except ValueError:
            resnum.append(len(resnum))
        if mol == "rna":
            wc.append(atoms.get(RNA_WC[one], nan3))
            wc2.append(atoms.get(RNA_WC2[one], nan3))

    if len(seq) < c["min_len"]:
        return None
    L = len(seq)
    if mol == "rna":
        partner = pairing.detect_pairs(np.asarray(wc, dtype=np.float64),
                                       np.asarray(wc2, dtype=np.float64))
    else:
        partner = np.full(L, -1, dtype=np.int64)
    return (chain, "".join(seq),
            np.asarray(coords, dtype=np.float32),
            np.asarray(resnum, dtype=np.int32),
            partner.astype(np.int32))


def kmers(s, k=3):
    return set(s[i:i + k] for i in range(len(s) - k + 1))


def cluster(seqs, thresh=0.5):
    """粗暴去冗余：3-mer Jaccard 超过阈值算同一簇（贪心）。

    这不是 mmseqs，只能挡掉明显同源。真要发文章用 mmseqs easy-cluster
    或者 RNA 侧的 cd-hit-est。RNA 字母表只有 4 个，3-mer 的区分度比蛋白低，
    所以阈值设得比蛋白高。
    """
    sets = [kmers(s) for s in seqs]
    labels = [-1] * len(seqs)
    reps = []
    for i, si in enumerate(sets):
        hit = -1
        for cid, sr in reps:
            inter = len(si & sr)
            if inter and inter / len(si | sr) >= thresh:
                hit = cid
                break
        if hit < 0:
            hit = len(reps)
            reps.append((hit, si))
        labels[i] = hit
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mol", default="rna", choices=["rna", "protein"])
    ap.add_argument("--ids", help="自备 ID 列表文件，每行一个 PDB ID")
    args = ap.parse_args()
    mol, c = args.mol, CFG[args.mol]
    os.makedirs(OUTDIR, exist_ok=True)

    if args.ids:
        ids = [l.strip().upper()[:4] for l in open(args.ids) if l.strip()]
        print("用你给的 ID 列表：%d 个" % len(ids))
    else:
        print("向 RCSB 要 %s 的 ID 列表 ..." % mol.upper())
        ids = search_ids(mol)
        if ids is None:
            ids = c["fallback"]
            print("  退到兜底列表（%d 个，其中一些可能无效，会自动跳过）" % len(ids))
            print("  要全量：去 rcsb.org 网页搜索导出 ID 存成 ids.txt，")
            print("  再跑 python prepare_data.py --mol %s --ids ids.txt" % mol)
        else:
            print("  拿到 %d 个" % len(ids))

    recs = []
    for n, pid in enumerate(ids, 1):
        if n % 25 == 0 or n == len(ids):
            print("  下载解析 %d/%d，已收 %d 条" % (n, len(ids), len(recs)))
        txt = fetch_pdb(pid)
        if txt is None:
            continue
        got = parse_entry(txt, mol)
        if got is None:
            continue
        ch, seq, xyz, rnum, partner = got
        if not (c["min_len"] <= len(seq) <= c["max_len"]):
            continue
        recs.append(("%s_%s" % (pid, ch), seq, xyz, rnum, partner))

    if len(recs) < 15:
        sys.exit("只拿到 %d 条，太少。检查网络，或手动准备 ids.txt。" % len(recs))

    seen, uniq = set(), []
    for r in recs:
        if r[1] in seen:
            continue
        seen.add(r[1]); uniq.append(r)
    recs = uniq

    labels = cluster([r[1] for r in recs])
    ncl = max(labels) + 1
    rng = np.random.RandomState(0)
    val_cl = set(rng.permutation(ncl)[:max(1, int(round(ncl * VAL_FRAC)))].tolist())
    split = ["val" if labels[i] in val_cl else "train" for i in range(len(recs))]

    def obj(xs):
        a = np.empty(len(xs), dtype=object)
        for i, x in enumerate(xs):
            a[i] = x
        return a

    out = os.path.join(OUTDIR, "%s.npz" % mol)
    np.savez_compressed(out,
                        ids=obj([r[0] for r in recs]), seqs=obj([r[1] for r in recs]),
                        coords=obj([r[2] for r in recs]), resnum=obj([r[3] for r in recs]),
                        partner=obj([r[4] for r in recs]), split=obj(split))
    nt = sum(1 for s in split if s == "train")
    print("\n存好了：%s" % out)
    print("  %d 条，%d 个簇，训练 %d / 验证 %d（按簇划分，不是随机划分）"
          % (len(recs), ncl, nt, len(recs) - nt))
    print("  平均长度 %.0f" % np.mean([len(r[1]) for r in recs]))

    if mol == "rna":
        # 自检：检测出来的配对应该绝大多数是 AU/GC/GU。
        # 不是的话说明配对检测器有问题，后面第 5 关的指标就不可信。
        comp = {}
        npair = 0
        for r in recs:
            for k, v in pairing.pair_composition(r[1], r[4]).items():
                comp[k] = comp.get(k, 0) + v
                npair += v
        canon = sum(v for k, v in comp.items() if k != "其他")
        print("\n配对检测自检：共 %d 对，规范配对占 %.1f%%" % (npair, 100 * canon / max(npair, 1)))
        print("  组成：" + "  ".join("%s=%d" % kv for kv in
                                  sorted(comp.items(), key=lambda kv: -kv[1])))
        print("  这个比例应该在 90%% 以上。明显偏低就说明配对检测不对，回来说一声。")


if __name__ == "__main__":
    main()
