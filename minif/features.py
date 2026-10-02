# -*- coding: utf-8 -*-
"""几何特征：把骨架坐标变成模型能吃的数组。RNA 和蛋白共用一套。

全部用 numpy 写，和 PyTorch 无关。这样几何部分可以单独测，坏了一眼看得出来。

两种分子的区别只有三处（这就是 gRNAde 和 ProteinMPNN 的全部差别）：
    字母表      RNA 4 个碱基      蛋白 20 个氨基酸
    骨架原子    P, C4', C1', N    N, CA, C, O (+ 虚拟 Cβ)
    赝扭转角    eta, theta, chi   phi, psi, omega
kNN 建图、RBF 距离展开、相对序列位置、消息传递、自回归解码——一行都不用改。
"""
import numpy as np

MOL = {
    "rna": dict(
        alphabet="ACGU",
        atoms=["P", "C4'", "C1'", "N"],   # N = 糖苷氮，嘌呤 N9 / 嘧啶 N1
        center=1,                          # 用 C4' 建 kNN 图
        radii=(10.0, 14.0, 18.0),
        d_max=24.0,
        virtual=False,
    ),
    "protein": dict(
        alphabet="ACDEFGHIKLMNPQRSTVWY",
        atoms=["N", "CA", "C", "O"],
        center=1,                          # 用 CA 建 kNN 图
        radii=(8.0, 10.0, 12.0),
        d_max=22.0,
        virtual=True,                      # 额外加一个虚拟 Cβ
    ),
}

N_RBF = 16
MAX_OFF = 16


def alphabet(mol="rna"):
    return MOL[mol]["alphabet"]


def n_letters(mol="rna"):
    return len(MOL[mol]["alphabet"])


def seq_to_idx(seq, mol="rna"):
    """ "ACGU" -> array([0,1,2,3])。表外字符映射到 0。"""
    a = MOL[mol]["alphabet"]
    m = {c: i for i, c in enumerate(a)}
    return np.array([m.get(c, 0) for c in seq], dtype=np.int64)


def idx_to_seq(idx, mol="rna"):
    a = MOL[mol]["alphabet"]
    return "".join(a[int(i)] for i in idx)


def n_feat_atoms(mol="rna"):
    return len(MOL[mol]["atoms"]) + (1 if MOL[mol]["virtual"] else 0)


def edge_dim(mol="rna"):
    """边特征维度。建模型的时候要传给 d_edge。"""
    n = n_feat_atoms(mol)
    return n * n * N_RBF + 2 * MAX_OFF + 1


def node_dim(mol="rna"):
    # RNA 多一维「这个位置配不配对」。见 node_features 的说明。
    return 6 + len(MOL[mol]["radii"]) + (1 if mol == "rna" else 0)


# ------------------------------------------------------------------ 基础几何

def virtual_cb(coords):
    """蛋白专用：从 N/CA/C 推一个虚拟 Cβ。甘氨酸没有真 Cβ，用虚拟的所有残基就统一了。

    常数来自 trRosetta / ProteinMPNN 的理想几何。
    """
    N, CA, C = coords[:, 0], coords[:, 1], coords[:, 2]
    b = CA - N
    c = C - CA
    a = np.cross(b, c)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA


def _dihedral(p0, p1, p2, p3):
    """四个点的二面角，弧度，范围 (-pi, pi]。输入都是 [...,3]。"""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / (np.linalg.norm(b1, axis=-1, keepdims=True) + 1e-8)
    v = b0 - np.sum(b0 * b1, axis=-1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=-1, keepdims=True) * b1
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1, v) * w, axis=-1)
    return np.arctan2(y, x)


def _sincos(angles):
    """[L,3] 弧度 -> [L,6] 的 sin/cos。

    为什么拆 sin/cos：角度是环形的，-179° 和 179° 其实挨着，
    直接喂数值会让模型以为它们差 358。
    """
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=-1).astype(np.float32)


def torsions_protein(coords):
    """主链 phi / psi / omega，返回 [L,6]。链两端没定义的补 0。"""
    L = coords.shape[0]
    N, CA, C = coords[:, 0], coords[:, 1], coords[:, 2]
    phi = np.zeros(L); psi = np.zeros(L); omg = np.zeros(L)
    if L > 1:
        phi[1:] = _dihedral(C[:-1], N[1:], CA[1:], C[1:])
        psi[:-1] = _dihedral(N[:-1], CA[:-1], C[:-1], N[1:])
        omg[1:] = _dihedral(CA[:-1], C[:-1], N[1:], CA[1:])
    return _sincos(np.stack([phi, psi, omg], axis=-1))


def torsions_rna(coords, chi_atoms=None):
    """RNA 的两个赝扭转角 + 真糖苷扭转角 chi，返回 [L,6]（各拆成 sin/cos）。

    **eta / theta —— Duarte & Pyle 的赝扭转角**

    RNA 主链有 6 个真扭转角（alpha 到 zeta），维度高且彼此相关，不好用。
    Duarte 和 Pyle 提出用两个赝扭转角概括，只需要 P 和 C4' 两种原子：

        eta(i)   = C4'(i-1) - P(i)   - C4'(i)   - P(i+1)
        theta(i) = P(i)     - C4'(i) - P(i+1)   - C4'(i+1)

    相邻两个角共享三个原子，像履带一样沿链滚动。A 型螺旋在 eta-theta
    平面上会聚成很紧的一团，这是第 3 关要你亲眼看到的事。

    出处：Duarte CM, Pyle AM. Stepping through an RNA structure: a novel
    approach to conformational analysis. J Mol Biol. 1998;284(5):1465-1478.
    （Pyle 组后来还出过用 C1' 代替 C4' 的变体，本项目用的是原始 C4' 版。）

    **chi —— 标准糖苷扭转角，不是赝角**

        嘌呤 (A, G)：O4' - C1' - N9 - C4
        嘧啶 (C, U)：O4' - C1' - N1 - C2

    描述碱基相对糖环平面的朝向。anti 约 ±(90°~180°)，syn 约 -90°~+90°。
    嘌呤出现 syn 的比例明显高于嘧啶，所以这个角直接带碱基身份的信息。

    chi_atoms: [L, 2, 3]，依次是 O4' 和碱基环上的那个碳（嘌呤 C4 / 嘧啶 C2）。
    由 prepare_data.py 解析时一并存进 npz。为 None 时 chi 这一列填 0——
    那说明数据集是旧版的，重跑一次 prepare_data.py 即可。
    """
    L = coords.shape[0]
    P, C4, C1, Nb = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
    eta = np.zeros(L); theta = np.zeros(L); chi = np.zeros(L)
    if L > 2:
        eta[1:-1] = _dihedral(C4[:-2], P[1:-1], C4[1:-1], P[2:])
        theta[1:-1] = _dihedral(P[1:-1], C4[1:-1], P[2:], C4[2:])
    if chi_atoms is not None:
        ca = np.asarray(chi_atoms, dtype=np.float64)
        chi = _dihedral(ca[:, 0], C1, Nb, ca[:, 1])   # O4' - C1' - N糖苷 - C碱基
        chi = np.nan_to_num(chi)                       # 缺原子的位置填 0
    ang = np.stack([eta, theta, chi], axis=-1)
    return _sincos(np.nan_to_num(ang))


def neighbor_counts(center, radii):
    """每个残基在若干半径内有多少个邻居。粗糙的"埋藏程度"。

    返回 [L, len(radii)]，除以 10 做了缩放（让数值别太大）。
    """
    d = np.linalg.norm(center[:, None, :] - center[None, :, :], axis=-1)
    np.fill_diagonal(d, 1e9)
    out = [(d < r).sum(axis=1) for r in radii]
    return (np.stack(out, axis=-1) / 10.0).astype(np.float32)


def knn_graph(center, k=16):
    """按中心原子距离建 k 近邻图。返回邻居下标 idx [L,k]，不含自己。

    L 比 k+1 小的时候用自己补齐。
    """
    L = center.shape[0]
    d = np.linalg.norm(center[:, None, :] - center[None, :, :], axis=-1)
    np.fill_diagonal(d, 1e9)
    kk = min(k, L - 1) if L > 1 else 0
    idx = np.argsort(d, axis=1)[:, :kk] if kk > 0 else np.zeros((L, 0), np.int64)
    if kk < k:
        pad = np.tile(np.arange(L)[:, None], (1, k - kk))
        idx = np.concatenate([idx, pad], axis=1)
    return idx.astype(np.int64)


def rbf(d, n_bins=N_RBF, d_min=2.0, d_max=22.0):
    """把一个距离摊成 n_bins 个高斯基函数的响应。

    为什么不直接用距离：一个标量线性进网络，表达力太弱。
    摊成一组"离 3Å 多近、离 5Å 多近 ..."，网络就能学非线性的距离依赖。
    """
    centers = np.linspace(d_min, d_max, n_bins, dtype=np.float32)
    sigma = (d_max - d_min) / n_bins
    return np.exp(-((d[..., None] - centers) / sigma) ** 2).astype(np.float32)


def relpos_onehot(resnum, idx, max_off=MAX_OFF):
    """邻居 j 在序列上离 i 多远，截断到 ±max_off 做 one-hot。

    为什么要这个：空间上挨着的两个残基，可能是序列上的邻居（i,i+1），
    也可能来自很远的另一段链——比如茎区里配对的两条链。含义完全不同。
    """
    off = resnum[idx] - resnum[:, None]
    off = np.clip(off, -max_off, max_off) + max_off
    oh = np.zeros(off.shape + (2 * max_off + 1,), dtype=np.float32)
    np.put_along_axis(oh, off[..., None], 1.0, axis=-1)
    return oh


def feat_atoms(coords, mol="rna"):
    """用于算边特征的原子集合。蛋白多一个虚拟 Cβ。"""
    if MOL[mol]["virtual"]:
        return np.concatenate([coords, virtual_cb(coords)[:, None, :]], axis=1)
    return coords


def edge_features(coords, idx, resnum, mol="rna"):
    """边特征：i 和 j 的每个原子两两之间的距离，各摊成 RBF，再拼上相对序列位置。

    RNA  4×4 = 16 对 -> 16*16 + 33 = 289 维
    蛋白 5×5 = 25 对 -> 25*16 + 33 = 433 维
    """
    atoms = feat_atoms(coords, mol)
    n = atoms.shape[1]
    ai = atoms[:, None, :, None, :]
    aj = atoms[idx][:, :, None, :, :]
    d = np.linalg.norm(ai - aj, axis=-1)
    L, k = idx.shape
    e = rbf(d.reshape(L, k, n * n), d_max=MOL[mol]["d_max"]).reshape(L, k, n * n * N_RBF)
    return np.concatenate([e, relpos_onehot(resnum, idx)], axis=-1)


def node_features(coords, mol="rna", partner=None, chi_atoms=None):
    """点特征。

    蛋白 9 维：phi/psi/omega 的 sin+cos（6）+ 三个半径的邻居数（3）
    RNA 10 维：eta/theta/chi 的 sin+cos（6）+ 邻居数（3）+ 配对状态（1）

    为什么 RNA 多一维配对状态：实测下来它是单个特征里最强的。
    RNA 的四种碱基在茎区和环区都大量出现，「周围挤不挤」区分力很弱；
    但「这个位置配不配对」直接把可选项压窄了——茎区偏 G/C，环区自由得多。
    蛋白那边则相反，埋藏程度和疏水性强相关，所以不需要这一维。

    partner=None 时这一列填 0。

    【一处更正，第 2 关查出来的】
    这里原来写着「partner 是从原子间距算出来的，不是从序列偷看来的，所以合法」。
    这句话说过头了。pairing.py 判氢键用的原子是嘌呤 N1 / 嘧啶 N3，第二个原子
    A->N6、G->O6、C->N4、U->O4 —— 四选一，要先知道碱基字母才知道取哪个。
    也就是说这一列的【实现】确实用到了答案。

    配对这个特征本身是合法的：纯几何判据（C1'-C1' 距离约 10.4 A，加上两个
    C1'->糖苷N 向量大致反平行、且与 C1'-C1' 轴共线）不需要碱基身份，
    这四个原子都是两种碱基共有的。所以要修的是实现，不是特征。

    同样的毛病 chi 更严重，而且无解：chi 的第四个原子（嘌呤 C4 / 嘧啶 C2）
    在碱基环上，而 gRNAde 的输入只有 P / C4' / N1-N9 三个珠子，没有这个原子。
    第 2 关实测这部分值 3.1~3.6 个百分点，全部拿不到。

    另外，显式给出配对标记本身就是相对 gRNAde 原文的一处简化：原文不给这个标记，
    让网络自己从几何里发现配对关系。第 3 关有了 kNN 图之后这一维可以直接删掉。
    第 7 关的「和原文差在哪」要记这两条。
    """
    t = (torsions_rna(coords, chi_atoms) if mol == "rna"
         else torsions_protein(coords))
    cols = [t, neighbor_counts(coords[:, MOL[mol]["center"]], MOL[mol]["radii"])]
    if mol == "rna":
        L = coords.shape[0]
        if partner is None:
            paired = np.zeros((L, 1), dtype=np.float32)
        else:
            paired = (np.asarray(partner) >= 0).astype(np.float32)[:, None]
        cols.append(paired)
    return np.concatenate(cols, axis=-1)


def featurize(coords, resnum, mol="rna", k=16, noise=0.0, rng=None,
              partner=None, chi_atoms=None):
    """一次算完一个分子要用的所有东西。

    noise：给坐标加高斯噪声（Å）。ProteinMPNN 训练时加 0.02，
    目的是别让模型靠晶体结构的微小细节作弊。
    """
    if noise > 0:
        rng = rng or np.random
        coords = coords + rng.randn(*coords.shape).astype(np.float32) * noise
    idx = knn_graph(coords[:, MOL[mol]["center"]], k)
    return {
        "V": node_features(coords, mol, partner, chi_atoms),
        "E": edge_features(coords, idx, resnum, mol),
        "idx": idx,
    }
