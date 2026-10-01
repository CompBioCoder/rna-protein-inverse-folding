# -*- coding: utf-8 -*-
"""几何自测（纯 numpy，不需要 PyTorch）。改了 features.py 就跑一遍。

    python tests_geometry.py

8 项全过是第 3 关 BOSS 的条件之一。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from minif import features as F, scoreboard
from minif.features import _dihedral


# ------------------------------------------------ 造测试结构（只用于自测）
def place(a, b, c, length, angle, torsion):
    bc = c - b; bc = bc / np.linalg.norm(bc)
    n = np.cross(b - a, bc); n = n / np.linalg.norm(n)
    m = np.stack([bc, np.cross(n, bc), n], axis=1)
    d = np.array([-length*np.cos(angle), length*np.sin(angle)*np.cos(torsion),
                  length*np.sin(angle)*np.sin(torsion)])
    return c + m @ d


def protein_helix(L=24, phi=-57., psi=-47., omega=180.):
    """理想 alpha 螺旋主链。"""
    D = np.deg2rad
    N = [np.array([0., 0., 0.])]; CA = [np.array([1.458, 0., 0.])]
    C = [place(np.array([0., 1., 0.]), N[0], CA[0], 1.525, D(111.0), D(psi))]
    for _ in range(1, L):
        n = place(N[-1], CA[-1], C[-1], 1.329, D(116.2), D(psi))
        ca = place(CA[-1], C[-1], n, 1.458, D(121.7), D(omega))
        c = place(C[-1], n, ca, 1.525, D(111.0), D(phi))
        N.append(n); CA.append(ca); C.append(c)
    O = [place(N[i], CA[i], C[i], 1.231, D(120.8), D(psi + 180)) for i in range(L)]
    return np.stack([np.stack(N), np.stack(CA), np.stack(C), np.stack(O)], axis=1).astype(np.float32)


def rna_helix(L=20, twist=32.7, rise=2.81):
    """A 型螺旋式的粗粒化 RNA：四种原子各放在自己的圆柱面上。

    半径和相位是示意值，不是真实 A 型 RNA 的精确坐标。
    这个结构只用来检查两件事：赝扭转角沿规则螺旋应当恒定；旋转平移不变性。
    """
    spec = [(8.8, 0.0), (9.0, 18.0), (7.0, 40.0), (4.5, 62.0)]   # (半径, 相位°)
    out = np.zeros((L, 4, 3), dtype=np.float32)
    for i in range(L):
        for a, (r, ph) in enumerate(spec):
            t = np.deg2rad(twist * i + ph)
            out[i, a] = (r*np.cos(t), r*np.sin(t), rise*i)
    return out


def praxeolitic(p0, p1, p2, p3):
    b0 = -1.0*(p1-p0); b1 = p2-p1; b2 = p3-p2
    b1 = b1/np.linalg.norm(b1)
    v = b0-np.dot(b0, b1)*b1; w = b2-np.dot(b2, b1)*b1
    return np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))


def rand_rot(seed=3):
    Q, _ = np.linalg.qr(np.random.RandomState(seed).randn(3, 3))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def main():
    ok = []
    rng = np.random.RandomState(1)

    # 1. 二面角公式：和通用实现逐点比
    err = max(abs(_dihedral(*(p := rng.randn(4, 3)*3)) - praxeolitic(*p)) for _ in range(500))
    print("1. 二面角 vs 参考实现，最大误差 %.1e" % err)
    ok.append(err < 1e-6)

    # 2. 蛋白：理想 alpha 螺旋应读回 -57 / -47 / 180
    P24 = protein_helix()
    d = F.torsions_protein(P24)
    deg = lambda s, c: np.degrees(np.arctan2(d[:, s], d[:, c]))
    phi, psi, omg = deg(0, 3), deg(1, 4), deg(2, 5)
    print("2. 蛋白 alpha 螺旋 phi=%.1f psi=%.1f |omega|=%.1f （期望 -57 / -47 / 180）"
          % (phi[10], psi[10], abs(omg[10])))
    ok.append(abs(phi[10]+57) < .5 and abs(psi[10]+47) < .5 and abs(abs(omg[10])-180) < .5)

    # 3. 蛋白：手性与尺寸（不依赖角度约定的判据）
    ca = P24[:, 1]
    chir = np.dot(np.cross(ca[6]-ca[5], ca[7]-ca[6]), ca[8]-ca[7])
    d1, d4 = np.linalg.norm(ca[6]-ca[5]), np.linalg.norm(ca[9]-ca[5])
    dcb = np.linalg.norm(F.virtual_cb(P24)[5]-ca[5])
    print("3. 蛋白 手性 %+.1f（右手为正） CA(i,i+1)=%.2f(~3.8) CA(i,i+4)=%.2f(~6.2) CA-CB=%.2f(~1.53)"
          % (chir, d1, d4, dcb))
    ok.append(chir > 0 and abs(d1-3.8) < .1 and abs(d4-6.2) < .6 and abs(dcb-1.53) < .1)

    # 4. RNA：规则螺旋上赝扭转角应当沿链恒定
    R = rna_helix()
    t = F.torsions_rna(R)
    eta = np.degrees(np.arctan2(t[:, 0], t[:, 3]))[2:-2]
    theta = np.degrees(np.arctan2(t[:, 1], t[:, 4]))[2:-2]
    chi = np.degrees(np.arctan2(t[:, 2], t[:, 5]))[2:-2]
    sd = lambda a: np.std(((a - a[0] + 180) % 360) - 180)
    print("4. RNA 规则螺旋 eta=%.1f±%.2f  theta=%.1f±%.2f  chi=%.1f±%.2f （标准差应≈0）"
          % (eta.mean(), sd(eta), theta.mean(), sd(theta), chi.mean(), sd(chi)))
    ok.append(sd(eta) < .1 and sd(theta) < .1 and sd(chi) < .1)

    # 5. 旋转+平移不变性 —— 五问第 1 问的实证
    res5 = []
    for mol, xyz in (("rna", R), ("protein", P24)):
        L = xyz.shape[0]; rn = np.arange(L, dtype=np.int64)
        Q = rand_rot()
        xyz2 = ((xyz.reshape(-1, 3) @ Q.T) + np.array([13., -7., 42.])).reshape(xyz.shape).astype(np.float32)
        f1, f2 = F.featurize(xyz, rn, mol, k=8), F.featurize(xyz2, rn, mol, k=8)
        eV = np.abs(f1["V"]-f2["V"]).max()
        # 固定同一张邻居表再比边特征：规则螺旋上 i±m 距离完全相等，
        # argsort 谁先谁后由浮点噪声决定。模型对邻居取 mean，本来就不看顺序。
        eE = np.abs(F.edge_features(xyz, f1["idx"], rn, mol)
                    - F.edge_features(xyz2, f1["idx"], rn, mol)).max()
        same = np.array_equal(np.sort(f1["idx"], 1), np.sort(f2["idx"], 1))
        print("5. %-7s 旋转平移后 V 差 %.1e  E 差 %.1e  邻居集合一致 %s" % (mol, eV, eE, same))
        res5.append(eV < 1e-3 and eE < 1e-3 and same)
    ok.append(all(res5))

    # 6. 镜像应改变特征（否则分不清 L 型和 D 型、右手和左手螺旋）
    res6 = []
    for mol, xyz in (("rna", R), ("protein", P24)):
        L = xyz.shape[0]; rn = np.arange(L, dtype=np.int64)
        m = xyz.copy(); m[..., 0] *= -1
        dv = np.abs(F.node_features(xyz, mol) - F.node_features(m, mol)).max()
        print("6. %-7s 镜像后 V 差 %.3f（应明显不为 0）" % (mol, dv))
        res6.append(dv > .1)
    ok.append(all(res6))

    # 7. 形状与短链兜底
    rn = np.arange(20, dtype=np.int64)
    fr = F.featurize(R, rn, "rna", k=8)
    fp = F.featurize(P24, np.arange(24, dtype=np.int64), "protein", k=8)
    fs = F.featurize(R[:4], np.arange(4, dtype=np.int64), "rna", k=16)
    print("7. RNA V%s E%s（边应 %d 维）　蛋白 V%s E%s（边应 %d 维）　L=4,k=16 -> idx%s"
          % (fr["V"].shape, fr["E"].shape, F.edge_dim("rna"),
             fp["V"].shape, fp["E"].shape, F.edge_dim("protein"), fs["idx"].shape))
    ok.append(fr["E"].shape[-1] == F.edge_dim("rna") == 289
              and fp["E"].shape[-1] == F.edge_dim("protein") == 433
              and fr["V"].shape == (20, F.node_dim("rna")) == (20, 10)
              and fp["V"].shape == (24, F.node_dim("protein")) == (24, 9)
              and fs["idx"].shape == (4, 16))

    # 8. kNN：规则螺旋上最近邻应当是序列上对称的那几个
    nb = np.sort(fr["idx"][10]) - 10
    nbp = np.sort(fp["idx"][12]) - 12
    print("8. 近邻偏移  RNA %s   蛋白 %s （都应左右对称）" % (nb, nbp))
    sym = lambda v: set(v.tolist()) == set((-v).tolist()) and 0 not in v.tolist()
    ok.append(sym(nb) and sym(nbp))

    n = sum(ok)
    print("\n%s  %d/8" % ("全部通过" if n == 8 else "有不通过的项", n))
    scoreboard.record(3, geometry_tests=n)
    return 0 if n == 8 else 1


if __name__ == "__main__":
    sys.exit(main())
