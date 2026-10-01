# -*- coding: utf-8 -*-
"""MiniMPNN：gRNAde / ProteinMPNN 的最小可用复刻。RNA 和蛋白共用这一套。

两部分：
  编码器 Encoder —— 只看结构，把每个残基变成一个向量。不含任何序列信息。
  解码器 Decoder —— 按一个随机顺序逐个残基预测氨基酸，
                    每一步只能看到"已经决定好的"邻居的氨基酸身份。

为了让你能一次看懂，做了三处简化（都在第 7 天的"和原文差在哪"里列着）：
  1. 批大小固定为 1（一次一个蛋白），因此完全不需要 padding 和 mask
  2. 采样时每步重算整张图，没做原文的缓存加速（慢，但短链无所谓）
  3. 编码器层数和维度都调小了
"""
import torch
import torch.nn as nn


def mlp(d_in, d_hidden, d_out):
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(),
                         nn.Linear(d_hidden, d_hidden), nn.GELU(),
                         nn.Linear(d_hidden, d_out))


class EncLayer(nn.Module):
    """一层消息传递：每个残基收集 k 个邻居传来的消息，加到自己身上。

    形状：h [L,D]，e [L,k,De]，idx [L,k] -> h [L,D]
    邻居用 mean 聚合，所以邻居的先后顺序不影响结果（图上本来也没有顺序）。
    """

    def __init__(self, d, d_edge, d_hidden):
        super().__init__()
        self.msg = mlp(2 * d + d_edge, d_hidden, d)
        self.ff = mlp(d, d_hidden, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, h, e, idx):
        hj = h[idx]                                   # [L,k,D] 邻居的向量
        hi = h.unsqueeze(1).expand_as(hj)             # [L,k,D] 自己的向量，复制 k 份
        m = self.msg(torch.cat([hi, hj, e], dim=-1))  # [L,k,D]
        h = self.norm1(h + m.mean(dim=1))
        h = self.norm2(h + self.ff(h))
        return h


class DecLayer(nn.Module):
    """一层解码：和编码层几乎一样，区别是消息里多了"邻居的氨基酸身份"，
    而且这一项对"还没轮到的邻居"会被乘上 0 抹掉。

    mask_bw [L,k,1]：邻居 j 的解码次序排在 i 前面就是 1，否则 0。
    这一个乘法就是整个自回归的全部机关。
    """

    def __init__(self, d, d_edge, d_seq, d_hidden):
        super().__init__()
        self.msg = mlp(d + d + d + d_seq + d_edge, d_hidden, d)
        self.ff = mlp(d, d_hidden, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, h_dec, h_enc, s_emb, e, idx, mask_bw):
        hj_enc = h_enc[idx]                           # [L,k,D] 邻居的结构信息，一直可见
        hj_dec = h_dec[idx] * mask_bw                 # [L,k,D] 已决定邻居的解码状态
        sj = s_emb[idx] * mask_bw                     # [L,k,Ds] 已决定邻居的氨基酸
        hi = h_dec.unsqueeze(1).expand_as(hj_enc)
        m = self.msg(torch.cat([hi, hj_enc, hj_dec, sj, e], dim=-1))
        h = self.norm1(h_dec + m.mean(dim=1))
        h = self.norm2(h + self.ff(h))
        return h


class MiniMPNN(nn.Module):
    def __init__(self, d_node=9, d_edge=289, n_letters=4, d_model=128, d_hidden=128,
                 n_enc=3, n_dec=3, d_seq=32, use_node_feats=True):
        super().__init__()
        self.use_node_feats = use_node_feats
        self.n_letters = n_letters
        self.embed_V = nn.Linear(d_node, d_model)
        self.embed_E = nn.Linear(d_edge, d_model)
        self.enc = nn.ModuleList([EncLayer(d_model, d_model, d_hidden) for _ in range(n_enc)])
        self.embed_S = nn.Embedding(n_letters, d_seq)
        self.dec = nn.ModuleList([DecLayer(d_model, d_model, d_seq, d_hidden) for _ in range(n_dec)])
        self.out = nn.Linear(d_model, n_letters)

    def encode(self, V, E, idx):
        """只吃结构，吐 [L, d_model]。"""
        h = self.embed_V(V)
        if not self.use_node_feats:
            h = torch.zeros_like(h)      # 第 6 天的消融实验：点特征全清零
        e = self.embed_E(E)
        for layer in self.enc:
            h = layer(h, e, idx)
        return h, e

    @staticmethod
    def backward_mask(order, idx):
        """order [L]：order[t] = 第 t 个被解码的残基编号。
        返回 mask_bw [L,k,1]，邻居排在自己前面才为 1。"""
        L = order.shape[0]
        rank = torch.empty(L, dtype=torch.long, device=order.device)
        rank[order] = torch.arange(L, device=order.device)
        return (rank[idx] < rank.unsqueeze(1)).unsqueeze(-1).to(torch.float32)

    def decode(self, h_enc, e, idx, seq, order):
        """教师强制：一次算出所有位置的 logits [L,20]。
        训练时用。seq 是真实序列 [L]，但第 i 位的答案绝不会流到第 i 位的 logits。"""
        mask_bw = self.backward_mask(order, idx)
        s = self.embed_S(seq)
        h = h_enc
        for layer in self.dec:
            h = layer(h, h_enc, s, e, idx, mask_bw)
        return self.out(h)

    def forward(self, V, E, idx, seq, order):
        h_enc, e = self.encode(V, E, idx)
        return self.decode(h_enc, e, idx, seq, order)

    @torch.no_grad()
    def sample(self, V, E, idx, order, temperature=0.1, fixed=None, bias=None):
        """真正地逐个生成。fixed: dict{位置: 氨基酸下标}，这些位置锁死不改。

        实现上每一步都把整张图重算一遍（O(L) 次前向），短蛋白够用。
        注意：固定位点要排在 order 的最前面，否则它们的信息传不给别人。
        """
        L = V.shape[0]
        h_enc, e = self.encode(V, E, idx)
        seq = torch.zeros(L, dtype=torch.long, device=V.device)
        fixed = fixed or {}
        for p, a in fixed.items():
            seq[p] = a
        mask_bw = self.backward_mask(order, idx)
        decided = torch.zeros(L, dtype=torch.bool, device=V.device)
        for p in fixed:
            decided[p] = True
        for t in range(L):
            i = int(order[t])
            if bool(decided[i]):
                continue
            # 只有"已决定"的邻居才允许贡献序列信息
            m = mask_bw * decided[idx].unsqueeze(-1).to(torch.float32)
            s = self.embed_S(seq)
            h = h_enc
            for layer in self.dec:
                h = layer(h, h_enc, s, e, idx, m)
            logits = self.out(h[i])
            if bias is not None:
                logits = logits + bias
            probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
            seq[i] = int(torch.multinomial(probs, 1).item())
            decided[i] = True
        return seq
