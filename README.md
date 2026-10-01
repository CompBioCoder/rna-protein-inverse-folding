# rna-protein-inverse-folding

一个学习用的极简重实现，用于理解 **gRNAde**（RNA 逆向折叠）与 **ProteinMPNN**（蛋白逆向折叠）
的共同框架与关键差异。不是官方实现，不要用于实际设计任务。
官方实现见 [gRNAde](https://github.com/chaitjo/geometric-rna-design) 与
[ProteinMPNN](https://github.com/dauparas/ProteinMPNN)，两者均为 MIT 许可。

核心观点：这两个方法的差别只在**表示层**——字母表、骨架原子、赝扭转角。
kNN 建图、RBF 距离展开、相对序列位置、消息传递编码器、随机顺序自回归解码、
固定位点约束，这些一行都不用改。所以本仓库用**一个参数**切换两种分子。

```python
mol = "rna"       # 4 个碱基，P / C4' / C1' / 糖苷氮，eta / theta / 赝chi
mol = "protein"   # 20 个氨基酸，N / CA / C / O (+虚拟Cβ)，phi / psi / omega
```

## 跑起来

```bash
conda activate grnade_proteinmpnn      # 需要 torch 和 numpy
python tests/test_geometry.py          # 纯 numpy，应 8/8
python prepare_data.py                 # 下载 RNA 结构，建数据集
python game.py                         # 七关路线图与进度
```

## 大文件都在外置盘

仓库里三个软链接，Mac 本地不存任何数据：

```
datasets -> /Volumes/Backup_1t/inverse_folding/datasets   数据集 npz
pdbs     -> /Volumes/Backup_1t/inverse_folding/pdbs       下载的结构文件缓存
runs     -> /Volumes/Backup_1t/inverse_folding/runs       模型权重
```

三个都在 `.gitignore` 里。下载过的 PDB 会缓存到 `pdbs/`，重跑不用再下一遍。
盘没挂载时脚本会直接说清楚，而不是抛一个 `FileNotFoundError`。

换台机器跑、或者不想用软链接，用环境变量覆盖：

```bash
export RNAIF_DATA=/somewhere/datasets
export RNAIF_RUNS=/somewhere/runs
export RNAIF_PDBS=/somewhere/pdbs
```

`progress.json` 是例外——几 KB 的通关记录，放在仓库里跟着 git 一起备份，
这样盘没插也看得到进度。

## 七关路线

每一关只改一件事，验收条件是「比上一关好多少」，不是绝对数字。

| 关 | 加了什么 | 验收 |
|---|---|---|
| 1 | 单个埋藏度特征 + 线性层 | 恢复率 > 最常见碱基基准线 + 1.5pt |
| 2 | eta/theta/赝chi + 多层感知机 | ≥ 第 1 关 + 1.5pt |
| 3 | kNN 图 + RBF 边特征 | 几何自测 8/8，且 ≥ 第 2 关 + 1.5pt |
| 4 | 消息传递编码器（一次性预测） | ≥ 第 3 关 + 3pt |
| 5 | 随机顺序自回归解码器 | 模型自测 8/8，配对有效率 ≥ 第 4 关 + 5pt |
| 6 | 长训练 + 分层评估 + 点特征消融 | ≥ 第 5 关 |
| 7 | 固定位点 / 温度 / logits 偏置 / 蛋白对照 | 固定位点 100% 保住，蛋白跑通 |

## 两个指标

**序列恢复率**：逐位猜中天然碱基的比例。在 RNA 上这个指标偏钝——字母表只有 4 个，
随机猜 25%，按最常见碱基猜接近 30%。看它必须带着基准线看。

**配对有效率**：天然结构里配对的位置上，设计出的两个碱基还能不能构成合法配对
（WC 或 GU 摇摆）。这个指标诚实得多：一个「每个位置独立预测」的模型在它上面会很难看，
因为算第 7 位时它不知道第 40 位会选什么。第 4 关和第 5 关在这个数上的落差，
就是自回归的价值。这个现象在蛋白上看不到这么干净。

## 自测

```bash
python tests/test_geometry.py   # 几何：二面角、旋转平移不变性、镜像敏感性
python tests/test_model.py      # 模型：形状、因果性、教师强制与逐步解码一致性
```

`test_model.py` 第 3、4 项专门查自回归解码器有没有偷看到自己要预测的那个碱基。
这类 bug 会让训练损失和恢复率都很好看，但采样生成时全盘崩掉，
从训练曲线上完全看不出来。

## 与官方实现的差异

1. 批大小固定为 1，不做 padding/mask——官方按长度分桶批处理
2. 采样时每步重算整张图，没做解码器状态缓存
3. 单构象。gRNAde 支持多构象输入，把同一条 RNA 的多个结构状态合并成一个表示
4. 没有自洽性验证（设计序列 → 结构预测 → 比对原骨架）
5. 数据量差着量级：本仓库是几百条链，官方是数万条

## 目录

```
minif/          包。features / pairing / data / model / engine / scoreboard / paths
levels/         七关的参考实现
tests/          两套自测
notes/          每关的学习记录
notebooks/      探索用
game.py         通关进度
prepare_data.py 下载 + 解析 + 配对检测 + 按簇去冗余划分
```

## 许可

代码 MIT（见 LICENSE）。结构数据来自 RCSB PDB，各有其使用条款。
