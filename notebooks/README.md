# notebooks

七关的 notebook 版。和 `levels/*.py` 是**同一份内容的两种形态**，脚本没有删：

- **notebook**：一格一格跑，看得见每一步的形状和中间结果，带图。学的时候用这个。
- **脚本**：一条命令跑完整关，适合重跑、改参数、后台跑长训练。

两边都从 `minif/` 这个包里导入同样的函数，不存在两份实现。

## 怎么开

```bash
conda activate grnade_proteinmpnn
cd ~/Documents/GitHub/rna-protein-inverse-folding
jupyter lab
```

然后打开 `notebooks/level1_baseline.ipynb`，从上往下一格一格跑。
每个 notebook 开头会自动找到仓库根目录，所以从哪个目录启动 jupyter 都不影响。

## 每关的结构

每个 notebook 都是「部分 0、部分 1 ...」的结构，第一格会列出这一关有几个部分。
一个部分 = 一段说明 + 一到几格代码。

最后一个部分统一是「BOSS + 自测」，里面有一格是

```
!cd .. && python game.py check N
```

跑它就判定过没过。分数由前面的格子自动写进 `progress.json`。

## 图

图的配色在 `minif/plotting.py` 里统一定义，三个分类色经过色觉缺陷校验，
连续量用单一色相。要改风格改那一个文件，七个 notebook 都跟着变。
