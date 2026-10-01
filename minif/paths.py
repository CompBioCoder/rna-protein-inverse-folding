# -*- coding: utf-8 -*-
"""大文件放哪儿。

本仓库的约定：三个软链接指向外置盘，Mac 本地只留代码和笔记。

    datasets -> /Volumes/Backup_1t/inverse_folding/datasets    数据集 npz
    pdbs     -> /Volumes/Backup_1t/inverse_folding/pdbs        下载的结构文件缓存
    runs     -> /Volumes/Backup_1t/inverse_folding/runs        模型权重

这三个都在 .gitignore 里，不进仓库。

progress.json 是例外：通关进度只有几 KB，放仓库根目录跟着 git 一起备份，
不放外置盘——不然换台机器或者盘没插，进度就看不到了。

不想用软链接（比如在另一台机器上跑），用环境变量覆盖：

    export RNAIF_DATA=/somewhere/datasets
    export RNAIF_RUNS=/somewhere/runs
    export RNAIF_PDBS=/somewhere/pdbs

写进 ~/.zshrc 长期生效；只想这次生效就在跑脚本前 export。
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.environ.get("RNAIF_DATA") or os.path.join(ROOT, "datasets")
RUNS_DIR = os.environ.get("RNAIF_RUNS") or os.path.join(ROOT, "runs")
PDB_DIR = os.environ.get("RNAIF_PDBS") or os.path.join(ROOT, "pdbs")
PROGRESS = os.path.join(ROOT, "progress.json")


def require(path, what):
    """目录不可用就报人话，而不是一句 FileNotFoundError。"""
    if os.path.isdir(path):
        return path
    if os.path.islink(path):
        raise SystemExit(
            "%s 用不了：\n  %s -> %s\n"
            "软链接指向的位置不在。多半是外置盘没插或没挂载，插上再跑。\n"
            "想换个位置：export RNAIF_DATA / RNAIF_RUNS / RNAIF_PDBS"
            % (what, path, os.readlink(path)))
    raise SystemExit(
        "%s 用不了：找不到 %s\n仓库里的软链接是不是丢了？见 README。" % (what, path))


def data_file(mol):
    require(DATA_DIR, "数据目录 datasets")
    return os.path.join(DATA_DIR, "%s.npz" % mol)


def run_file(name):
    """权重落点。不管在哪个目录下跑脚本，落点都一样。"""
    require(RUNS_DIR, "权重目录 runs")
    return os.path.join(RUNS_DIR, name)


def pdb_file(pid):
    """下载的结构文件缓存。有了它，重跑 prepare_data 不用再下一遍。"""
    require(PDB_DIR, "结构缓存目录 pdbs")
    return os.path.join(PDB_DIR, "%s.pdb" % pid.upper())


def available(path):
    return os.path.isdir(path)


def describe():
    tag = lambda p, env: "%s%s%s" % (
        p, "  (来自 %s)" % env if os.environ.get(env) else "",
        "" if os.path.isdir(p) else "   [不可用]")
    return ("数据 %s\n权重 %s\n结构缓存 %s\n进度 %s"
            % (tag(DATA_DIR, "RNAIF_DATA"), tag(RUNS_DIR, "RNAIF_RUNS"),
               tag(PDB_DIR, "RNAIF_PDBS"), PROGRESS))
