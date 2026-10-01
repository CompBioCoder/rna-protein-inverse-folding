# -*- coding: utf-8 -*-
"""存通关进度。各关脚本跑完自动写，game.py 负责读和判定。"""
import json
import os

from . import paths

# 进度只有几 KB，放仓库根目录跟着 git 一起备份，不放外置盘：
# 盘没插的时候也要能看见自己打到第几关。
PATH = paths.PROGRESS


def read():
    if not os.path.exists(PATH):
        return {"levels": {}, "hints": {}, "achievements": []}
    try:
        with open(PATH, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return {"levels": {}, "hints": {}, "achievements": []}
    d.setdefault("levels", {}); d.setdefault("hints", {}); d.setdefault("achievements", [])
    return d


def write(d):
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


def _plain(v):
    """numpy 的整数/浮点不能直接写进 JSON，统一转成 Python 类型。"""
    if hasattr(v, "item"):
        try:
            v = v.item()
        except Exception:
            pass
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return float(v)
    return v


def record(level, **metrics):
    """记一关的成绩。同一关多次跑，每个指标只留最好的那次。"""
    d = read()
    k = str(level)
    cur = d["levels"].get(k, {})
    for name, v in metrics.items():
        if v is None:
            continue
        v = _plain(v)
        if isinstance(v, float) and v != v:      # NaN 不记
            continue
        if isinstance(v, (int, float)) and name in cur and isinstance(cur[name], (int, float)):
            cur[name] = max(cur[name], v)
        else:
            cur[name] = v
    d["levels"][k] = cur
    write(d)
    print("\n[记分板] 第 %s 关：%s" % (level, "  ".join(
        "%s=%s" % (n, ("%.3f" % v) if isinstance(v, float) else v) for n, v in sorted(cur.items()))))
    print("[记分板] 挑战 BOSS：python game.py check %s" % level)
    return cur


def use_hint(level):
    d = read()
    k = str(level)
    d["hints"][k] = d["hints"].get(k, 0) + 1
    write(d)
    return d["hints"][k]


def unlock(name):
    d = read()
    if name not in d["achievements"]:
        d["achievements"].append(name)
        write(d)
        return True
    return False
