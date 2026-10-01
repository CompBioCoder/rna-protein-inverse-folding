"""统一作图设置。每个 notebook 开头:from minif.plotting import setup; plt = setup()"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# 分类配色，固定顺序，不要循环使用。这三个颜色的任意两两组合
# 在正常视觉和三种色觉缺陷下都能分开（验证过）。超过三个系列就别再加颜色了，
# 改成分面小图或者归到"其他"。
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a"]      # 蓝 / 橙 / 青

# 连续量用单一色相由浅到深（热图、按大小着色）。永远不要用彩虹色。
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

INK = "#0b0b0b"
MUTED = "#52514e"


# 中文字体回退。Arial 没有中文字形，只写 Arial 的话图里的中文会变成方框。
# 把 Arial 放第一位保证西文还是 Arial，中文落到后面的字体上。
# matplotlib >= 3.6 支持逐字符回退。
CJK_CANDIDATES = ["PingFang SC", "Hiragino Sans GB", "Heiti SC", "Songti SC",
                  "Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei",
                  "WenQuanYi Zen Hei"]


def _font_stack():
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    cjk = [f for f in CJK_CANDIDATES if f in have]
    if not cjk:
        print("[plotting] 没找到中文字体，图里的中文可能显示成方框。"
              "候选：" + "、".join(CJK_CANDIDATES[:4]))
    return ["Arial", "Helvetica"] + cjk + ["DejaVu Sans"]


def setup():
    mpl.rcParams.update({
        "font.size": 18,
        "font.family": "sans-serif",
        "font.sans-serif": _font_stack(),
        "axes.unicode_minus": False,   # 用 CJK 字体时负号会变方框，关掉
        "pdf.fonttype": 42,   # 存矢量 PDF 时文字保持可编辑
        "ps.fonttype": 42,
    })
    # 以下是作图规范：网格和边框要"退后"，数据要"站出来"
    mpl.rcParams.update({
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": "medium",
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": "#e5e4e0",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 8,
        "figure.dpi": 110,
    })
    return plt


def seq_cmap(name="minif_blue"):
    """单色相连续色标，给热图用。"""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, SEQ)


def tidy(ax, ygrid=True):
    """统一收尾：只留横向网格，并放到数据后面。"""
    ax.set_axisbelow(True)
    ax.grid(ygrid, axis="y")
    ax.grid(False, axis="x")
    return ax
