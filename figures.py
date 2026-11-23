import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
from matplotlib import font_manager
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
_avail = {fe.name for fe in font_manager.fontManager.ttflist}
FONT = next((fn for fn in ("Times New Roman", "Liberation Serif", "DejaVu Serif") if fn in _avail), "DejaVu Serif")

with open("results.json", encoding="utf-8") as fh:
    r = json.load(fh)
plt.rcParams.update({"font.family": FONT, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
                     "xtick.color": "#3d3d3a", "ytick.color": "#3d3d3a"})
COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
MK = {"A": "o", "B": "s", "C": "^"}
LS = {"A": "-", "B": "--", "C": "-"}
NAME = {"A": "А — більшість", "B": "Б — без групування", "C": "В — повна методика"}

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))

# (а) FPR залежно від кількості агрегаторів, delta = 1
Ms = [0, 4, 8, 16]
for m in ("A", "B", "C"):
    y = [100 * r["copiers"][str(M)][m]["fpr"][0] for M in Ms]
    ax[0].plot(Ms, y, LS[m], color=COL[m], marker=MK[m], ms=6, lw=2,
               markeredgecolor="white", markeredgewidth=1, label=NAME[m])
    ax[0].annotate(f"{y[-1]:.1f}", (Ms[-1], y[-1]), xytext=(5, 0),
                   textcoords="offset points", va="center", fontsize=9, color="#3d3d3a")
ax[0].set_xlabel("Кількість агрегаторів-копіювальників M")
ax[0].set_ylabel("Частка хибнопозитивних, %")
ax[0].set_xticks(Ms)
ax[0].set_xlim(-0.8, 18.5)
ax[0].set_ylim(0, 30)
ax[0].grid(axis="y", color="#e1e0d9", lw=0.6)
ax[0].set_title("а) зростання тиражування (δ = 1)", fontsize=10, loc="left")
ax[0].legend(frameon=False, fontsize=8.5, loc="center right", bbox_to_anchor=(1.0, 0.6))

# (б) FPR і повнота методики В залежно від частки виявленої залежності
ds = [0.0, 0.25, 0.5, 0.75, 1.0]
fprC = [100 * r["delta"][str(d)]["C"]["fpr"][0] for d in ds]
recC = [100 * r["delta"][str(d)]["C"]["recall"][0] for d in ds]
ax[1].plot(ds, fprC, "-", color=COL["C"], marker="^", ms=6, lw=2,
           markeredgecolor="white", markeredgewidth=1)
ax[1].plot(ds, recC, ":", color="#3d3d3a", marker="D", ms=5, lw=2,
           markeredgecolor="white", markeredgewidth=1)
ax[1].annotate("хибнопозитивні, %", (0.5, fprC[2]), xytext=(0, 14),
               textcoords="offset points", ha="center", fontsize=9, color="#3d3d3a")
ax[1].annotate("повнота, %", (0.5, recC[2]), xytext=(0, 9),
               textcoords="offset points", ha="center", fontsize=9, color="#3d3d3a")
ax[1].annotate(f"{fprC[0]:.1f}", (0.0, fprC[0]), xytext=(0, 7), textcoords="offset points",
               ha="center", fontsize=9, color="#3d3d3a")
ax[1].annotate(f"{fprC[-1]:.1f}", (1.0, fprC[-1]), xytext=(0, 8), textcoords="offset points",
               ha="center", fontsize=9, color="#3d3d3a")
for x, y in ((0.0, recC[0]), (1.0, recC[-1])):
    ax[1].annotate(f"{y:.1f}", (x, y), xytext=(0, 7), textcoords="offset points",
                   ha="center", fontsize=9, color="#3d3d3a")
ax[1].set_xlabel("Частка виявлених залежностей δ")
ax[1].set_ylabel("%")
ax[1].set_xticks(ds)
ax[1].set_xlim(-0.08, 1.08)
ax[1].set_ylim(0, 100)
ax[1].grid(axis="y", color="#e1e0d9", lw=0.6)
ax[1].set_title("б) методика В, M = 8", fontsize=10, loc="left")

from matplotlib.ticker import FuncFormatter
comma = FuncFormatter(lambda v, p: f"{v:g}".replace(".", ","))
for a in ax:
    a.yaxis.set_major_formatter(comma); a.xaxis.set_major_formatter(comma)
for a in ax:
    for t in a.texts:
        t.set_text(t.get_text().replace(".", ","))
fig.tight_layout()
fig.savefig("fig2.png", dpi=300)
print("Рисунок збережено у файл fig2.png")
