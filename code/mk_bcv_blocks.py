"""fig_bcv_blocks: the bi-cross-validation split, showing which blocks build the
prediction and which block is scored. Schematic only, no data.
Project convention: matplotlib owns the geometry, PDF out for Inkscape."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotstyle as ps          # shared thesis figure style (SciencePlots base)
ps.apply()

from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(0.66 * ps.WIDTH, 2.45))
W1, W2, H1, H2 = 1.55, 0.95, 1.0, 0.62          # widths (units), heights (rows)
x0, y0 = 0.0, 0.0
cells = {                                        # (x, y, w, h, face, label)
    "Y11": (x0,      y0+H2, W1, H1, "0.90", r"$Y_{11}$"),
    "Y12": (x0+W1,   y0+H2, W2, H1, "0.90", r"$Y_{12}$"),
    "Y21": (x0,      y0,    W1, H2, "0.75", r"$Y_{21}$"),
    "Y22": (x0+W1,   y0,    W2, H2, "1.00", r"$Y_{22}$"),
}
for k,(x,y,w,h,fc,lab) in cells.items():
    ls = "--" if k == "Y22" else "-"
    ax.add_patch(Rectangle((x,y), w, h, facecolor=fc, edgecolor="black",
                           lw=1.2, linestyle=ls, zorder=1))
    ax.text(x+w/2, y+h/2, lab, ha="center", va="center", zorder=3)

ax.text(x0+W1/2,      y0+H2+H1+0.11, "training units", ha="center", fontsize=9)
ax.text(x0+W1+W2/2,   y0+H2+H1+0.11, "held-out",       ha="center", fontsize=9)
ax.text(x0-0.10, y0+H2+H1/2, "training\nrows", ha="right", va="center", fontsize=9)
ax.text(x0-0.10, y0+H2/2,    "held-out\nrows", ha="right", va="center", fontsize=9)

# loadings come from the whole training-row strip; scores from Y21 predict Y22
ax.annotate("", xy=(x0+W1+W2+0.44, y0+H2+H1/2), xytext=(x0+W1+W2+0.06, y0+H2+H1/2),
            arrowprops=dict(arrowstyle="->", lw=1.1))
ax.text(x0+W1+W2+0.52, y0+H2+H1/2, r"loadings $\Phi_q$", va="center", fontsize=9)
ax.annotate("", xy=(x0+W1+0.22, y0+H2/2), xytext=(x0+W1-0.26, y0+H2/2),
            arrowprops=dict(arrowstyle="->", lw=1.4, color="black"))
ax.text(x0+W1-0.02, y0+H2/2+0.15, "predict", ha="center", fontsize=8.5)
ax.text(x0+W1+W2/2, y0-0.24, "scored, never seen", ha="center", fontsize=8.5,
        style="italic")

ax.set_xlim(-0.95, x0+W1+W2+1.95); ax.set_ylim(-0.45, y0+H2+H1+0.40)
ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout()
fig.savefig("figures/fig_bcv_blocks.pdf", bbox_inches="tight")
fig.savefig("figures/fig_bcv_blocks.png", dpi=200, bbox_inches="tight")
