"""make_metric_matrix.py — the 2x2 metric drawn at a few points of the (t, theta) cylinder.

The cylinder is drawn whole, and each sampled point is called out to a small heat map of the
metric matrix there.

In each call-out, colour is the metric in a chart where time has been rescaled by the constant
c below, so that one diverging scale can be shared by every panel; the number printed in each
cell is the raw entry, in the metric's own units. Raw g_tt and g_thth differ by about 180x on
this session, so a colour map of the raw matrix would show only the theta-theta corner.
cos alpha, printed under each panel, is the normalised off-diagonal and does not depend on c.

The diagonal is speed (top left) and direction-code length (bottom right); the off-diagonal is
the time-direction shear.

Usage:  python3 make_metric_matrix.py [--dirs 0 180] [--bins 5 12 24] [--session 20160914]
Output: figures/fig43_metric_matrix.pdf and .png
"""
from __future__ import annotations
import argparse

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import plotstyle as ps
import metric_convention as mc

FIELDS = "msa_fields.npz"
CMAP = "RdBu_r"


def pick_directions(theta, degs):
    return [int(np.argmin(np.abs(np.angle(np.exp(1j * (theta - np.deg2rad(d)))))))
            for d in degs]


def draw_cylinder(ax, theta, kidx, bins, T, cols):
    """The (t, theta) domain on its side: t left to right, theta round."""
    u = np.linspace(0, 2 * np.pi, 160)
    v = np.linspace(0, 1, 30)
    V, U = np.meshgrid(v, u)
    ax.plot_surface(V, np.cos(U), np.sin(U), color="#d9dee2", alpha=.30,
                    linewidth=0, antialiased=True, shade=False, zorder=1)
    for w in theta:                                        # the 8 measured directions
        ax.plot([0, 1], [np.cos(w)] * 2, [np.sin(w)] * 2, color="#c8ccd0", lw=.5, zorder=2)
    pts = {}
    for j, t in enumerate(bins):
        z = t / (T - 1)
        ax.plot(z * np.ones_like(u), np.cos(u), np.sin(u), color=cols[j], lw=1.7, zorder=4)
        # the bin label sits inside its own ring, in the plane of that ring
        ax.text(z, 0, 0, f"bin {t}", zdir="x", color=cols[j], fontsize=7.5,
                ha="center", va="center", zorder=6)
        for k in kidx:
            w = theta[k]
            ax.scatter([z], [np.cos(w)], [np.sin(w)], s=26, color=cols[j],
                       edgecolor="white", linewidth=.6, depthshade=False, zorder=7)
            pts[(k, t)] = (z, np.cos(w), np.sin(w))
    ax.set_box_aspect((2.05, 1, 1))
    ax.view_init(elev=13, azim=-64)
    ax.set_axis_off()
    return pts


def fig_frac(ax3d, fig, xyz):
    """3-D data point -> figure-fraction coordinates, for drawing callout arrows."""
    x2, y2, _ = proj3d.proj_transform(*xyz, ax3d.get_proj())
    disp = ax3d.transData.transform((x2, y2))
    return fig.transFigure.inverted().transform(disp)


def heat_panel(fig, rect, G, Gs, vmax, colour, label, cosa, below=False, fs=6.2):
    """One 2x2 heat map: colour from the rescaled matrix, numbers are the raw entries."""
    ax = fig.add_axes(rect)
    ax.imshow(Gs, cmap=CMAP, vmin=-vmax, vmax=vmax, interpolation="nearest")
    for i in range(2):
        for j in range(2):
            v = Gs[i, j]
            # 3 dp, fixed width: %g runs to 8 characters on a value like -0.00668 and
            # overflows the cell, which it did in PMd
            ax.text(j, i, f"{G[i, j]:.3f}", ha="center", va="center", fontsize=fs,
                    color="white" if abs(v) > .58 * vmax else ps.INK)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(colour); sp.set_linewidth(1.4)
    txt = rf"{label}   $\cos\alpha={cosa:+.2f}$"
    col = ps.INK
    if below:            # bottom row: the label goes UNDER the panel, so the incoming
        ax.text(.5, -.06, txt, transform=ax.transAxes, ha="center", va="top",
                fontsize=fs + .4, color=col)          # arrow has the top edge to itself
    else:
        ax.set_title(txt, fontsize=fs + .4, pad=2.5, color=col)
    return ax



def draw_block(fig, z, session, region, dirs, bins, band, fig_h, label_region):
    """One region: a cylinder with its call-out panels, inside the vertical band (y0, y1)."""
    y0, y1 = band
    bh = y1 - y0
    key = f"{session}|{region}|full"
    # The metric at the SETTLED CONVENTION (metric_convention: pcubic, lam = 0.3,
    # pushforward), recomputed from the cached xbar/theta/A/C. The stored "|g" and
    # "|g_push" fields are method="cubic", lam=0 and are not used here.
    theta = z[key + "|theta"]
    g = mc.metric(z[key + "|xbar"], theta, z[key + "|A"], z[key + "|C"])
    T = g.shape[1]
    D = int(z[key + "|A"].shape[0])

    c = float(np.sqrt(g[..., 1, 1]).mean() / np.sqrt(g[..., 0, 0]).mean())
    S = np.diag([c, 1.0])
    g_s = np.einsum("ab,ktbc,cd->ktad", S, g, S)

    kidx = pick_directions(theta, dirs)
    if len(kidx) > 2:
        raise SystemExit("--dirs takes at most two directions: the cylinder occupies the "
                         "middle band, so a third row of panels would sit on top of it.")
    cols = [plt.cm.viridis(t / (T - 1)) for t in bins]
    vmax = max(abs(g_s[k, t]).max() for k in kidx for t in bins)

    ax3 = fig.add_axes([-.07, y0 + .015 * bh, 1.14, .97 * bh], projection="3d")
    pts = draw_cylinder(ax3, theta, kidx, bins, T, cols)
    fig.canvas.draw()

    # panels are sized in INCHES and converted, so they do not grow when the canvas does
    W, H = .70 / ps.WIDTH, .50 / fig_h
    fs = 6.2 if bh > .8 else 5.3
    xs = np.linspace(.190, .655, len(bins))
    rows = [y0 + bh - H - .035 * bh, y0 + .045 * bh]
    for i, k in enumerate(kidx):
        for j, t in enumerate(bins):
            cosa = float(g[k, t, 0, 1] / np.sqrt(g[k, t, 0, 0] * g[k, t, 1, 1]))
            heat_panel(fig, [xs[j], rows[i], W, H], g[k, t], g_s[k, t], vmax, cols[j],
                       rf"$\theta={dirs[i]}^\circ$", cosa, below=(i == 1), fs=fs)
            x0, yq = fig_frac(ax3, fig, pts[(k, t)])
            yp = rows[i] if i == 0 else rows[i] + H
            fig.add_artist(FancyArrowPatch(
                (x0, yq), (xs[j] + W / 2, yp), transform=fig.transFigure,
                arrowstyle="-|>", mutation_scale=7, lw=.8, color=cols[j],
                shrinkA=3, shrinkB=4, alpha=.9, connectionstyle="arc3,rad=0.0"))

    cax = fig.add_axes([.945, y0 + .36 * bh, .011, .28 * bh])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(-vmax, vmax))
    cb = fig.colorbar(sm, cax=cax); cb.ax.tick_params(labelsize=6)
    cb.set_label("rescaled entry", fontsize=6)
    cb.ax.yaxis.set_label_position("left")

    if label_region:
        fig.text(.015, y0 + .92 * bh, rf"\textbf{{{region}}}, $D={D}$" if False
                 else f"{region},  D = {D}", fontsize=9, color=ps.INK, ha="left",
                 va="center", fontweight="bold")
    return c, D, [(dirs[i], [float(g[k, t, 0, 1] / np.sqrt(g[k, t, 0, 0] * g[k, t, 1, 1]))
                             for t in bins]) for i, k in enumerate(kidx)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="20160914")
    ap.add_argument("--regions", nargs="+", default=["M1", "PMd"],
                    help="one region for a single-block figure, two to stack them on one page")
    ap.add_argument("--dirs", type=int, nargs="+", default=[90, 270])
    ap.add_argument("--bins", type=int, nargs="+", default=[5, 12, 24])
    ap.add_argument("--fields", default=FIELDS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ps.apply()

    z = np.load(a.fields)
    n = len(a.regions)
    out = a.out or ("figures/fig43_metric_matrix" if n > 1
                    else f"figures/fig43_metric_matrix_{a.regions[0]}")
    # one block is 4.80 in; two are stacked into a single page-height figure
    fig_h = 4.80 if n == 1 else 8.70
    fig = plt.figure(figsize=(ps.WIDTH, fig_h))

    top = .955 if n == 1 else .965
    for r, region in enumerate(a.regions):
        band = (0.0, top) if n == 1 else (top * (n - 1 - r) / n, top * (n - r) / n)
        c, D, cosines = draw_block(fig, z, a.session, region, a.dirs, a.bins, band,
                                   fig_h, label_region=(n > 1))
        print(f"{region}: D={D}  c={c:.3f}", flush=True)
        for deg, row in cosines:
            print(f"  theta={deg:3d}deg cos alpha: " +
                  "  ".join(f"{v:+.3f}" for v in row), flush=True)

    ttl = (rf"The metric matrix $\mathbf{{G}}(t,\theta)$ on the cylinder  ---  {a.session}"
           if n > 1 else
           rf"The metric matrix $\mathbf{{G}}(t,\theta)$ on the cylinder  ---  {a.session} "
           rf"{a.regions[0]}, $D={D}$")
    fig.suptitle(ttl, y=.988)

    # a small key for the two coordinates, once, at the foot of the figure
    key = fig.add_axes([.035, .020, .080, .080 * 4.80 / fig_h]); key.set_axis_off()
    key.set_xlim(0, 1); key.set_ylim(0, 1)
    key.annotate("", (.92, .12), (.10, .12), arrowprops=dict(arrowstyle="-|>", lw=.9,
                 color=ps.MUTED, mutation_scale=7))
    key.annotate("", (.10, .94), (.10, .12), arrowprops=dict(arrowstyle="-|>", lw=.9,
                 color=ps.MUTED, mutation_scale=7))
    key.text(.95, .12, "$t$", fontsize=7.5, color=ps.MUTED, ha="left", va="center")
    key.text(.10, 1.0, r"$\theta$", fontsize=7.5, color=ps.MUTED, ha="center", va="bottom")

    ps.save(fig, out)


if __name__ == "__main__":
    main()
