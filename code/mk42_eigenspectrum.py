"""mk42_eigenspectrum: eig(A) over EM iterations, session 20160914, both initialisations.

Same two runs as fig_init_floor: identity initialisation and the PCA warm start, D = 8 in M1
and 12 in PMd, 80 EM iterations. That figure follows the likelihood of those runs; this one
follows where the dynamics land.

Eigenvalues are drawn as lambda itself, on the ordinary Argand diagram: a mode decays when
|lambda| < 1, so the stability boundary is the unit circle centred on the origin, drawn dashed.
The fitted modes all sit just inside +1, so each panel carries a magnified inset of that patch.
The identity initialisation starts with every mode at lambda = 1, so its track begins at a
single point on the boundary and spreads inwards.

Source: code/eigtrack_20160914.npz (run_eigtrack.py). No fits are done here.

Run from code/:  python3 mk42_eigenspectrum.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotstyle as ps

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
CODE = os.path.join(HERE, "..", "..", "code")
SESS = "20160914"
REGIONS = ["M1", "PMd"]
INITS = [("identity", "identity initialisation"), ("warmstart", "warm start")]
TIME = plt.cm.viridis

z = np.load(os.path.join(CODE, f"eigtrack_{SESS}.npz"))

ps.apply()
fig, axs = plt.subplots(2, 2, figsize=(ps.WIDTH, 4.75), sharex=True, sharey=True)
FULL = ((-1.16, 1.16), (-1.16, 1.16))     # the whole stability circle
ZOOM = ((0.882, 1.022), (-0.152, 0.152))   # where the fitted modes actually are
th = np.linspace(0, 2 * np.pi, 1441)
panel = "ABCD"
OUT = os.environ.get("EIGOUT", "fig42_eigenspectrum")


def draw(ax, e, small=False):
    """The tracks on an Argand diagram. Stability boundary: the unit circle |lambda| = 1."""
    n = e.shape[0]
    ax.plot(np.cos(th), np.sin(th), color=ps.MUTED, lw=.9, ls="--", zorder=1)
    ax.axhline(0, color=ps.GRID, lw=.6, zorder=0)
    # the axes cross at the centre of the stability circle, the origin; in the magnified
    # inset the origin is off the edge, so there the vertical marks lambda = 1 instead
    ax.axvline(1 if small else 0, color=ps.GRID, lw=.6, zorder=0)
    for k in range(n):
        ax.scatter(e[k].real, e[k].imag, s=3.5 if small else 9,
                   color=TIME(k / (n - 1)), alpha=.85, lw=0, zorder=3)
    ax.scatter(e[-1].real, e[-1].imag, s=9 if small else 22, facecolor="none",
               edgecolor=ps.INK, lw=.45 if small else .6, zorder=4)


for r, reg in enumerate(REGIONS):
    D = int(z[f"{reg}|identity|meta"][2])
    for c, (init, name) in enumerate(INITS):
        ax = axs[r, c]
        e = z[f"{reg}|{init}|eigs"]                       # (nit+1, D)
        draw(ax, e)
        ax.set(xlim=FULL[0], ylim=FULL[1])
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6.5)
        ax.set_title(f"{panel[2 * r + c]}   {reg} $D={D}$, {name}", loc="left", fontsize=7.6)
        if r == 1:
            ax.set_xlabel(r"$\mathrm{Re}\,\lambda$")
        if c == 0:
            ax.set_ylabel(r"$\mathrm{Im}\,\lambda$")

        # the modes occupy a patch about a sixth of the circle across, so each panel
        # carries a magnified copy of that patch
        ins = ax.inset_axes([.655, .655, .335, .335])
        draw(ins, e, small=True)
        ins.set(xlim=ZOOM[0], ylim=ZOOM[1])
        ins.set_xticks([]); ins.set_yticks([])
        for sp in ins.spines.values():
            sp.set_linewidth(.5); sp.set_color(ps.MUTED)
        ax.add_patch(plt.Rectangle((ZOOM[0][0], ZOOM[1][0]),
                                   ZOOM[0][1] - ZOOM[0][0], ZOOM[1][1] - ZOOM[1][0],
                                   fill=False, ec=ps.MUTED, lw=.5, zorder=5))

        print(f"{reg:>3} {init:<9}: final max|lambda| {np.abs(e[-1]).max():.4f}, "
              f"{int((np.abs(e[-1].imag) > 1e-9).sum() // 2)} complex pairs")

sm = plt.cm.ScalarMappable(cmap=TIME, norm=plt.Normalize(0, z["M1|identity|eigs"].shape[0] - 1))
cb = fig.colorbar(sm, ax=axs, fraction=.030, pad=.03)
cb.set_label("EM iteration", fontsize=7.5)
cb.ax.tick_params(labelsize=6.5)
ps.save(fig, os.path.join(HERE, OUT))
