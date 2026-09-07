"""mk43_geodesic.py — the two ring figures for the metric-field section, at the convention.

fig43_circumference   L(t) = closed integral of sqrt(g_thth) dtheta, six sessions of monkey C,
                      both regions, session mean heavy.
fig43_mds             the ring of the eight measured directions embedded by classical MDS on
                      its own geodesic distances, session 20160914, one octagon per time bin.

Source: ring_fields_push.npz (build_ring_fields.py), which is method="pcubic", lam = 0.3,
PUSHFORWARD -- the settled convention.

Run from code/:  python3 mk43_geodesic.py
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import geodesic_ring as gr
import plotstyle as ps

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
CODE = os.path.join(HERE, "..", "..", "code")

_ap = argparse.ArgumentParser()
_ap.add_argument("--animal", choices=["subC", "subM"], default="subC")
_a = _ap.parse_args()
STEM = "ring_fields_push" if _a.animal == "subC" else "ring_fields_push_subm"
SUF = "" if _a.animal == "subC" else "_subm"

Z = np.load(os.path.join(CODE, STEM + ".npz"))
J = json.load(open(os.path.join(CODE, STEM + ".json")))
SESS, FOCUS = J["sessions"], J["focus"]
ONSET, BIN_MS = J["onset_bin"], J["bin_ms"]
TIME = plt.cm.viridis

ps.apply()

# circumference L(t)
fig, ax = plt.subplots(figsize=(ps.WIDTH, 2.9))
for reg in ("M1", "PMd"):
    curves = np.array([Z[f"L_{s}_{reg}"] for s in SESS])
    t = (np.arange(curves.shape[1]) - ONSET) * BIN_MS
    for c in curves:
        ax.plot(t, c, color=ps.COL[reg], lw=.8, alpha=.32, zorder=1)
    m = curves.mean(0)
    ax.plot(t, m, color=ps.COL[reg], lw=2.4, zorder=3, label=f"{reg} (mean of {len(curves)})")
    print(f"{reg}: onset {m[ONSET]:.2f}  end {m[-1]:.2f}  ratio {m[-1] / m[ONSET]:.3f}  "
          f"start {m[0]:.2f}")
ax.axvline(0, color=ps.MUTED, ls="--", lw=.9, zorder=0)
ax.text(BIN_MS * .4, ax.get_ylim()[1] * .99, "movement onset", fontsize=7.5,
        color=ps.MUTED, va="top")
ax.set(xlabel="time from movement onset (ms)",
       ylabel=r"$L(t)=\oint\sqrt{\mathbf{G}_{\theta\theta}}\,\mathrm{d}\theta$")
ax.grid(alpha=.6); ax.set_axisbelow(True); ax.legend(fontsize=7.5)
fig.tight_layout(pad=.4)
ps.save(fig, os.path.join(HERE, "fig43_circumference" + SUF))

# geodesic MDS
fig, axs = plt.subplots(1, 2, figsize=(ps.WIDTH, 3.15))
diag = {}
for ax, reg in zip(axs, ("M1", "PMd")):
    gm = Z[f"geomat_{FOCUS}_{reg}"]                     # (T,K,K)
    th = Z[f"theta_{FOCUS}_{reg}"]
    T = gm.shape[0]
    for t in range(0, T, 4):
        r = gr.embed_ring(gm[t], th)
        e = np.vstack([r["xy"], r["xy"][:1]])
        ax.plot(e[:, 0], e[:, 1], color=TIME(t / (T - 1)), lw=1.5, alpha=.9)
        ax.scatter(e[:-1, 0], e[:-1, 1], s=12, color=TIME(t / (T - 1)), zorder=3)
    allb = [gr.embed_ring(gm[t], th) for t in range(T)]
    diag[reg] = dict(stress_mean=float(np.mean([b["stress"] for b in allb])),
                     stress_max=float(np.max([b["stress"] for b in allb])),
                     resultant_min=float(np.min([b["resultant"] for b in allb])),
                     floor=float(allb[0]["floor"]))
    ax.set_aspect("equal")
    ax.axhline(0, color=ps.GRID, lw=.7); ax.axvline(0, color=ps.GRID, lw=.7)
    ax.set(title=f"{reg}   (stress {diag[reg]['stress_mean']:.3f}, "
                 f"floor {diag[reg]['floor']:.3f})",
           xlabel="geodesic MDS 1", ylabel="geodesic MDS 2")
sm = plt.cm.ScalarMappable(cmap=TIME, norm=plt.Normalize(-ONSET * BIN_MS,
                                                         (24 - ONSET) * BIN_MS))
cb = fig.colorbar(sm, ax=axs, fraction=.030, pad=.04)
cb.set_label("time from movement onset (ms)", fontsize=7.5)
cb.ax.tick_params(labelsize=6.5)
ps.save(fig, os.path.join(HERE, "fig43_mds" + SUF))
json.dump(diag, open(os.path.join(HERE, "geodesic_mds_push_diagnostics%s.json" % SUF), "w"), indent=1)
print("MDS:", json.dumps(diag, indent=1))
