"""mk_L_constructions.py — L(t) under both tangent constructions, with and without smoothing.

The chapter figure (fig43_circumference) is one cell of this one: pushforward at lam = 0.3.
Here all four combinations are drawn as session means, so the choice of convention can be read
off rather than argued.

  panel        construction:  pushforward (tangent taken once at t = 0, carried by A^t)
                              direct      (spline refitted at every time bin)
  colour       region
  line style   solid lam = 0.3 (the settled value), dashed lam = 0 (interpolating spline)

Six sessions of monkey C, means only; the per-session spread is in the chapter figure.

Writes dissertation/figures/fig43_L_constructions.{pdf,png}. Run from code/.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import geodesic_ring as gr
import metric_convention as mc
import plotstyle as ps

SESS = ["20160909", "20160914", "20160919", "20160921", "20160929", "20161013"]
REGIONS = ["M1", "PMd"]
ONSET, BIN_MS = 5, 20.0
OUT = "figures/fig43_L_constructions"
d = np.load("msa_fields.npz", allow_pickle=True)


def mean_L(reg, lam, push):
    mc.LAM = lam                                   # mc.ring reads the module global
    rows = []
    for s in SESS:
        k = f"{s}|{reg}|full"
        thf, radius = mc.ring(d[k + "|xbar"], d[k + "|theta"], d[k + "|A"], d[k + "|C"],
                              push=push)
        rows.append([gr.arc_length(radius[:, t], thf)[0] for t in range(radius.shape[1])])
    mc.LAM = 0.3
    return np.mean(rows, axis=0)


ps.apply()
fig, axs = plt.subplots(1, 2, figsize=(ps.WIDTH, 2.95), sharey=True)
for ax, (push, name) in zip(axs, [(True, "pushforward"), (False, "direct")]):
    for reg in REGIONS:
        for lam, ls, lw in [(0.3, "-", 2.2), (0.0, "--", 1.2)]:
            m = mean_L(reg, lam, push)
            t = (np.arange(len(m)) - ONSET) * BIN_MS
            ax.plot(t, m, color=ps.COL[reg], ls=ls, lw=lw,
                    label=rf"{reg}, $\lambda={lam}$")
            print(f"{name:>11} {reg:>3} lam={lam}: onset {m[ONSET]:6.2f}  "
                  f"peak {m.max():6.2f} at bin {int(m.argmax()):2d}  end {m[-1]:6.2f}  "
                  f"ratio {m[-1] / m[ONSET]:.3f}", flush=True)
    ax.axvline(0, color=ps.MUTED, ls="--", lw=.8, zorder=0)
    ax.set(xlabel="time from movement onset (ms)", title=name)
    ax.grid(alpha=.6); ax.set_axisbelow(True)
axs[0].set_ylabel(r"$L(t)=\oint\sqrt{\mathbf{G}_{\theta\theta}}\,\mathrm{d}\theta$")
axs[0].legend(fontsize=6.6, ncol=2, loc="lower left")
fig.tight_layout(pad=.4)
ps.save(fig, OUT)
