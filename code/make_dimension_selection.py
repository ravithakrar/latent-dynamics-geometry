"""make_dimension_selection.py -- how the latent dimension D is chosen.

Three panels, all six sessions, all with the operating D marked:
  A  bi-cross-validated PCA: held-out variance explained against D.
  B  co-smoothing: R^2 on held-out NEURONS against D. This is the selector.
  C  the same R^2 with held-out TRIALS instead of held-out neurons. It rises and
     does not turn over, so it does not select D.

B and C are the same score computed the same way; only what is held out differs.

A reads bcvpca_sessions.json; B and C read cosmooth6b.jsonl.

Usage:  python3 make_dimension_selection.py
Output: figures/fig_dimension_selection.pdf/.png
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotstyle as ps

SESSIONS = ["20160909", "20160914", "20160919", "20160921", "20160929", "20161013"]
REGIONS = ("M1", "PMd")
OPERATING = {"M1": 8, "PMd": 12}
COSMOOTH6 = "cosmooth6b.jsonl"


def main():
    ps.apply()
    B = json.load(open("bcvpca_sessions.json"))
    rows6 = [json.loads(l) for l in open(COSMOOTH6)] if os.path.exists(COSMOOTH6) else []

    def knee(s, r):
        """Seed-mean argmax of the co-smoothing curve for one session and area."""
        rs = [q for q in rows6 if q["session"] == s and q["region"] == r]
        Ds = sorted({q["D"] for q in rs})
        m = [np.mean([q["r2"] for q in rs if q["D"] == d]) for d in Ds]
        return Ds[int(np.argmax(m))]
    fig, axs = plt.subplots(1, 3, figsize=(ps.WIDTH, 2.5))

    PANELS = [("A", None, "held-out variance explained", "bi-cross-validation"),
              ("B", "r2", "$R^2$, held-out neurons", "co-smoothing"),
              ("C", "trial_r2", "$R^2$, held-out trials", "holding out trials")]

    for ax, (tag, field, ylab, name) in zip(axs, PANELS):
        for r in REGIONS:
            curves, grid = [], np.arange(1, 31)
            for s_ in SESSIONS:
                if field is None:
                    e = B[f"{s_}_{r}_singletrial"]
                    d, v = np.array(e["dims"]), np.array(e["ve_te"])
                    m = d <= 30
                    d, v = d[m], v[m]
                else:
                    rs = [q for q in rows6 if q["session"] == s_ and q["region"] == r]
                    if not rs:
                        continue
                    d = np.array(sorted({q["D"] for q in rs}))
                    v = np.array([np.mean([q[field] for q in rs if q["D"] == dd])
                                  for dd in d])
                    grid = d
                ax.plot(d, v, "-", color=ps.COL[r], lw=.5, alpha=.35)
                curves.append(np.interp(grid, d, v))
            if curves:
                ax.plot(grid, np.mean(curves, 0), "-", color=ps.COL[r], lw=1.7,
                        label=r)
                ax.axvline(OPERATING[r], ls=":", color=ps.COL[r], lw=.9)
        ax.set(xlabel="latent dimension $D$", ylabel=ylab)
        if field is None:
            ax.set(xlim=(1, 30), ylim=(-.02, .30))
        elif field == "r2":
            ax.set_ylim(-.25, .30)          # the fall-off past the peak matters
        else:
            ax.set_ylim(0, .72)             # this one keeps climbing
        ax.set_title(f"{tag}   {name}", loc="left")
        if tag == "A":
            ax.legend(loc="lower left", fontsize=6.5, frameon=False,
                      handletextpad=.3)

    fig.tight_layout(pad=.7, w_pad=1.6)
    ps.save(fig, "figures/fig_dimension_selection")
    print("mode:", ps.mode())


if __name__ == "__main__":
    main()
