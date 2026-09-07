"""plot_recovery_v2.py — the Methods recovery figure, all three components of g.

Panels A-C take the metric apart: the two diagonal terms as
functions of time, and the off-diagonal as a pointwise field. Panel D is the fit-D sweep.

Panel C is a scatter, not a curve. cos alpha averages to roughly zero over directions, so a
mean-vs-time panel would show two flat lines near zero. The pointwise field spans about +-0.8,
and every (theta, t) point is plotted against its true value.

No between-session reference line is drawn.

Inputs: components_<sess>_<region>.npz (from components.py), metric_recovery.json (the 10-seed
exemplar sweep) or metric_recovery_multi_M1.json.
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib.pyplot as plt
import plotstyle as ps
from components import comps

SESS, REGION = "20160914", "M1"
DS = [4, 6, 8, 12, 20]
DSTAR = 8
ONSET = 5


def band(ax, t, hat, color):
    """mean +- sd across seeds."""
    m, s = hat.mean(0), hat.std(0, ddof=1)
    ax.plot(t, m, color=color, lw=1.8, ls="--", label="recovered")
    ax.fill_between(t, m - s, m + s, color=color, alpha=.18, lw=0)


def main():
    ps.apply()
    z = np.load(f"components_{SESS}_{REGION}.npz")
    gt, G = z["g_true"], z["g_hat"]
    tt, th, ca = comps(gt)
    H = [comps(g) for g in G]
    t = np.arange(tt.shape[1])

    fig, axs = plt.subplots(2, 2, figsize=(ps.WIDTH, 5.0))

    # A: neural speed
    ax = axs[0, 0]
    ax.plot(t, tt.mean(0), color=ps.INK, lw=2.2, label="true")
    band(ax, t, np.stack([h[0].mean(0) for h in H]), ps.M1)
    ax.axvline(ONSET, color=ps.MUTED, ls=":", lw=.9)
    ax.set(title=r"\textbf{A}  Neural speed $\sqrt{\mathbf{G}_{tt}}$" if False else
           r"A   Neural speed $\sqrt{\mathbf{G}_{tt}}$",
           xlabel="time bin (20 ms)", ylabel=r"$\sqrt{\mathbf{G}_{tt}}$")
    ax.grid(alpha=.7); ax.set_axisbelow(True); ax.legend(fontsize=7)

    # B: direction-code length
    ax = axs[0, 1]
    ax.plot(t, th.mean(0), color=ps.INK, lw=2.2, label="true")
    band(ax, t, np.stack([h[1].mean(0) for h in H]), ps.M1)
    ax.axvline(ONSET, color=ps.MUTED, ls=":", lw=.9)
    ax.set(title=r"B   Direction-code length $\sqrt{\mathbf{G}_{\theta\theta}}$",
           xlabel="time bin (20 ms)", ylabel=r"$\sqrt{\mathbf{G}_{\theta\theta}}$")
    ax.grid(alpha=.7); ax.set_axisbelow(True); ax.legend(fontsize=7)

    # C: alignment, pointwise
    ax = axs[1, 0]
    hc = np.stack([h[2] for h in H]).mean(0)
    lim = (min(ca.min(), hc.min()) - .05, max(ca.max(), hc.max()) + .05)
    ax.plot(lim, lim, color=ps.MUTED, lw=1.0, ls=":", zorder=1)
    sc = ax.scatter(ca.ravel(), hc.ravel(), c=np.tile(t, (ca.shape[0], 1)).ravel(),
                    cmap="viridis", s=7, lw=0, alpha=.85, zorder=2)
    r = np.corrcoef(ca.ravel(), hc.ravel())[0, 1]
    ax.text(.04, .93, f"$r = {r:.3f}$", transform=ax.transAxes, fontsize=8, va="top")
    ax.text(.04, .84, f"mean $= {ca.mean():+.2f}$", transform=ax.transAxes,
            fontsize=6.5, va="top", color=ps.MUTED)
    cb = fig.colorbar(sc, ax=ax, pad=.02, fraction=.046)
    cb.set_label("time bin", fontsize=7); cb.ax.tick_params(labelsize=6)
    ax.set(title=r"C   Alignment $\cos\alpha$, every $(\theta, t)$",
           xlabel=r"true $\cos\alpha$", ylabel=r"recovered $\cos\alpha$",
           xlim=lim, ylim=lim)
    ax.grid(alpha=.7); ax.set_axisbelow(True)

    # D: the sweep
    ax = axs[1, 1]
    try:
        S = json.load(open("metric_recovery.json"))["per_fitD"]
        mu = [S[str(d)]["d_msa_mean"] for d in DS]
        sd = [S[str(d)]["d_msa_sd"] for d in DS]
        st = [S[str(d)]["same_truth_mean"] for d in DS]
        stsd = [S[str(d)]["same_truth_sd"] for d in DS]
    except FileNotFoundError:
        R = [r for r in json.load(open("metric_recovery_multi_M1.json"))["results"]
             if r["session"] == SESS][0]["per_fitD"]
        mu = [R[str(d)]["d_msa_mean"] for d in DS]
        sd = [R[str(d)]["d_msa_sd"] for d in DS]
        st = [R[str(d)]["same_truth_mean"] for d in DS]
        stsd = [R[str(d)]["same_truth_sd"] for d in DS]
    x = np.arange(len(DS))
    ax.errorbar(x, mu, yerr=sd, fmt="o-", color=ps.M1, lw=1.8, capsize=3, label="vs truth")
    ax.errorbar(x, st, yerr=stsd, fmt="^--", color="#0f766e", lw=1.6, capsize=3,
                label="same truth, new data")
    ax.axvline(DS.index(DSTAR), color=ps.GRID, lw=6, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels([str(d) for d in DS])
    ax.set(title=r"D   Recovery error vs fitted $D$",
           xlabel="latent dimension used for the fit", ylabel="$d_{MSA}$")
    ax.grid(alpha=.7); ax.set_axisbelow(True); ax.legend(fontsize=6.5, loc="upper left")

    fig.tight_layout(pad=.6)
    ps.save(fig, "figures/fig_metric_recovery")
    print("wrote figures/fig_metric_recovery.{pdf,png}")


if __name__ == "__main__":
    main()
