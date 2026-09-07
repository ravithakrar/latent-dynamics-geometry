"""
fig_metric_entries: the three entries of the pullback metric through the reach,
M1 and PMd, each at its operating D, under the current convention.

Everything is read from msa_fields.npz, which already stores, per session:
  full|g       direct construction  (spline refitted at every time bin)
  full|g_push  pushforward          (tangent estimated once at t=0, carried by A^t)
  h1|*, h2|*   the two disjoint trial halves, used for the error bar

Error band: the 95% interval over the B = 100 trial-bootstrap
refits of resample43.jsonl, taken as 1.96 sd of the session-mean curve across replicates,
elementwise in time. It replaces the old SE_full = 1/2 sd(a_A - a_B) from a single disjoint
trial half-split, which had one degree of freedom per session and disagreed with the interval
drawn in the resampling figure. Pass --halfsplit to get the old band back for comparison.

Run from code/ so that msa_fields.npz is on the path, then move the output into
dissertation/figures/:
    python3 mk_metric_entries.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotstyle as ps          # shared thesis style; replaces a hard-coded styles path
                                # that pointed inside one machine's .venv and did not port
import metric_convention as mc  # pcubic, lam=0.3, pushforward -- the settled convention

SESS = ["20160909", "20160914", "20160919", "20160921", "20160929", "20161013"]
REGIONS = ["M1", "PMd"]
HERO, ONSET, BIN_MS = "20160914", 5, 20.0
d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "msa_fields.npz"), allow_pickle=True)


def entries(g):
    """(K,T,2,2) -> the three entries, averaged over the K directions."""
    gtt, gqq, gtq = g[..., 0, 0], g[..., 1, 1], g[..., 0, 1]
    cos = gtq / np.sqrt(np.maximum(gtt * gqq, 1e-30))
    return (np.sqrt(gtt).mean(0), np.sqrt(gqq).mean(0), cos.mean(0))


def g_of(sess, reg, variant, push):
    """The metric at the settled convention, recomputed from the cached xbar/theta/A/C.

    The stored `g` and `g_push` fields are method="cubic", lam=0, which is NOT the
    convention; everything here goes through metric_convention instead.
    """
    k = f"{sess}|{reg}|{variant}"
    return mc.metric(d[k + "|xbar"], d[k + "|theta"], d[k + "|A"], d[k + "|C"], push=push)


import json as _json
import collections as _collections

BOOT = os.environ.get("RESAMPLE_JSONL", "resample43.jsonl")


def boot_band(reg, sessions):
    """(3,T) half-width of the 95% trial-bootstrap interval on the SESSION-MEAN curve.

    One replicate is one refit of every session; the statistic banded is the mean over
    sessions, which is the heavy line in the figure, so the band belongs to that line and
    not to any one session.
    """
    per = _collections.defaultdict(dict)
    for line in open(BOOT):
        r = _json.loads(line)
        if r["mode"] == "trials" and r["region"] == reg:
            per[r["rep"]][r["sess"]] = r
    reps = [b for b in sorted(per) if all(s in per[b] for s in sessions)]
    curves = np.array([[[per[b][s]["s_tt"], per[b][s]["s_qq"], per[b][s]["cos"]]
                        for s in sessions] for b in reps])        # (B,S,3,T)
    return 1.96 * curves.mean(axis=1).std(axis=0, ddof=1)          # (3,T)


def region_arrays(reg):
    push = np.array([entries(g_of(s, reg, "full", True)) for s in SESS])    # (S,3,T)
    dire = np.array([entries(g_of(s, reg, "full", False)) for s in SESS])
    if "--halfsplit" in sys.argv:
        h1 = np.array([entries(g_of(s, reg, "h1", True)) for s in SESS])
        h2 = np.array([entries(g_of(s, reg, "h2", True)) for s in SESS])
        return push, dire, np.std(h1 - h2, axis=0) / 2.0                    # (3,T)
    return push, dire, boot_band(reg, SESS)                                 # (3,T)


ps.apply()
CY = plt.rcParams["axes.prop_cycle"].by_key()["color"]
PUSH, DIRECT = CY[0], CY[3]
titles = [r"neural speed $\sqrt{\mathbf{G}_{tt}}$",
          r"direction-code length $\sqrt{\mathbf{G}_{\theta\theta}}$",
          r"alignment $\cos\alpha$"]
ylabs = ["per time bin", "per radian", "dimensionless"]
panel = "ABCDEF"
hero = SESS.index(HERO)

fig, axs = plt.subplots(len(REGIONS), 3, figsize=(ps.WIDTH, 4.25), sharex=True)
for r, reg in enumerate(REGIONS):
    push, dire, se = region_arrays(reg)
    T = push.shape[-1]
    t = (np.arange(T) - ONSET) * BIN_MS
    for j in range(3):
        ax = axs[r, j]
        for i in range(len(SESS)):
            ax.plot(t, push[i, j], color=PUSH, lw=0.6, alpha=0.28, zorder=1)
        ax.plot(t, push[hero, j], color=PUSH, lw=0.9, alpha=0.75, zorder=2, label=HERO)
        m = push[:, j].mean(0)
        ax.fill_between(t, m - se[j], m + se[j], color=PUSH, alpha=0.25, lw=0, zorder=3)
        ax.plot(t, m, color=PUSH, lw=1.5, zorder=4, label="pushforward")
        ax.plot(t, dire[:, j].mean(0), color=DIRECT, lw=1.2, ls="--", zorder=4,
                label="direct")
        ax.axvline(0, color="0.75", lw=0.7, zorder=0)
        if j == 2:
            ax.axhline(0, color="0.75", lw=0.7, zorder=0)
        ax.set_title(f"{panel[3 * r + j]}   {titles[j]}", loc="left", fontsize=7.6)
        ax.set_ylabel(ylabs[j], fontsize=7.4)
        if r == len(REGIONS) - 1:
            ax.set_xlabel("time from movement onset (ms)", fontsize=7.4)
    axs[r, 0].text(-.34, .5, reg, transform=axs[r, 0].transAxes, rotation=90,
                   ha="center", va="center", fontsize=9, fontweight="bold")
    print(f"--- {reg} ---")
    for j, lab in enumerate(["sqrt(g_tt)", "sqrt(g_qq)", "cos alpha"]):
        m = push[:, j].mean(0)
        print(f"{lab:>12}: onset {m[ONSET]:+.3f}  end {m[-1]:+.3f}  "
              f"ratio {m[-1] / m[ONSET]:+.3f}  median half-width {np.median(se[j]):.4f}")

axs[0, 0].legend(fontsize=6.4, loc="best")
fig.tight_layout(pad=.5)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
ps.save(fig, args[0] if args else "figures/fig43_metric_entries")
