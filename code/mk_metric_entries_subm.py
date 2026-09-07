"""
fig_metric_entries_subm: the cross-animal panel. The three entries of the metric through the
reach for monkey M, drawn exactly as fig_metric_entries but from a second animal.

Seven sessions, 2014-02-03 to 2014-03-07, 33 days, matched to the 34-day span of the six
monkey-C sessions. D = 4 (M1) and 6 (PMd), the co-smoothing knee for THIS animal: sub-M yields
26 to 52 M1 units and 66 to 121 PMd units against 55 to 95 and 114 to 258 in sub-C, so fewer
latent dimensions are supported and the two animals are not compared at a matched D.

Reads subm_fields.npz (build_subm_fields.py), which stores xbar/theta/A/C per session, region
and variant, so both constructions are recomputed here without refitting.

Run from code/:  python3 mk_metric_entries_subm.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotstyle as ps
import metric_convention as mc  # pcubic, lam=0.3, pushforward

SESS = ["20140203", "20140217", "20140218", "20140303", "20140304", "20140306", "20140307"]
REGIONS = ["M1", "PMd"]
D_REGION = {"M1": 4, "PMd": 6}
HERO, ONSET, BIN_MS = "20140303", 5, 20.0
d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "subm_fields.npz"))


def fields(sess, reg, variant):
    """(direct, pushforward) at the settled convention: pcubic, lam = 0.3."""
    k = f"{sess}|{reg}|{variant}"
    xbar, theta = d[k + "|xbar"], d[k + "|theta"]
    A, C = d[k + "|A"], d[k + "|C"]
    return (mc.metric(xbar, theta, A, C, push=False),
            mc.metric(xbar, theta, A, C, push=True))


def entries(g):
    gtt, gqq, gtq = g[..., 0, 0], g[..., 1, 1], g[..., 0, 1]
    cos = gtq / np.sqrt(np.maximum(gtt * gqq, 1e-30))
    return (np.sqrt(gtt).mean(0), np.sqrt(gqq).mean(0), cos.mean(0))


import json as _json
import collections as _collections

#: Error band: the 95% interval over the B = 100 trial-bootstrap refits of
#: resample43_subm.jsonl, 1.96 sd of the session-mean curve across replicates.
#: --halfsplit uses the single disjoint half-split SE instead.
BOOT = os.environ.get("RESAMPLE_JSONL", "resample43_subm.jsonl")


def boot_band(reg, sessions):
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
    push = np.array([entries(fields(s, reg, "full")[1]) for s in SESS])
    dire = np.array([entries(fields(s, reg, "full")[0]) for s in SESS])
    if "--halfsplit" in sys.argv:
        h1 = np.array([entries(fields(s, reg, "h1")[1]) for s in SESS])
        h2 = np.array([entries(fields(s, reg, "h2")[1]) for s in SESS])
        return push, dire, np.std(h1 - h2, axis=0) / 2.0
    return push, dire, boot_band(reg, SESS)


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
        ax.plot(t, dire[:, j].mean(0), color=DIRECT, lw=1.2, ls="--", zorder=4, label="direct")
        ax.axvline(0, color="0.75", lw=0.7, zorder=0)
        if j == 2:
            ax.axhline(0, color="0.75", lw=0.7, zorder=0)
        ax.set_title(f"{panel[3 * r + j]}   {titles[j]}", loc="left", fontsize=7.6)
        ax.set_ylabel(ylabs[j], fontsize=7.4)
        if r == len(REGIONS) - 1:
            ax.set_xlabel("time from movement onset (ms)", fontsize=7.4)
    axs[r, 0].text(-.34, .5, f"{reg}\n$D={D_REGION[reg]}$", transform=axs[r, 0].transAxes,
                   rotation=90, ha="center", va="center", fontsize=8.5, fontweight="bold")
    print(f"--- sub-M {reg} (D={D_REGION[reg]}) ---")
    for j, lab in enumerate(["sqrt(g_tt)", "sqrt(g_qq)", "cos alpha"]):
        m = push[:, j].mean(0)
        print(f"{lab:>12}: onset {m[ONSET]:+.3f}  end {m[-1]:+.3f}  "
              f"ratio {m[-1] / m[ONSET]:+.3f}  median SE {np.median(se[j]):.4f}")

axs[0, 0].legend(fontsize=6.4, loc="best")
fig.tight_layout(pad=.5)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
ps.save(fig, args[0] if args else "figures/fig43_metric_entries_subm")
