"""mk44_msa: d_MSA across sessions, monkey C, both regions.

Two figures, one per null, with the same panels in each.

  --null units   each fit uses one of two disjoint halves of the units; the dotted line carries
                 the trial-split floor for scale
  --null trials  both fits use the same units and differ only in which trials they saw

  Row A   d_MSA against days between sessions, one point per session pair
  Row B   the same values as densities over the 91 pairs

Reads the run_msa_dist_allpairs.py summaries (50 split seeds, 2,800 fits). Nothing is refitted.

Run from code/:  python3 mk44_msa.py
"""
import argparse
import json
import os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotstyle as ps

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
CODE = os.path.join(HERE, "..", "..", "code")
CONV = "pcubic_lam0.3_push"
NULLC = "0.45"
SUBJECTS = {
    "subC": [("M1", "msa_allpairs_{}_summary.json", "msa_size_M1.json"),
             ("PMd", "msa_allpairs_PMd_{}_summary.json", "msa_size_PMd.json")],
    "subM": [("M1", "msa_allpairs_subM_M1_{}_summary.json", "msa_size_subM_M1.json"),
             ("PMd", "msa_allpairs_subM_PMd_{}_summary.json", "msa_size_subM_PMd.json")],
}


#: number of distinct sessions behind the pairs, set by load()
N_SESS = 0


def load(fmt, mode):
    global N_SESS
    d = json.load(open(os.path.join(CODE, fmt.format(mode))))
    p = d["pairs"][CONV]
    N_SESS = len({q["a"] for q in p} | {q["b"] for q in p})
    return (d["D"], np.array([q["days"] for q in p], float),
            np.array([q["between_mean"] for q in p]),
            np.array([q["within_mean"] for q in p]),
            np.array([q["p"] for q in p]))


def load_size(f, mode):
    p = json.load(open(os.path.join(CODE, f)))[mode]["per_unit"]["pairs"]
    return (np.array([q["days"] for q in p], float),
            np.array([q["between_mean"] for q in p]),
            np.array([q["within_mean"] for q in p]))


def ols(x, y):
    A = np.vstack([np.ones_like(x), x]).T
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    r = y - A @ b
    return b[1], b[0], np.sqrt(r @ r / (len(x) - 2) * np.linalg.inv(A.T @ A)[1, 1])


def floor(ax, days, v):
    """the trial-split floor as a dotted mean and a 1 sd band, for scale."""
    g = np.linspace(0, days.max() * 1.04, 40)
    ax.plot(g, np.full_like(g, v.mean()), color=NULLC, lw=1.0, ls=":")
    ax.fill_between(g, v.mean() - v.std(), v.mean() + v.std(), color=NULLC, alpha=.18, lw=0)


def stars(pv):
    return "***" if pv < .001 else "**" if pv < .01 else "*" if pv < .05 else "n.s."


def compare(ax, win, btw, col):
    """Mean +- s.d. above each distribution, with the paired test between.

    The two samples are paired, one within-session null per between-session value, so the test
    is a two-sided Wilcoxon SIGNED-RANK rather than the rank-sum used on the raw per-pair samples.
    """
    pv = float(stats.wilcoxon(btw, win).pvalue)
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.42)
    for v, cc, y in ((win, NULLC, top * 1.10), (btw, col, top * 1.22)):
        ax.errorbar(v.mean(), y, xerr=v.std(ddof=1), color=cc, marker="o", ms=3,
                    lw=1.0, capsize=1.8)
    # the significance label goes in the corner with the sample size rather than over the
    # markers: under the split-units null the two means are ~1% of the axis apart, so a
    # bracket or a centred star lands on top of them
    return pv


def band(ax, days, btw, win, col, filled):
    """One null's pair of clouds: between in colour, within in grey.

    """
    grid = np.linspace(0, days.max() * 1.04, 40)
    for v, cc in ((win, NULLC), (btw, col)):
        kw = dict(s=8, lw=0, color=cc, alpha=.75) if filled else \
             dict(s=9, lw=.5, facecolor="none", edgecolor=cc, alpha=.6)
        ax.scatter(days, v, **kw)
        s, i, _ = ols(days, v)
        ax.plot(grid, i + s * grid, color=cc, lw=1.4 if filled else 1.0,
                ls="-" if filled else "--")


_ap = argparse.ArgumentParser()
_ap.add_argument("--null", choices=["units", "trials"], default="units")
_ap.add_argument("--subject", choices=["subC", "subM"], default="subC")
_a = _ap.parse_args()
NULL, SUBJECT = _a.null, _a.subject
REGIONS = SUBJECTS[SUBJECT]
SUF = ("" if NULL == "units" else "_trials") + ("" if SUBJECT == "subC" else "_subm")

ps.apply()
fig, axs = plt.subplots(2, 2, figsize=(ps.WIDTH, 4.7))
out = {}

for c, (reg, fmt, sizef) in enumerate(REGIONS):
    D, days, btw, win, pv = load(fmt, NULL)
    _, _, btw_t, win_t, _ = load(fmt, "trials")     # the trial-split floor, for scale
    col = ps.COL[reg]

    ax = axs[0, c]
    if NULL == "units":
        floor(ax, days, win_t)
    band(ax, days, btw, win, col, filled=True)
    ax.set(xlabel="days between sessions")
    ax.set_title(f"{'ab'[c]}   {reg}, $D={D}$, split {NULL}", loc="left", fontsize=7.6)
    if c == 0:
        ax.set_ylabel(r"$d_\mathrm{MSA}$")
    ax.text(.985, .03, f"{N_SESS} sessions, {len(days)} pairs", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.2, color=ps.MUTED)
    ax.grid(alpha=.5); ax.set_axisbelow(True)

    ax = axs[1, c]
    lo = min(win.min(), btw.min()) - .015
    hi = max(win.max(), btw.max()) + .015
    bins = np.linspace(lo, hi, 26)
    for v, cc in ((win, NULLC), (btw, col)):
        ax.hist(v, bins=bins, density=True, color=cc, alpha=.55)
        # MEDIAN. The marker above the histogram is the mean, so the two separate on a
        # skewed distribution; the caption says which is which.
        ax.axvline(np.median(v), color=cc, lw=1.2)
    pv_pair = compare(ax, win, btw, col)
    ax.set(xlabel=r"$d_\mathrm{MSA}$")
    ax.set_title(f"{'cd'[c]}   {reg}", loc="left", fontsize=7.6)
    if c == 0:
        ax.set_ylabel("density")
    ax.text(.985, .955, f"$n={len(btw)}$ pairs each", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.2, color=ps.MUTED)
    ax.text(.90, .865, f"({stars(pv_pair)})", transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color=ps.INK)

    sl, ic, se = ols(days, btw - win)
    out[reg] = dict(
        D=D, n=len(days),
        shape=dict(between=float(btw.mean()), null=float(win.mean()),
                   between_trials=float(btw_t.mean()), floor=float(win_t.mean()),
                   excess=float(np.mean(btw - win)), slope10=float(10 * sl),
                   se10=float(10 * se), cross=float(-ic / sl),
                   ns=int((pv > .05).sum())))
    print(f"{reg} D={D}\n  shape split trials: between {btw_t.mean():.3f} within "
          f"{win_t.mean():.3f} | split units: between {btw.mean():.3f} within {win.mean():.3f}"
          f"  slope {10 * sl:+.4f}+-{10 * se:.4f}/10d  cross {-ic / sl:.1f}d  "
          f"{int((pv > .05).sum())}/{len(pv)} n.s.  signed-rank p={pv_pair:.3g}")

h = [plt.Line2D([], [], marker="o", ls="", color=ps.M1, ms=3.6, label="between sessions"),
     plt.Line2D([], [], marker="o", ls="", color=NULLC, ms=3.6, label="within one session")]
if NULL == "units":
    h.append(plt.Line2D([], [], ls=":", color=NULLC, label="split trials, same session"))
fig.legend(handles=h, loc="upper center", ncol=3, fontsize=6.6, frameon=False,
           bbox_to_anchor=(.5, 1.005), handletextpad=.35, columnspacing=1.6)
fig.tight_layout(pad=.5, rect=(0, 0, 1, .975))
ps.save(fig, os.path.join(HERE, "fig44_msa" + SUF))
json.dump(out, open(os.path.join(HERE, f"msa_section44{SUF}.json"), "w"), indent=1)
