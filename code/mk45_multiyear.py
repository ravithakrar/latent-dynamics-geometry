"""mk45_multiyear: M1 across three years, monkey C.

1,326 pairs over 1,114 days, from every centre-out sub-C recording on disk: 9 from 2013, 29
from 2015 and 14 from 2016. None of the earlier sessions has PMd, so this is M1 alone.

Every session is cut to 41 units and 152 trials before the half-split, redrawn per seed, so
that array size does not vary with calendar distance. That cut raises every d_MSA, so the level
here is not comparable with Section 4.4.

Two figures, two panels each:

  fig45_levels   a  d_MSA against days apart on a log axis, split units, with the split-trials
                    null underneath as a dotted line and least squares fits in log10 days
                 b  the same values as densities, between-session side split at one year

  fig45_drift    a  the excess over the null by gap band, both nulls, on twin axes
                 b  the rate, as a slope in log10 days with a delete-one-session jackknife
                    error over the 52 recordings

Gaps run from 1 day to 1,114 days on a logarithmic axis, and rates are quoted per decade of
days.

Run from code/:  python3 mk45_multiyear.py
"""
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
BANDS = [(0, 4, "1-4 d"), (5, 9, "5-9 d"), (10, 15, "10-15 d"), (16, 22, "16-22 d"),
         (23, 30, "23-30 d"), (31, 42, "31-42 d"), (43, 120, "1.5-4 mo"), (121, 270, "4-9 mo"),
         (271, 450, "9-15 mo"), (451, 750, "15-25 mo"), (751, 1200, "25-37 mo")]


def load(mode):
    d = json.load(open(os.path.join(CODE, f"msa_allpairs_M1my_{mode}_summary.json")))
    p = d["pairs"][CONV]
    return dict(days=np.array([q["days"] for q in p], float),
                btw=np.array([q["between_mean"] for q in p]),
                win=np.array([q["within_mean"] for q in p]),
                a=np.array([q["a"] for q in p]), b=np.array([q["b"] for q in p]),
                n_sess=len({q["a"] for q in p} | {q["b"] for q in p}), D=d["D"])


def slope(x, y):
    A = np.vstack([np.ones_like(x), x]).T
    return np.linalg.lstsq(A, y, rcond=None)[0]


def jackknife(x, y, a, b):
    """delete-one-SESSION, because the 1,326 pairs come from 52 recordings and each recording
    sits in 51 of them. The recording is the unit that repeats, not the pair."""
    S = sorted(set(a) | set(b))
    js = np.array([slope(x[k], y[k])[1] for s in S for k in [(a != s) & (b != s)]])
    return slope(x, y)[1], np.sqrt(len(S) - 1) * js.std(ddof=0)


def stars(pv):
    return "***" if pv < .001 else "**" if pv < .01 else "*" if pv < .05 else "n.s."


U, T = load("units"), load("trials")
col = ps.COL["M1"]
ps.apply()

# figure 1: levels
def levels(d, ref, mode, out):
    """The two-panel levels figure for ONE null, so both nulls come from one run.

    d     the null being plotted (U or T)
    ref   the other null, drawn behind panel a as a dotted mean +- s.d. band for
          scale, or None to omit it
    mode  panel-a title suffix, e.g. "split units"
    out   basename written under HERE
    """
    fig, axs = plt.subplots(1, 2, figsize=(ps.WIDTH, 3.0))

    ax = axs[0]
    days, btw, win = d["days"], d["btw"], d["win"]
    if ref is not None:
        g = np.logspace(np.log10(.8), np.log10(1400), 40)
        ax.plot(g, np.full_like(g, ref["win"].mean()), color=NULLC, lw=1.0, ls=":")
        ax.fill_between(g, ref["win"].mean() - ref["win"].std(),
                        ref["win"].mean() + ref["win"].std(),
                        color=NULLC, alpha=.18, lw=0)
    ax.scatter(days, win, s=8, color=NULLC, alpha=.7, lw=0, label="within one session")
    ax.scatter(days, btw, s=8, color=col, alpha=.7, lw=0, label="between sessions")
    gl = np.logspace(np.log10(.9), np.log10(1300), 60)
    for v, cc, lab in ((win, NULLC, "within"), (btw, col, "between")):
        c0, c1 = slope(np.log10(days), v)
        ax.plot(gl, c0 + c1 * np.log10(gl), color=cc, lw=1.4)
        print(f"  {mode:12s} {lab:8s} level slope {c1:+.4f} per decade of days")
    ax.set_xscale("log")
    ax.set(xlabel="days between sessions", ylabel=r"$d_\mathrm{MSA}$", xlim=(.8, 1500))
    ax.legend(fontsize=6.4, loc="upper left", handletextpad=.3, framealpha=.85)
    ax.grid(alpha=.5); ax.set_axisbelow(True)
    ax.set_title(f"a   M1, $D={d['D']}$, {mode}", loc="left", fontsize=7.6)
    ax.text(.985, .03, f"{d['n_sess']} sessions, {len(days)} pairs", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.2, color=ps.MUTED,
            bbox=dict(fc="white", ec="none", alpha=.75, pad=1.2))

    ax = axs[1]
    near, far = days <= 43, days > 365
    bins = np.linspace(min(win.min(), btw.min()) - .01, max(win.max(), btw.max()) + .01, 26)
    groups = [(win, None, NULLC, .55, "within one session"),
              (btw[near], win[near], col, .35, r"between, $\leq$ 42 d"),
              (btw[far], win[far], col, .85, "between, $>$ 1 yr")]
    for v, _, cc, al, lab in groups:
        ax.hist(v, bins=bins, density=True, color=cc, alpha=al, label=lab)
    top = ax.get_ylim()[1]
    # headroom: histograms keep the bottom 55 per cent, the mean +- s.d. strip sits above them
    # and the legend above that, so nothing overlaps
    ax.set_ylim(0, top * 1.82)
    for i, (v, nul, cc, al, lab) in enumerate(groups):
        ax.axvline(np.median(v), color=cc, lw=1.1, alpha=max(al, .6))
        y, sd = top * (1.05 + .13 * i), v.std(ddof=1)
        ax.errorbar(v.mean(), y, xerr=sd, color=cc, marker="o", ms=3, lw=1.0,
                    capsize=1.8, alpha=max(al, .6))
        if nul is None:
            continue
        # each between-session group against ITS OWN within-session null, the same paired
        # two-sided signed-rank test as Section 4.4. Two-sided, so a star can mean BELOW the
        # null as well as above: under split units the near group sits below it.
        pv = float(stats.wilcoxon(v, nul).pvalue)
        ax.text(v.mean() + sd + .004, y, f"({stars(pv)})", ha="left", va="center",
                fontsize=6.4, color=ps.INK)
        k = near if "42" in lab else far
        print(f"  {mode:12s} {lab}: n={k.sum()} between {btw[k].mean():.3f} "
              f"own null {win[k].mean():.3f} excess {np.mean(btw[k]-win[k]):+.4f} "
              f"p={pv:.2e} {100*np.mean(btw[k]>win[k]):.0f}% above")
    ax.set(xlabel=r"$d_\mathrm{MSA}$", ylabel="density")
    ax.set_title("b", loc="left", fontsize=7.6)
    ax.legend(fontsize=5.8, loc="upper left", handletextpad=.4, framealpha=.85)

    fig.tight_layout(pad=.5)
    ps.save(fig, os.path.join(HERE, out))


levels(U, T, "split units", "fig45_levels")
levels(T, None, "split trials", "fig45_levels_trials")

# figure 2: drift
fig, axs = plt.subplots(1, 2, figsize=(ps.WIDTH, 3.0))

ax = axs[0]
xs = list(range(len(BANDS)))
labs = [b[2] for b in BANDS]
for d, axx, cc, al, ls, mk, lab in ((U, ax, col, 1.0, "-", "o", "split units"),
                                    (T, ax, col, .45, "--", "s", "split trials")):
    ex = d["btw"] - d["win"]
    ms, es, ns = [], [], []
    for lo, hi, _ in BANDS:
        k = (d["days"] >= lo) & (d["days"] <= hi)
        ms.append(ex[k].mean()); es.append(ex[k].std(ddof=1) / np.sqrt(k.sum()))
        ns.append(int(k.sum()))
    axx.errorbar(xs, ms, yerr=es, color=cc, alpha=al, lw=1.3, ls=ls, marker=mk,
                 ms=4 if mk == "o" else 3.4, capsize=2.4, label=lab)
    d["bands"] = (ms, es, ns)
ax.axhline(0, color=ps.MUTED, lw=.8)
ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=6.4, rotation=45, ha="right")
ax.set_ylabel("excess over the null", fontsize=7.4)
ax.grid(alpha=.5, axis="y"); ax.set_axisbelow(True)
ax.set_title("a", loc="left", fontsize=7.6)
h = [plt.Line2D([], [], color=col, lw=1.3, marker="o", ms=3.6, label="split units"),
     plt.Line2D([], [], color=col, lw=1.3, ls="--", marker="s", ms=3.2, alpha=.45,
                label="split trials")]
ax.legend(handles=h, fontsize=6.0, loc="upper left", bbox_to_anchor=(0, .93),
          handletextpad=.4, framealpha=.85)
lo_c, hi_c = ax.get_ylim()
ax.set_ylim(lo_c, hi_c + .13 * (hi_c - lo_c))          # room for the row of counts
# counts go in one row along the top rather than beside each marker, where they collided
# with the second curve
for x, n in zip(xs, U["bands"][2]):
    ax.annotate(f"{n}", (x, .975), xycoords=("data", "axes fraction"), ha="center",
                va="top", fontsize=5.6, color=ps.MUTED)

ax = axs[1]
W = .34
for j, (d, name) in enumerate(((U, "split units"), (T, "split trials"))):
    x = np.log10(d["days"])
    for i, (y, cc, al, lab) in enumerate(((d["btw"], col, 1.0, "between sessions"),
                                          (d["win"], NULLC, 1.0, "within one session"))):
        s, se = jackknife(x, y, d["a"], d["b"])
        ax.bar(j + (i - .5) * W, s, W * .88, color=cc, alpha=al,
               label=lab if j == 0 else None)
        ax.errorbar(j + (i - .5) * W, s, yerr=se, color=ps.INK, lw=1.0, capsize=2.6)
        print(f"  {name:12s} {lab:20s} {s:+.4f} +- {se:.4f}  ({s/se:4.1f} SE)")
    sx, sse = jackknife(x, d["btw"] - d["win"], d["a"], d["b"])
    print(f"  {name:12s} {'excess':20s} {sx:+.4f} +- {sse:.4f}  ({sx/sse:4.1f} SE)")
ax.axhline(0, color=ps.MUTED, lw=.8)
ax.set_xticks([0, 1]); ax.set_xticklabels(["split units", "split trials"], fontsize=7)
ax.set_xlim(-.55, 1.55)
ax.set_ylabel("slope per decade of days", fontsize=7.4)
ax.set_title("b", loc="left", fontsize=7.6)
ax.legend(fontsize=6.2, loc="upper left", handletextpad=.4, framealpha=.85)
ax.grid(alpha=.5, axis="y"); ax.set_axisbelow(True)
lo_c, hi_c = ax.get_ylim()
ax.set_ylim(lo_c, hi_c * 1.30)

fig.tight_layout(pad=.5)
ps.save(fig, os.path.join(HERE, "fig45_drift"))
