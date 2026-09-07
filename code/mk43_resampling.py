"""mk43_resampling.py — fig43_resampling: the resampling bands on the contraction, both animals.

Two panels, one per animal, on a shared vertical scale: monkey C at D = 8 / 12, monkey M at
D = 4 / 6. The quantity plotted is the dimensionless contraction L(end)/L(onset).

Each panel carries, per session and region:
    dot        the contraction from the fit to the whole session
    thin line  95% interval over B = 100 trial-bootstrap refits (mean +- 1.96 sd)
    pale bar   range over twenty half arrays: ten disjoint splits of the recorded units,
               each half fitted alone
The line at 1 is no contraction.

Reads resample43.jsonl and resample43_subm.jsonl. No refits.

Usage:  python3 mk43_resampling.py [--out figures/fig43_resampling] [--usetex]
"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import plotstyle as ps

ONSET = 5
REGIONS = ("M1", "PMd")
#: (label, jsonl, D per region) per animal, in panel order
ANIMALS = [("monkey C", "resample43.jsonl", {"M1": 8, "PMd": 12}),
           ("monkey M", "resample43_subm.jsonl", {"M1": 4, "PMd": 6})]

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="figures/fig43_resampling")
ap.add_argument("--usetex", action="store_true")
a = ap.parse_args()

ps.apply(usetex=a.usetex)


def ratio(rec):
    L = np.asarray(rec["L"], float)
    return L[-1] / L[ONSET]


def load(path):
    full, boot, half = {}, collections.defaultdict(list), collections.defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        k = (r["sess"], r["region"])
        if r["mode"] == "full":
            full[k] = r
        elif r["mode"] == "trials":
            boot[k].append(r)
        elif r["mode"] == "units":
            half[k].append(r)
    return full, boot, half


data = [(lab, load(path), D) for lab, path, D in ANIMALS]
nsess = [len({k[0] for k in d[1][0]}) for d in data]

fig, axes = plt.subplots(1, 2, figsize=(ps.WIDTH, 2.75), sharey=True,
                         gridspec_kw=dict(width_ratios=nsess, wspace=0.05,
                                          left=0.085, right=0.995, bottom=0.17, top=0.97))

off = {"M1": -0.155, "PMd": +0.155}
for ax, (lab, (full, boot, half), D) in zip(axes, data):
    SESS = sorted({k[0] for k in full})
    xs = np.arange(len(SESS))
    for reg in REGIONS:
        for i, s in enumerate(SESS):
            x = xs[i] + off[reg]
            v = np.array([ratio(r) for r in boot[(s, reg)]])
            m, sd = v.mean(), v.std(ddof=1)
            hv = np.array([ratio(r) for r in half[(s, reg)]])
            if len(hv):
                ax.plot([x, x], [hv.min(), hv.max()], color=ps.COL[reg], lw=3.4,
                        alpha=0.26, solid_capstyle="butt", zorder=1)
                for yv in (hv.min(), hv.max()):
                    ax.plot([x - 0.06, x + 0.06], [yv, yv], color=ps.COL[reg], lw=0.6,
                            alpha=0.75, zorder=1)
            ax.plot([x, x], [m - 1.96 * sd, m + 1.96 * sd], color=ps.COL[reg], lw=1.1,
                    zorder=2)
            ax.plot([x], [ratio(full[(s, reg)])], "o", color=ps.COL[reg], ms=3.4,
                    mec="white", mew=0.5, zorder=3)
    ax.axhline(1, color=ps.INK, lw=0.7)
    ax.set(xticks=xs, xticklabels=[s[4:6] + "/" + s[6:] for s in SESS],
           xlim=(-0.5, len(SESS) - 0.5), ylim=(0.06, 1.30))
    ax.set_xlabel(f"session ({SESS[0][:4]})")
    # the strip above the line at 1 is empty, so it carries two rows: the animal label
    # on top and the element key below it, neither ever sitting on a bar.
    ax.text(0.5, 0.965, f"{lab}   $D = {D['M1']}\\,/\\,{D['PMd']}$", transform=ax.transAxes,
            ha="center", va="top", fontsize=7.6)

axes[0].set_ylabel(r"$L(\mathrm{end})\,/\,L(\mathrm{onset})$")
axes[1].text(nsess[1] - 0.55, 1.015, "no contraction", fontsize=6.5, color=ps.INK,
             ha="right", va="bottom")

# region key on the left panel, element key spanning the top of both
axes[0].text(0.02, 0.845, "M1", transform=axes[0].transAxes, color=ps.M1, fontsize=8,
             fontweight="bold", va="center")
axes[0].text(0.15, 0.845, "PMd", transform=axes[0].transAxes, color=ps.PMD, fontsize=8,
             fontweight="bold", va="center")
handles = [Line2D([], [], color=ps.MUTED, marker="o", ls="none", ms=3.4, mec="white",
                  mew=0.5, label="full fit"),
           Line2D([], [], color=ps.MUTED, lw=1.1,
                  label="95\\% bootstrap CI" if a.usetex else "95% bootstrap CI"),
           Line2D([], [], color=ps.MUTED, lw=3.4, alpha=0.26,
                  label="range, 20 half-arrays")]
axes[1].legend(handles=handles, frameon=False, loc="center", ncol=3, handlelength=1.3,
               columnspacing=1.6, bbox_to_anchor=(0.5, 0.845))

ps.save(fig, a.out)

# numbers for prose
for lab, (full, boot, half), D in data:
    SESS = sorted({k[0] for k in full})
    ok = nh = nhc = 0
    print(f"\n-- {lab}, {len(SESS)} sessions --")
    for s in SESS:
        row = f"  {s} "
        for reg in REGIONS:
            v = np.array([ratio(r) for r in boot[(s, reg)]])
            m, sd = v.mean(), v.std(ddof=1)
            ok += (m + 1.96 * sd) < 1
            hv = [ratio(r) for r in half[(s, reg)]]
            nh += len(hv); nhc += sum(h < 1 for h in hv)
            row += (f" {reg} {ratio(full[(s, reg)]):.3f} "
                    f"[{m - 1.96 * sd:.3f},{m + 1.96 * sd:.3f}]  ")
        print(row)
    print(f"  {ok}/{2 * len(SESS)} intervals exclude 1; {nhc}/{nh} half fits contract")
    for reg in REGIONS:
        vals = [ratio(full[(s, reg)]) for s in SESS]
        print(f"  {reg} session mean {np.mean(vals):.3f}  range {min(vals):.3f}-{max(vals):.3f}")
