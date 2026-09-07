"""mk44_slopes: the slope of the between-minus-within excess against days apart.

One bar per region and null, in four groups, with the two tangent constructions side by side in
each. Monkey C only.

The error bar is a delete-one-session jackknife: drop one of the 14 sessions, refit the slope on
the 78 pairs that do not involve it, repeat for all 14, and take

    se_jack = sqrt( (n-1)/n * sum_i (b_i - b_bar)^2 ),   n = 14.

The ordinary regression standard error is printed alongside it.

Reads the run_msa_dist_allpairs.py summaries (pushforward) and run_msa_direct.py. No refits.

Run from code/:  python3 mk44_slopes.py
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotstyle as ps

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
CODE = os.path.join(HERE, "..", "..", "code")
CONV = "pcubic_lam0.3_push"
REGIONS = [("M1", "msa_allpairs_{}_summary.json", "msa_direct_M1.json"),
           ("PMd", "msa_allpairs_PMd_{}_summary.json", "msa_direct_PMd.json")]
NULLS = [("trials", "split trials"), ("units", "split units")]


def slope(days, y):
    A = np.vstack([np.ones_like(days), days]).T
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    r = y - A @ b
    se = np.sqrt(r @ r / (len(days) - 2) * np.linalg.inv(A.T @ A)[1, 1])
    return b[1], se


def jackknife(pairs):
    """delete-one-session jackknife of the slope of (between - within) on days."""
    sess = sorted({p["a"] for p in pairs} | {p["b"] for p in pairs})
    full = np.array([[p["days"], p["between_mean"] - p["within_mean"]] for p in pairs])
    b_all, se_ols = slope(full[:, 0], full[:, 1])
    reps = []
    for s in sess:
        sub = np.array([[p["days"], p["between_mean"] - p["within_mean"]]
                        for p in pairs if p["a"] != s and p["b"] != s])
        reps.append(slope(sub[:, 0], sub[:, 1])[0])
    reps = np.array(reps)
    n = len(sess)
    se_j = np.sqrt((n - 1) / n * np.sum((reps - reps.mean()) ** 2))
    return b_all, se_ols, se_j, len(sess)


def pairs_direct(f, mode):
    return json.load(open(os.path.join(CODE, f)))[mode]["pairs"]


def pairs_push(fmt, mode):
    return json.load(open(os.path.join(CODE, fmt.format(mode))))["pairs"][CONV]





ps.apply()
fig, ax = plt.subplots(figsize=(ps.WIDTH, 2.5))
labels, out = [], {}
x = 0
W = 0.34
for reg, fmt, dfile in REGIONS:
    for mode, nice in NULLS:
        for j, (cons, pr) in enumerate([("pushforward", pairs_push(fmt, mode)),
                                        ("direct", pairs_direct(dfile, mode))]):
            b, se_ols, se_j, n = jackknife(pr)
            col = ps.COL[reg]
            ax.bar(x + (j - .5) * W, 10 * b, width=W * .92, color=col,
                   alpha=1.0 if j == 0 else .45, lw=0)
            ax.errorbar(x + (j - .5) * W, 10 * b, yerr=10 * se_j, color=ps.INK,
                        lw=.9, capsize=2.2, ls="none")
            out[f"{reg}|{mode}|{cons}"] = dict(slope10=10 * b, se_ols10=10 * se_ols,
                                                 se_jack10=10 * se_j, n_sessions=n)
            print(f"{reg:>3} {mode:<6} {cons:<11}: {10 * b:+.4f} per 10 d   "
                  f"jackknife SE {10 * se_j:.4f}   OLS SE {10 * se_ols:.4f}   "
                  f"{abs(b / se_j):.1f} SE from zero   n_sessions={n}")
        labels.append(f"{reg}\n{nice}")
        x += 1

ax.axhline(0, color=ps.MUTED, lw=.8)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("excess over the null\nper 10 days apart", fontsize=7.4)
h = [plt.Rectangle((0, 0), 1, 1, color=ps.INK, alpha=1.0, label="pushforward"),
     plt.Rectangle((0, 0), 1, 1, color=ps.INK, alpha=.45, label="direct")]
ax.legend(handles=h, fontsize=6.6, loc="upper right", frameon=False)
ax.set_title("monkey C, 14 sessions, 91 pairs", loc="left", fontsize=6.8, color=ps.MUTED)
ax.grid(alpha=.5, axis="y"); ax.set_axisbelow(True)
fig.tight_layout(pad=.4)
ps.save(fig, os.path.join(HERE, "fig44_slopes"))
json.dump(out, open(os.path.join(HERE, "msa_slopes.json"), "w"), indent=1)
