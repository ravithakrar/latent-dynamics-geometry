"""run_dt_regression.py — d_MSA against the calendar gap between sessions.

Fourteen sub-C 2016 centre-out sessions, 91 pairs spanning 1 to 42 days. Per session three
identity-init EM fits, 42 in all: full (all trials) and h1/h2 (stratified trial halves). Both
metric conventions are evaluated from the stored xbar/A/C/theta.

Significance is by Mantel permutation of the session labels, which preserves the dependency
between pairs that share a session. Also reported: Spearman rho, the same partialled for mean
unit count and |Delta M|, a saturating fit d = a + b(1 - exp(-dt/tau)) alongside the linear one
(both in the summary json, only the linear one drawn), and the within-session floor.

Usage
    python run_dt_regression.py --fit            # 42 fits, ~2 min after warm-up
    python run_dt_regression.py --report
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools as it
import json
import os
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import fitting as mr
import lds_em_highD as em
import metric_compare as met
import pullback_metric as pb
import tangent_data as td

REGION = "M1"
D = 8
NIT = 80
FIELDS_OUT = "dt_fields.npz"
OUT_JSON = "dt_regression.json"
N_PERM = 200_000
#: The one M1 comparison outside the 2016 block. The separation is derived from the two
#: session dates rather than typed, so it cannot disagree with them.
LONG_RANGE_PAIR = ("20150312", "20160914")
LONG_RANGE = 0.4538
SEED = 0

#: SHARED day-bin edges, half-open [lo, hi). Every binned display in this project uses these,
#: so a table row and a figure point always mean the same set of pairs.
#: Counts per bin over the 91 pairs: 13, 20, 17, 18, 15, 8.
#: The 1-day pairs are NOT given their own bin. There are only three of them and their mean
#: sits ~8 s.e.m. below the straight-line fit, so as a separate point it dominates the eye
#: while resting on n = 3. They are merged into 1-4 here; the 1-day value is still quoted
#: in the write-ups where it is discussed, with its n stated.
BIN_EDGES = (1, 5, 10, 16, 23, 31, 43)

#: every sub-C 2016 center-out session in DANDI:000688. The PMd array exists only in this
#: block, so this is also the only sub-C window where a two-region version is possible.
SESSIONS = ["20160909", "20160912", "20160914", "20160915", "20160919", "20160921",
            "20160923", "20160929", "20161005", "20161006", "20161007", "20161011",
            "20161013", "20161021"]
#: 20160914 lives inside y_tensors.npz; the rest are tensor_<date>.npz
TENSOR = {s: (f"tensor_{s}.npz" if s != "20160914" else "y_tensors.npz") for s in SESSIONS}

CONVENTIONS = [("pcubic_lam0.3_push", "pcubic", 0.3, True),
               ("cubic_lam0_push", "cubic", 0.0, True)]
VARIANTS = ("full", "h1", "h2")
STORE_FIELDS = ("xbar", "A", "C", "theta")


def date(s):
    return dt.date(int(s[:4]), int(s[4:6]), int(s[6:]))


LONG_RANGE_DAYS = (date(LONG_RANGE_PAIR[1]) - date(LONG_RANGE_PAIR[0])).days


def load(sess, root="."):
    z = np.load(os.path.join(root, TENSOR[sess]))
    y = np.asarray(z["y_" + REGION], float)
    targ = np.asarray(z["target_" + REGION], float)
    keep = np.isfinite(targ) & np.isfinite(y).all(axis=(1, 2))
    return y[keep], targ[keep]


# fits
def fit_one(y, targ):
    params, lls, _ = mr.fit_identity(y, D, mr.RF[REGION], NIT)
    A, C = np.asarray(params[0]), np.asarray(params[1])
    xh = np.asarray(em.e_step_batch(y, *params)[0])
    xbar, theta = pb.condition_means(xh, targ)
    return dict(xbar=np.asarray(xbar, float), A=A, C=C, theta=np.asarray(theta, float),
                ll=float(lls[-1]), eig_max=float(np.abs(np.linalg.eigvals(A)).max()),
                n=int(y.shape[0]), M=int(y.shape[2]))


def run_fits(root="."):
    store = dict(np.load(FIELDS_OUT)) if os.path.exists(FIELDS_OUT) else {}
    log = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {"fits": []}
    for sess in SESSIONS:
        y, targ = load(sess, root)
        m1, m2 = td.split_halves(targ, seed=0)          # SPLIT_SEED = 0, as msa_cache
        jobs = [("full", y, targ), ("h1", y[m1], targ[m1]), ("h2", y[m2], targ[m2])]
        for variant, yy, tt in jobs:
            if f"{sess}|{variant}|xbar" in store:
                continue
            t0 = time.time()
            d = fit_one(yy, tt)
            for f in STORE_FIELDS:
                store[f"{sess}|{variant}|{f}"] = d[f]
            log["fits"].append(dict(session=sess, variant=variant, n=d["n"], M=d["M"],
                                    ll=d["ll"], eig_max=d["eig_max"],
                                    secs=round(time.time() - t0, 1)))
            print(f"  {sess} {variant:<5} n={d['n']:<4} M={d['M']:<3} "
                  f"|eig|max {d['eig_max']:.4f}  [{time.time() - t0:.0f}s]", flush=True)
        np.savez_compressed(FIELDS_OUT, **store)
        json.dump(log, open(OUT_JSON, "w"), indent=1)
    print(f"\n{FIELDS_OUT}: {len({tuple(k.split('|')[:2]) for k in store})} fits", flush=True)


# stats
def rank(x):
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    sx = np.asarray(x, float)[o]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float(ra @ rb / np.sqrt((ra @ ra) * (rb @ rb)))


def partial_spearman(a, b, c):
    """Spearman of a vs b with c held out."""
    ab, ac, bc = spearman(a, b), spearman(a, c), spearman(b, c)
    return float((ab - ac * bc) / np.sqrt((1 - ac ** 2) * (1 - bc ** 2)))


def mantel(d, days, pairs, sessions, n_perm=N_PERM, seed=SEED):
    """Permute which session was recorded on which DATE, keeping the pair structure.

    The 91 pair distances share sessions, so they are not independent and an ordinary
    correlation p-value is anticonservative. Permuting session->date reassignments
    preserves that dependency exactly: every relabeled dataset has the same 14 sessions
    and the same 91 pairs, only the calendar is scrambled.
    """
    rng = np.random.default_rng(seed)
    idx = {s: i for i, s in enumerate(sessions)}
    ii = np.array([idx[a] for a, _ in pairs])
    jj = np.array([idx[b] for _, b in pairs])
    day0 = np.array([date(s).toordinal() for s in sessions], float)
    obs = spearman(d, days)
    ge = 0
    for _ in range(n_perm):
        p = day0[rng.permutation(len(sessions))]
        if spearman(d, np.abs(p[ii] - p[jj])) >= obs:
            ge += 1
    return obs, (ge + 1) / (n_perm + 1)


def fit_saturating(days, d):
    """Least squares d = a + b*(1 - exp(-dt/tau)) over a grid of tau."""
    best = None
    for tau in np.exp(np.linspace(np.log(0.5), np.log(400), 400)):
        X = np.vstack([np.ones_like(days), 1 - np.exp(-days / tau)]).T
        coef, *_ = np.linalg.lstsq(X, d, rcond=None)
        sse = float(((X @ coef - d) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, float(coef[0]), float(coef[1]), float(tau))
    sse, a, b, tau = best
    return dict(a=a, b=b, tau_days=tau, sse=sse,
                r2=float(1 - sse / ((d - d.mean()) ** 2).sum()))


# report
def build_g(z, sess, variant, method, lam, push):
    d = {f: z[f"{sess}|{variant}|{f}"] for f in STORE_FIELDS}
    fn = pb.metric_pushforward if push else pb.metric
    g, _, _ = fn(d["xbar"], d["A"], C=d["C"], theta=d["theta"], method=method, lam=lam)
    return np.asarray(g, float)


def collect(label, method, lam, push):
    z = np.load(FIELDS_OUT)
    have = [s for s in SESSIONS if f"{s}|full|xbar" in z.files]
    g = {s: build_g(z, s, "full", method, lam, push) for s in have}
    floor = {s: float(met.msa(build_g(z, s, "h1", method, lam, push),
                              build_g(z, s, "h2", method, lam, push))["d_msa"])
             for s in have}
    M = {s: int(z[f"{s}|full|C"].shape[0]) for s in have}

    pairs = list(it.combinations(have, 2))
    d = np.array([met.msa(g[a], g[b])["d_msa"] for a, b in pairs])
    days = np.array([abs((date(a) - date(b)).days) for a, b in pairs], float)
    meanM = np.array([(M[a] + M[b]) / 2 for a, b in pairs], float)
    dM = np.array([abs(M[a] - M[b]) for a, b in pairs], float)

    rho, p = mantel(d, days, pairs, have)
    slope, icpt = np.polyfit(days, d, 1)
    sat = fit_saturating(days, d)
    return dict(
        convention=label, n_sessions=len(have), n_pairs=len(pairs),
        days_min=float(days.min()), days_max=float(days.max()),
        rho=rho, mantel_p=p,
        rho_partial_meanM=partial_spearman(d, days, meanM),
        rho_partial_dM=partial_spearman(d, days, dM),
        rho_d_vs_meanM=spearman(d, meanM), rho_d_vs_dM=spearman(d, dM),
        slope_per_10d=float(slope * 10), intercept=float(icpt),
        saturating=sat,
        floor_mean=float(np.mean(list(floor.values()))),
        floor_min=float(min(floor.values())), floor_max=float(max(floor.values())),
        d_mean=float(d.mean()), d_min=float(d.min()), d_max=float(d.max()),
        pairs=[dict(a=a, b=b, days=float(t), d_msa=float(v))
               for (a, b), t, v in zip(pairs, days, d)],
        floors=floor, units=M,
    )


def report(out_prefix="dt_regression"):
    res = [collect(*c) for c in CONVENTIONS]
    hdr = (f"{'convention':<22}{'pairs':>6}{'rho':>8}{'Mantel p':>10}{'rho|M':>8}"
           f"{'slope/10d':>11}{'icpt':>7}{'tau(d)':>8}{'floor':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in res:
        print(f"{r['convention']:<22}{r['n_pairs']:>6}{r['rho']:>+8.3f}{r['mantel_p']:>10.4f}"
              f"{r['rho_partial_meanM']:>+8.3f}{r['slope_per_10d']:>+11.4f}"
              f"{r['intercept']:>7.3f}{r['saturating']['tau_days']:>8.1f}"
              f"{r['floor_mean']:>8.3f}")
    json.dump(dict(region=REGION, D=D, nit=NIT, sessions=SESSIONS, results=res),
              open(f"{out_prefix}_summary.json", "w"), indent=1)
    print(f"\nWROTE {out_prefix}_summary.json")
    make_figure(res, out_prefix)
    return res


def make_figure(res, out_prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PT, FIT, FLOOR = "#2563eb", "#d1495b", "#6b6b68"
    INK, MUTED, SURF = "#1b1b1a", "#6b6b68", "white"
    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 9.8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "text.color": INK,
                         "legend.frameon": False,
                         "figure.facecolor": SURF, "axes.facecolor": SURF})

    r = res[0]
    days = np.array([p["days"] for p in r["pairs"]])
    d = np.array([p["d_msa"] for p in r["pairs"]])

    fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.1))

    ax = axs[0]
    ax.scatter(days, d, s=26, color=PT, alpha=.62, edgecolor="none",
               label=f"{len(d)} session pairs")
    xs = np.linspace(0, days.max() * 1.04, 200)
    ax.plot(xs, r["intercept"] + r["slope_per_10d"] / 10 * xs, color=FIT, lw=1.9,
            label=f"linear fit  {r['slope_per_10d']:+.3f} / 10 d")
    ax.axhspan(r["floor_min"], r["floor_max"], color=FLOOR, alpha=.13)
    ax.axhline(r["floor_mean"], color=FLOOR, lw=1.4, ls=":",
               label=f"within-session floor  {r['floor_mean']:.3f}")
    ax.set(xlabel="days between sessions", ylabel=r"$d_{\rm MSA}$",
           title=f"A · M1, all 14 sub-C 2016 sessions\n"
                 f"$\\rho$ = {r['rho']:+.3f},  Mantel p = {r['mantel_p']:.4f}")
    ax.legend(fontsize=7.7, loc="lower right")

    # B — binned means, so the trend is legible without reading through the scatter
    ax = axs[1]
    edges = np.array(BIN_EDGES)
    ctr, mu, se, nn = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (days >= lo) & (days < hi)
        if m.sum() < 2:
            continue
        ctr.append(days[m].mean()); mu.append(d[m].mean())
        se.append(d[m].std(ddof=1) / np.sqrt(m.sum())); nn.append(int(m.sum()))
    ax.errorbar(ctr, mu, yerr=se, fmt="o-", color=PT, lw=1.8, capsize=3, ms=6.5)
    for x, y, n in zip(ctr, mu, nn):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=7.2, color=MUTED)
    ax.axhspan(r["floor_min"], r["floor_max"], color=FLOOR, alpha=.13)
    ax.axhline(r["floor_mean"], color=FLOOR, lw=1.4, ls=":")
    ax.set(xlabel="days between sessions", ylabel=r"$d_{\rm MSA}$",
           title="B · Binned, mean ± s.e.m.")

    fig.suptitle(f"{r['convention']} · D = {D} · shaded band = range of the 14 "
                 f"within-session floors", fontsize=9, y=1.02, color=MUTED)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/fig_{out_prefix}.{ext}", dpi=150,
                    bbox_inches="tight", facecolor=SURF)
    print(f"WROTE figures/fig_{out_prefix}.pdf (+ .png)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fit", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--root", default=".")
    a = p.parse_args()
    if a.fit:
        print(f"fitting {len(SESSIONS)} sessions x 3 variants, "
              f"{REGION} D={D} nit={NIT} identity init\n", flush=True)
        run_fits(a.root)
    if a.report or not a.fit:
        report()


if __name__ == "__main__":
    main()
