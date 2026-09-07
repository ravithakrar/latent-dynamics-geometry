"""run_msa_distributions.py — within- and between-session d_MSA as distributions.

Per split seed, fits four models (two sessions x two halves) and records two within-session
distances and four between-session ones from those same fits, so the two samples are paired
seed by seed and both are built from half-size fits.

Each fit is stored as xbar/A/C/theta, so every metric convention can be evaluated from one set
of fits.

M1 only, D = 8, nit = 80, identity init. Trial splits are stratified within direction; unit
splits are uniform over units. Trials are not subsampled in units mode, nor units in trials
mode.

Usage
    python run_msa_distributions.py --mode trials --seeds 0 12    # chunk, resumable
    python run_msa_distributions.py --mode trials --all
    python run_msa_distributions.py --report                      # tests + figure, no fitting
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import plotstyle as ps          # shared thesis figure style
import jax

jax.config.update("jax_enable_x64", True)

import fitting as mr
import lds_em_highD as em
import metric_compare as met
import pullback_metric as pb
import tangent_data as td

# settings
REGION = "M1"
PAIR = ("20160914", "20160929")
D = 8
NIT = 80
N_SEEDS = 50
FIELDS_OUT = "msa_dist_fields.npz"
OUT_JSON = "msa_distributions.json"

#: (label, method, lam, push).  Evaluated from the SAME fits; adding one costs no EM.
CONVENTIONS = [
    ("pcubic_lam0.3_push", "pcubic", 0.3, True),
    ("cubic_lam0_push", "cubic", 0.0, True),
]

STORE_FIELDS = ("xbar", "A", "C", "theta")


# splits
def split_units(M, seed):
    """Uniform split of unit indices into two disjoint halves. Not stratified by rate."""
    idx = np.random.default_rng(1000 + seed).permutation(M)
    return np.sort(idx[: M // 2]), np.sort(idx[M // 2:])


def halves(y, targ, seed, mode):
    """-> [(y1, targ1), (y2, targ2)] for the requested split axis.

    trials      all units, disjoint trial halves      (shared neurons, half the trials)
    units       all trials, disjoint unit halves      (no shared neurons, half the array)
    units_ctrl  ONE unit half, disjoint trial halves  (shared neurons, half the array)

    units_ctrl is the control that makes `units` readable. `units` changes two things at
    once -- the array shrinks AND the two fits see no neuron in common -- so on its own it
    cannot say which one moved the floor. units_ctrl holds the array size at exactly the
    same M (it reuses `units`' first half verbatim) and varies only the trials, isolating
    the size effect. Whatever `units` costs on top of units_ctrl is the price of
    DISJOINTNESS, which is the property a between-session comparison also has.
    """
    if mode == "trials":
        m1, m2 = td.split_halves(targ, seed=seed)
        return [(y[m1], targ[m1]), (y[m2], targ[m2])]
    u1, u2 = split_units(y.shape[2], seed)
    if mode == "units":
        return [(y[:, :, u1], targ), (y[:, :, u2], targ)]
    if mode == "units_ctrl":
        ys = y[:, :, u1]                       # same first half as `units`, same M
        m1, m2 = td.split_halves(targ, seed=seed)
        return [(ys[m1], targ[m1]), (ys[m2], targ[m2])]
    raise ValueError(f"unknown mode {mode!r}")


# fit
def fit_one(y, targ):
    """One identity-init EM fit -> the arrays every convention is built from."""
    params, lls, _ = mr.fit_identity(y, D, mr.RF[REGION], NIT)
    A = np.asarray(params[0])
    C = np.asarray(params[1])
    xh = np.asarray(em.e_step_batch(y, *params)[0])
    xbar, theta = pb.condition_means(xh, targ)
    return dict(xbar=np.asarray(xbar, float), A=A, C=C, theta=np.asarray(theta, float),
                ll=float(lls[-1]), eig_max=float(np.abs(np.linalg.eigvals(A)).max()),
                n=int(y.shape[0]), M=int(y.shape[2]))


def key(mode, seed, sess, half, field):
    return f"{mode}|{seed}|{sess}|h{half}|{field}"


def run_seeds(seeds, mode, root="."):
    store = dict(np.load(FIELDS_OUT)) if os.path.exists(FIELDS_OUT) else {}
    log = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {"fits": []}
    tensors = {s: td.load_session_tensor(s, root=root) for s in PAIR}

    for seed in seeds:
        for sess in PAIR:
            y = np.asarray(tensors[sess][REGION]["y"], float)
            targ = np.asarray(tensors[sess][REGION]["targ"], float)
            for h, (yy, tt) in enumerate(halves(y, targ, seed, mode), start=1):
                if key(mode, seed, sess, h, "xbar") in store:
                    continue
                t0 = time.time()
                d = fit_one(yy, tt)
                for f in STORE_FIELDS:
                    store[key(mode, seed, sess, h, f)] = d[f]
                log["fits"].append(dict(mode=mode, seed=seed, session=sess, half=h,
                                        n=d["n"], M=d["M"], ll=d["ll"],
                                        eig_max=d["eig_max"],
                                        secs=round(time.time() - t0, 1)))
                print(f"  {mode} seed {seed:>2} {sess} h{h}  n={d['n']:<4} M={d['M']:<4} "
                      f"|eig|max {d['eig_max']:.4f}  [{time.time() - t0:.0f}s]", flush=True)
        np.savez_compressed(FIELDS_OUT, **store)          # checkpoint per seed
        json.dump(log, open(OUT_JSON, "w"), indent=1)

    np.savez_compressed(FIELDS_OUT, **store)
    json.dump(log, open(OUT_JSON, "w"), indent=1)
    have = {tuple(k.split("|")[:4]) for k in store}
    print(f"\n{FIELDS_OUT}: {len(have)} fits stored", flush=True)


def missing(mode, n_seeds=N_SEEDS):
    if not os.path.exists(FIELDS_OUT):
        return list(range(n_seeds))
    have = set(np.load(FIELDS_OUT).files)
    out = []
    for s in range(n_seeds):
        if not all(key(mode, s, sess, h, "xbar") in have
                   for sess in PAIR for h in (1, 2)):
            out.append(s)
    return out


# metrics
def build_g(d, method, lam, push):
    fn = pb.metric_pushforward if push else pb.metric
    g, _, _ = fn(d["xbar"], d["A"], C=d["C"], theta=d["theta"], method=method, lam=lam)
    return np.asarray(g, float)


def cos_alpha(g):
    """Normalised metric off-diagonal, averaged over the (direction, time) grid."""
    den = np.sqrt(np.maximum(g[..., 0, 0] * g[..., 1, 1], 1e-300))
    return float(np.nanmean(g[..., 0, 1] / den))


def collect(mode, label, method, lam, push):
    """-> dict of within/between value lists for one convention, over all completed seeds."""
    z = np.load(FIELDS_OUT)
    have = set(z.files)
    A_s, B_s = PAIR
    within = {A_s: [], B_s: []}
    between, paired, alphas, seeds_used = [], [], {A_s: [], B_s: []}, []

    for seed in range(N_SEEDS):
        if not all(key(mode, seed, s, h, "xbar") in have for s in PAIR for h in (1, 2)):
            continue
        seeds_used.append(seed)
        g = {}
        for s in PAIR:
            for h in (1, 2):
                d = {f: z[key(mode, seed, s, h, f)] for f in STORE_FIELDS}
                g[(s, h)] = build_g(d, method, lam, push)
                alphas[s].append(cos_alpha(g[(s, h)]))
        w = [met.msa(g[(s, 1)], g[(s, 2)])["d_msa"] for s in PAIR]
        for s, v in zip(PAIR, w):
            within[s].append(float(v))
        b = [met.msa(g[(A_s, i)], g[(B_s, j)])["d_msa"]
             for i in (1, 2) for j in (1, 2)]
        between += [float(v) for v in b]
        paired.append(dict(seed=seed, within_mean=float(np.mean(w)),
                           between_mean=float(np.mean(b)),
                           within_A=float(w[0]), within_B=float(w[1])))

    return dict(mode=mode, convention=label, method=method, lam=lam, push=push,
                seeds=seeds_used, n_seeds=len(seeds_used),
                within_A=within[A_s], within_B=within[B_s],
                within=within[A_s] + within[B_s], between=between,
                paired=paired,
                cos_alpha_A=alphas[A_s], cos_alpha_B=alphas[B_s])


# tests
def mannwhitney(x, y):
    """Two-sided Mann-Whitney U with a normal approximation and tie correction.

    Also returns the common-language effect size P(Y > X) + 0.5 P(Y = X) -- the probability
    that a randomly drawn between-session value exceeds a randomly drawn within-session one.
    0.5 means the distributions are interchangeable, 1.0 means they do not overlap at all.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    allv = np.concatenate([x, y])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), float)
    sv = allv[order]
    i = 0
    ties = 0.0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        t = j - i + 1
        ties += t ** 3 - t
        i = j + 1
    Rx = ranks[:nx].sum()
    Ux = Rx - nx * (nx + 1) / 2.0
    Uy = nx * ny - Ux
    mu = nx * ny / 2.0
    n = nx + ny
    sd = np.sqrt(nx * ny / 12.0 * ((n + 1) - ties / (n * (n - 1))))
    zstat = (min(Ux, Uy) - mu + 0.5) / sd
    from math import erfc, sqrt
    p = erfc(abs(zstat) / sqrt(2.0))
    return dict(U=float(min(Ux, Uy)), z=float(zstat), p=float(p),
                cles=float(Uy / (nx * ny)))


def paired_sign_test(diffs):
    """Exact two-sided sign test on per-seed (between_mean - within_mean)."""
    from math import comb
    d = np.asarray(diffs, float)
    d = d[d != 0]
    n, k = len(d), int((d > 0).sum())
    if n == 0:
        return dict(n=0, n_positive=0, p=float("nan"))
    tail = min(k, n - k)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(tail + 1)) / 2.0 ** n)
    return dict(n=n, n_positive=k, p=float(p))


def overlap_fraction(x, y):
    """Fraction of the two supports that overlap: |[max min, min max]| / |[min, max]|."""
    lo, hi = max(np.min(x), np.min(y)), min(np.max(x), np.max(y))
    span = max(np.max(x), np.max(y)) - min(np.min(x), np.min(y))
    return float(max(hi - lo, 0.0) / span) if span > 0 else 0.0


def summarise(res):
    w, b = np.array(res["within"]), np.array(res["between"])
    mw = mannwhitney(w, b)
    sg = paired_sign_test([p["between_mean"] - p["within_mean"] for p in res["paired"]])
    return dict(
        mode=res["mode"], convention=res["convention"], n_seeds=res["n_seeds"],
        within_mean=float(w.mean()), within_sd=float(w.std(ddof=1)),
        within_A_mean=float(np.mean(res["within_A"])),
        within_B_mean=float(np.mean(res["within_B"])),
        between_mean=float(b.mean()), between_sd=float(b.std(ddof=1)),
        ratio=float(b.mean() / w.mean()),
        n_within=len(w), n_between=len(b),
        mannwhitney=mw, paired_sign=sg,
        overlap_fraction=overlap_fraction(w, b),
        cos_alpha_A=float(np.mean(res["cos_alpha_A"])),
        cos_alpha_B=float(np.mean(res["cos_alpha_B"])),
    )


# report
def report(modes=("trials", "units", "units_ctrl"), out_prefix="msa_distributions"):
    all_res, all_sum = [], []
    for mode in modes:
        for label, method, lam, push in CONVENTIONS:
            r = collect(mode, label, method, lam, push)
            if r["n_seeds"] == 0:
                continue
            all_res.append(r)
            all_sum.append(summarise(r))

    if not all_sum:
        print("nothing to report yet")
        return

    hdr = (f"{'mode':<12}{'convention':<22}{'seeds':>6}{'within':>16}{'between':>16}"
           f"{'ratio':>7}{'p(MWU)':>10}{'CLES':>7}{'ovlp':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for s in all_sum:
        wcol = "%.3f ± %.3f" % (s["within_mean"], s["within_sd"])
        bcol = "%.3f ± %.3f" % (s["between_mean"], s["between_sd"])
        print(f"{s['mode']:<12}{s['convention']:<22}{s['n_seeds']:>6}{wcol:>16}{bcol:>16}"
              f"{s['ratio']:>7.2f}{s['mannwhitney']['p']:>10.2e}"
              f"{s['mannwhitney']['cles']:>7.2f}{s['overlap_fraction']:>7.2f}")

    json.dump(dict(pair=PAIR, region=REGION, D=D, nit=NIT,
                   conventions=[c[0] for c in CONVENTIONS],
                   summary=all_sum,
                   raw={f"{r['mode']}|{r['convention']}":
                        {k: r[k] for k in ("within_A", "within_B", "between", "paired",
                                           "cos_alpha_A", "cos_alpha_B", "seeds")}
                        for r in all_res}),
              open(f"{out_prefix}_summary.json", "w"), indent=1)
    print(f"\nWROTE {out_prefix}_summary.json")
    make_figure(all_res, all_sum, out_prefix)
    return all_sum


def make_figure(all_res, all_sum, out_prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    WITHIN, BETWEEN = "#2563eb", "#d1495b"
    INK, MUTED, SURF = "#1b1b1a", "#6b6b68", "white"
    ps.apply()

    #: panels A-C show the PRIMARY convention only; the second convention differs by <0.015
    #: everywhere and is in the json. Order is the argument of the figure: trial noise,
    #: then the same at reduced array size, then disjoint neurons.
    ORDER = ["trials", "units_ctrl", "units"]
    TITLE = {"trials":     "A · Split TRIALS, full array",
             "units_ctrl": "B · Split TRIALS, half array",
             "units":      "C · Split NEURONS, half array"}
    SUB = {"trials":     "shared neurons, half the trials",
           "units_ctrl":  "shared neurons, half the array",
           "units":      "no shared neurons, half the array"}
    prim = CONVENTIONS[0][0]
    panels = [(r, s) for r, s in zip(all_res, all_sum) if r["convention"] == prim]
    panels = [p for m in ORDER for p in panels if p[0]["mode"] == m]
    if not panels:
        return

    allv = np.concatenate([np.array(r["within"] + r["between"]) for r, _ in panels])
    bins = np.linspace(allv.min() - 0.015, allv.max() + 0.015, 34)

    fig, axs = plt.subplots(1, len(panels) + 1,
                            figsize=(ps.WIDTH, 2.9))
    for ax, (r, s) in zip(axs, panels):
        w, b = np.array(r["within"]), np.array(r["between"])
        ax.hist(w, bins=bins, color=WITHIN, alpha=.60, label=f"within  (n={len(w)})")
        ax.hist(b, bins=bins, color=BETWEEN, alpha=.60, label=f"between (n={len(b)})")
        ax.axvline(w.mean(), color=WITHIN, lw=1.7)
        ax.axvline(b.mean(), color=BETWEEN, lw=1.7)
        verdict = ("separated" if s["overlap_fraction"] < 0.05
                   else "indistinguishable" if s["mannwhitney"]["p"] > 0.05 else "overlapping")
        ax.set(xlabel=r"$d_{\rm MSA}$", ylabel="count", xlim=(bins[0], bins[-1]),
               title=f"{TITLE[r['mode']]}\n{SUB[r['mode']]}")
        ax.text(.97, .04, f"ratio {s['ratio']:.2f}\nCLES {s['mannwhitney']['cles']:.2f}\n"
                          f"p = {s['mannwhitney']['p']:.1e}\n{verdict}",
                transform=ax.transAxes, va="bottom", ha="right", fontsize=7.8, color=MUTED)
        ax.legend(fontsize=7.6, loc="upper right")

    # D — the decomposition: what each null costs, and where between-session sits
    ax = axs[-1]
    xs = np.arange(len(panels))
    wm = np.array([s["within_mean"] for _, s in panels])
    ws = np.array([s["within_sd"] for _, s in panels])
    bm = np.array([s["between_mean"] for _, s in panels])
    bs = np.array([s["between_sd"] for _, s in panels])
    # yerr is the S.D. OF THE DISTRIBUTION, not the s.e.m. of the mean. With n = 100 / 200 the
    # s.e.m. would be ~10x smaller and invisible; the s.d. is what makes this panel carry the
    # same information as the histograms beside it. Said in the axis label so it is not guessed.
    ax.errorbar(xs - .07, wm, yerr=ws, fmt="s", color=WITHIN, capsize=3, ms=7,
                lw=1.6, label="within-session null")
    ax.errorbar(xs + .07, bm, yerr=bs, fmt="o", color=BETWEEN, capsize=3, ms=7,
                lw=1.6, label="between-session")
    ax.plot(xs - .07, wm, color=WITHIN, lw=1.2, ls="--")
    ax.plot(xs + .07, bm, color=BETWEEN, lw=1.2, ls="--")
    for i, (_, s) in enumerate(panels):
        ax.annotate(f"{s['ratio']:.2f}x", (i, max(wm[i], bm[i]) + .055),
                    ha="center", fontsize=8.4, color=INK)
    ax.annotate("", xy=(0.86, wm[1]), xytext=(0.86, wm[0]),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.1))
    ax.text(0.80, (wm[0] + wm[1]) / 2, f"smaller array\n+{wm[1] - wm[0]:.3f}",
            fontsize=7.6, color=MUTED, va="center", ha="right")
    ax.annotate("", xy=(1.86, wm[2]), xytext=(1.86, wm[1]),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
    ax.text(1.80, (wm[1] + wm[2]) / 2, f"DISJOINT neurons\n+{wm[2] - wm[1]:.3f}",
            fontsize=7.9, color=INK, va="center", ha="right", fontweight="medium")
    ax.set_xticks(xs)
    ax.set_xticklabels([TITLE[r["mode"]].split("· ")[1].replace(", ", "\n") for r, _ in panels],
                       fontsize=8.2)
    ax.set(ylabel=r"$d_{\rm MSA}$", ylim=(0, max(bm.max(), wm.max()) + .13),
           xlim=(-0.45, 2.35),
           title="D · What each null costs\n(error bars: s.d. of the distribution, not s.e.m.)")
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(f"M1, {PAIR[0]} vs {PAIR[1]} · D={D} · {prim} · "
                 f"{panels[0][1]['n_seeds']} split seeds", fontsize=9.6, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/fig_{out_prefix}.{ext}", dpi=150,
                    bbox_inches="tight", facecolor=SURF)
    print(f"WROTE figures/fig_{out_prefix}.pdf (+ .png)")


# main
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("trials", "units", "units_ctrl"), default="trials")
    p.add_argument("--seeds", nargs=2, type=int, metavar=("START", "STOP"),
                   help="half-open seed range to fit in this process")
    p.add_argument("--all", action="store_true", help="fit every seed still missing")
    p.add_argument("--n-seeds", type=int, default=N_SEEDS)
    p.add_argument("--budget", type=float, default=None,
                   help="stop starting new seeds after this many seconds")
    p.add_argument("--report", action="store_true", help="tests + figure, no fitting")
    p.add_argument("--root", default=".")
    a = p.parse_args()
    globals()["N_SEEDS"] = a.n_seeds

    if a.report:
        return report()

    if a.all:
        seeds = missing(a.mode, a.n_seeds)
    elif a.seeds:
        seeds = [s for s in range(*a.seeds) if s in set(missing(a.mode, a.n_seeds))]
    else:
        p.error("give --seeds START STOP, or --all, or --report")

    print(f"{a.mode}: {len(seeds)} seeds to fit  {seeds}\n"
          f"  pair={PAIR} region={REGION} D={D} nit={NIT} init=identity\n", flush=True)
    if not seeds:
        return
    if a.budget:
        t0 = time.time()
        for s in seeds:
            if time.time() - t0 > a.budget:
                print(f"budget {a.budget}s reached, stopping before seed {s}", flush=True)
                break
            run_seeds([s], a.mode, a.root)
    else:
        run_seeds(seeds, a.mode, a.root)


if __name__ == "__main__":
    main()
