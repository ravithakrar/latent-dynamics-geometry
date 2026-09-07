"""run_msa_dist_allpairs.py — within and between d_MSA distributions for all 91 M1 session pairs.

Fits are cached per session rather than per pair: session s's two halves at seed k are fitted
once and reused by every pair containing s.

    14 sessions x 2 halves x 50 seeds  =  1,400 fits
      -> 14 x 50           =    700 within-session values
      -> 91 x 50 x 4 cross = 18,200 between-session values

Each pair's between-distribution uses the same seed-matched halves as both of its sessions'
within-distributions. Per pair it writes a within distribution (100 values), a between
distribution (200 values) and their Mann-Whitney, CLES and overlap summary, then the separation
against days apart over all 91 pairs.

M1 only, D = 8, nit = 80, identity init. Trial splits stratified within direction; unit splits
uniform over units. Conventions are evaluated after fitting from stored xbar/A/C/theta.

Usage
    python run_msa_dist_allpairs.py --mode trials --budget 480     # chunk, resumable
    python run_msa_dist_allpairs.py --mode trials --progress
    python run_msa_dist_allpairs.py --report
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools as it
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
from run_msa_distributions import mannwhitney, overlap_fraction, split_units

#: set by --region. D follows the per-region co-smoothing knee.
REGION = "M1"
D_REGION = {"M1": 8, "PMd": 12}
D = D_REGION[REGION]
NIT = 80
N_SEEDS = 50

#: MATCHED-N mode. None leaves every session at its own trial and unit counts.
#: When either is set, each session is first cut down to exactly this many units / trials,
#: with a fresh draw per seed, BEFORE the half-split. That is the only way to compare
#: sessions whose arrays differ in size: unit count is the largest single term in d_MSA, so
#: an unmatched comparison between a 41-unit session and a 95-unit one measures the array
#: before it measures the calendar.
MATCH_UNITS = None
MATCH_TRIALS = None
#: overrides tag() so a matched run gets its own store and its own summary files rather than
#: being merged into the full-array M1 store, whose fits are not comparable to it.
TAG = None
#: the fit store is SHARDED by seed. A single npz reaches ~25 MB, over the 20 MB per-file
#: cap on writing files back to the user's disk, and rewriting one big file on every
#: checkpoint gets slower as it grows. Shards of 25 seeds keep each file ~12 MB and make a
#: checkpoint cost only the shard being written.
#: shards are keyed by MODE as well as seed. Without the mode in the name the trials and
#: units stores share files and each one lands back over the 20 MB cap.
#: region (and D, when it is not that region's default) go in the FILENAME rather than in the
#: key, so load_store() globs one configuration at a time and the keys stay config-free.
#: A region at its OWN default D gets the bare region tag, so the M1 D=8 files written before
#: PMd was added keep working without a third rename.
def tag():
    if TAG:
        return TAG
    return REGION if D == D_REGION[REGION] else f"{REGION}D{D}"


def fields_glob():
    return f"msa_allpairs_fields_{tag()}_*_p*.npz"
#: 10 seeds per shard. PMd at D=12 stores ~1 MB per seed (258 units x 12 latents in C), so
#: 25 seeds in one file lands at 25 MB, over the 20 MB per-file write-back cap. 10 keeps every
#: shard under ~12 MB for the heaviest configuration.
SHARD_SEEDS = 10
LOG_JSON = "msa_allpairs_fits.json"
CKPT_EVERY = 5

SESSIONS = ["20160909", "20160912", "20160914", "20160915", "20160919", "20160921",
            "20160923", "20160929", "20161005", "20161006", "20161007", "20161011",
            "20161013", "20161021"]
TENSOR = {s: (f"tensor_{s}.npz" if s != "20160914" else "y_tensors.npz") for s in SESSIONS}

CONVENTIONS = [("pcubic_lam0.3_push", "pcubic", 0.3, True),
               ("cubic_lam0_push", "cubic", 0.0, True)]
STORE_FIELDS = ("xbar", "A", "C", "theta")


def shard_path(mode, seed):
    return f"msa_allpairs_fields_{tag()}_{mode}_p{seed // SHARD_SEEDS}.npz"


def load_store():
    """Merge every shard on disk into one dict."""
    import glob
    store = {}
    for f in sorted(glob.glob(fields_glob())):
        z = np.load(f)
        store.update({k: z[k] for k in z.files})
    return store


def save_store(store, only=None):
    """Write each seed-shard back, one file per SHARD_SEEDS block.

    `only` restricts writing to the given shard paths. This matters when two workers share
    the box: each loads the WHOLE store, so without the restriction a worker would rewrite
    the other's shard from its own stale copy and silently drop its fits.
    """
    shards = {}
    for k, v in store.items():
        mode, seed = k.split("|")[0], int(k.split("|")[1])
        shards.setdefault(shard_path(mode, seed), {})[k] = v
    for path, d in shards.items():
        if only is None or path in only:
            np.savez_compressed(path, **d)


def date(s):
    return dt.date(int(s[:4]), int(s[4:6]), int(s[6:]))


def days_apart(a, b):
    return abs((date(a) - date(b)).days)


def load(sess, root="."):
    z = np.load(os.path.join(root, TENSOR[sess]))
    y = np.asarray(z["y_" + REGION], float)
    targ = np.asarray(z["target_" + REGION], float)
    keep = np.isfinite(targ) & np.isfinite(y).all(axis=(1, 2))
    return y[keep], targ[keep]


def match_n(y, targ, seed):
    """Cut a session down to MATCH_UNITS units and MATCH_TRIALS trials. No-op when unset.

    Both draws are reseeded per `seed`, so the reported distribution is averaged over which
    units and which trials were kept rather than resting on one arbitrary subset. The trial
    draw is stratified by direction, on the same rule as the half-split, so all 8 conditions
    survive the reduction.
    """
    if MATCH_UNITS is not None and y.shape[2] > MATCH_UNITS:
        u = np.random.default_rng(7000 + seed).permutation(y.shape[2])[:MATCH_UNITS]
        y = y[:, :, np.sort(u)]
    # the per-direction cap is applied unconditionally, not only to sessions above the total.
    # Gated on the total it would leave a session that happens to sit at exactly MATCH_TRIALS
    # with its own unbalanced direction counts while every larger session got balanced, so two
    # sessions with the same trial count would not have the same design.
    if MATCH_TRIALS is not None:
        rng = np.random.default_rng(9000 + seed)
        dirs = np.unique(targ)
        per = MATCH_TRIALS // len(dirs)
        keep = []
        for d in dirs:
            idx = np.flatnonzero(targ == d)
            rng.shuffle(idx)
            keep.append(idx[:per])
        keep = np.sort(np.concatenate(keep))
        y, targ = y[keep], targ[keep]
    return y, targ


def halves(y, targ, seed, mode):
    """Two disjoint halves along the requested axis. Same rules as run_msa_distributions."""
    y, targ = match_n(y, targ, seed)
    if mode == "trials":
        m1, m2 = td.split_halves(targ, seed=seed)
        return [(y[m1], targ[m1]), (y[m2], targ[m2])]
    if mode == "units":
        u1, u2 = split_units(y.shape[2], seed)
        return [(y[:, :, u1], targ), (y[:, :, u2], targ)]
    raise ValueError(f"unknown mode {mode!r}")


def key(mode, seed, sess, h, field):
    return f"{mode}|{seed}|{sess}|h{h}|{field}"


# fits
def fit_one(y, targ):
    params, lls, _ = mr.fit_identity(y, D, mr.RF[REGION], NIT)
    A, C = np.asarray(params[0]), np.asarray(params[1])
    xh = np.asarray(em.e_step_batch(y, *params)[0])
    xbar, theta = pb.condition_means(xh, targ)
    return dict(xbar=np.asarray(xbar, float), A=A, C=C, theta=np.asarray(theta, float),
                ll=float(lls[-1]), eig_max=float(np.abs(np.linalg.eigvals(A)).max()),
                n=int(y.shape[0]), M=int(y.shape[2]))


def missing(mode, store_files, n_seeds=N_SEEDS, seed_range=None):
    seeds = range(n_seeds) if seed_range is None else range(*seed_range)
    return [(s, sess) for s in seeds for sess in SESSIONS
            if not all(key(mode, s, sess, h, "xbar") in store_files for h in (1, 2))]


def run(mode, budget=None, root=".", n_seeds=N_SEEDS, seed_range=None):
    # one worker per seed_range, each owning its own shard file, so two processes
    # can share the box without clobbering each other's store
    store = load_store()
    log = {"fits": []}
    todo = missing(mode, set(store), n_seeds, seed_range)
    print(f"{mode}: {len(todo)} (seed, session) cells outstanding of "
          f"{n_seeds * len(SESSIONS)}\n  D={D} nit={NIT} region={REGION} init=identity\n",
          flush=True)
    if not todo:
        return

    owned = {shard_path(mode, sd) for sd, _ in todo}
    tensors, t0, done = {}, time.time(), 0
    for seed, sess in todo:
        if budget and time.time() - t0 > budget:
            print(f"\nbudget {budget}s reached, stopping cleanly", flush=True)
            break
        if sess not in tensors:
            tensors[sess] = load(sess, root)
        y, targ = tensors[sess]
        for h, (yy, tt) in enumerate(halves(y, targ, seed, mode), start=1):
            if key(mode, seed, sess, h, "xbar") in store:
                continue
            t1 = time.time()
            d = fit_one(yy, tt)
            for f in STORE_FIELDS:
                store[key(mode, seed, sess, h, f)] = d[f]
            log["fits"].append(dict(mode=mode, seed=seed, session=sess, half=h, n=d["n"],
                                    M=d["M"], ll=d["ll"], eig_max=d["eig_max"],
                                    secs=round(time.time() - t1, 1)))
        done += 1
        if done % CKPT_EVERY == 0:
            save_store(store, only=owned)
            json.dump(log, open(LOG_JSON, "w"), indent=1)
            print(f"  {done}/{len(todo)} cells  [{time.time() - t0:.0f}s]", flush=True)

    save_store(store, only=owned)
    json.dump(log, open(LOG_JSON, "w"), indent=1)
    left = len(missing(mode, set(store), n_seeds, seed_range))
    print(f"\n{fields_glob()}: {len(store) // len(STORE_FIELDS)} fits stored, "
          f"{left} cells still outstanding", flush=True)


def progress(mode, n_seeds=N_SEEDS):
    have = set(load_store())
    left = missing(mode, have, n_seeds)
    tot = n_seeds * len(SESSIONS)
    print(f"{mode}: {tot - len(left)}/{tot} cells done ({len(left)} left)")
    return len(left)


# metrics
def build_g(z, mode, seed, sess, h, method, lam, push):
    d = {f: z[key(mode, seed, sess, h, f)] for f in STORE_FIELDS}
    fn = pb.metric_pushforward if push else pb.metric
    g, _, _ = fn(d["xbar"], d["A"], C=d["C"], theta=d["theta"], method=method, lam=lam)
    return np.asarray(g, float)


def collect(mode, label, method, lam, push, n_seeds=N_SEEDS):
    """-> per-session within lists and per-pair between lists, over completed seeds."""
    z = load_store()
    have = set(z)
    seeds = [s for s in range(n_seeds)
             if all(key(mode, s, sess, h, "xbar") in have
                    for sess in SESSIONS for h in (1, 2))]
    if not seeds:
        return None

    within = {s: [] for s in SESSIONS}
    between = {f"{a}|{b}": [] for a, b in it.combinations(SESSIONS, 2)}
    for seed in seeds:
        g = {(sess, h): build_g(z, mode, seed, sess, h, method, lam, push)
             for sess in SESSIONS for h in (1, 2)}
        for sess in SESSIONS:
            within[sess].append(float(met.msa(g[(sess, 1)], g[(sess, 2)])["d_msa"]))
        for a, b in it.combinations(SESSIONS, 2):
            between[f"{a}|{b}"] += [float(met.msa(g[(a, i)], g[(b, j)])["d_msa"])
                                    for i in (1, 2) for j in (1, 2)]
    return dict(mode=mode, convention=label, seeds=seeds, n_seeds=len(seeds),
                within=within, between=between)


def summarise_pairs(r):
    out = []
    for pk, b in r["between"].items():
        a, c = pk.split("|")
        w = np.array(r["within"][a] + r["within"][c])
        bb = np.array(b)
        mw = mannwhitney(w, bb)
        out.append(dict(
            a=a, b=c, days=days_apart(a, c),
            within_mean=float(w.mean()), within_sd=float(w.std(ddof=1)),
            between_mean=float(bb.mean()), between_sd=float(bb.std(ddof=1)),
            excess=float(bb.mean() - w.mean()),
            ratio=float(bb.mean() / w.mean()),
            cles=mw["cles"], p=mw["p"], overlap=overlap_fraction(w, bb),
            n_within=len(w), n_between=len(bb)))
    return sorted(out, key=lambda d: d["days"])


# report
def report(mode="trials", out_prefix=None):
    out_prefix = out_prefix or f"msa_allpairs_{tag()}"
    from run_dt_regression import mantel

    results, summaries = [], []
    for label, method, lam, push in CONVENTIONS:
        r = collect(mode, label, method, lam, push)
        if r is None:
            print("no complete seeds yet"); return
        results.append(r)
        summaries.append(summarise_pairs(r))

    r, S = results[0], summaries[0]
    days = np.array([p["days"] for p in S], float)
    exc = np.array([p["excess"] for p in S])
    cles = np.array([p["cles"] for p in S])
    btw = np.array([p["between_mean"] for p in S])
    pairs = [(p["a"], p["b"]) for p in S]

    print(f"\n{REGION} D={D} · {r['convention']} · {mode} · {r['n_seeds']} seeds · "
          f"{len(S)} pairs\n")
    hdr = (f"{'pair':<22}{'days':>5}{'within':>16}{'between':>16}{'excess':>8}"
           f"{'CLES':>7}{'ovlp':>7}{'p':>10}")
    print(hdr); print("-" * len(hdr))
    for p in S:
        w = "%.3f ± %.3f" % (p["within_mean"], p["within_sd"])
        b = "%.3f ± %.3f" % (p["between_mean"], p["between_sd"])
        print(f"{p['a']}-{p['b']:<9}{p['days']:>5}{w:>16}{b:>16}"
              f"{p['excess']:>+8.3f}{p['cles']:>7.2f}{p['overlap']:>7.2f}{p['p']:>10.1e}")

    sep = sum(1 for p in S if p["overlap"] < 0.05)
    ns = sum(1 for p in S if p["p"] > 0.05)
    print(f"\n{sep}/{len(S)} pairs fully separated (overlap < 0.05); "
          f"{ns}/{len(S)} not significant at 0.05")
    for name, v in (("excess", exc), ("CLES", cles), ("between mean", btw)):
        rho, pv = mantel(v, days, pairs, SESSIONS, n_perm=50_000)
        print(f"  {name:<13} vs days apart:  rho = {rho:+.3f}   Mantel p = {pv:.4f}")

    json.dump(dict(region=REGION, D=D, nit=NIT, mode=mode, sessions=SESSIONS,
                   conventions=[c[0] for c in CONVENTIONS],
                   n_seeds=r["n_seeds"],
                   pairs={c[0]: s for c, s in zip(CONVENTIONS, summaries)},
                   within={c[0]: res["within"] for c, res in zip(CONVENTIONS, results)}),
              open(f"{out_prefix}_{mode}_summary.json", "w"), indent=1)
    print(f"\nWROTE {out_prefix}_{mode}_summary.json")
    make_grid(r, S, mode, out_prefix)
    make_summary_fig(S, mode, out_prefix)
    return S


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ps.apply()
    return plt


WITHIN, BETWEEN = "#2563eb", "#d1495b"
INK, MUTED, SURF = "#1b1b1a", "#6b6b68", "white"


def make_grid(r, S, mode, out_prefix, n_show=12):
    """Histograms for n_show pairs spanning the Delta-t range, on one shared x-axis."""
    plt = _style()
    idx = np.unique(np.linspace(0, len(S) - 1, n_show).round().astype(int))
    show = [S[i] for i in idx]

    allv = []
    for p in show:
        allv += r["within"][p["a"]] + r["within"][p["b"]] + r["between"][f"{p['a']}|{p['b']}"]
    bins = np.linspace(min(allv) - .015, max(allv) + .015, 30)

    ncol = 4
    nrow = int(np.ceil(len(show) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(ps.WIDTH, 0.72 * ps.WIDTH / ncol * nrow),
                            sharex=True, sharey=True)
    for ax, p in zip(axs.ravel(), show):
        w = np.array(r["within"][p["a"]] + r["within"][p["b"]])
        b = np.array(r["between"][f"{p['a']}|{p['b']}"])
        ax.hist(w, bins=bins, color=WITHIN, alpha=.62)
        ax.hist(b, bins=bins, color=BETWEEN, alpha=.62)
        ax.axvline(w.mean(), color=WITHIN, lw=1.4)
        ax.axvline(b.mean(), color=BETWEEN, lw=1.4)
        ax.set_title(f"{p['a'][4:]} vs {p['b'][4:]}  ·  {p['days']} d\n"
                     f"CLES {p['cles']:.2f}, overlap {p['overlap']:.2f}", fontsize=8.6)
    for ax in axs.ravel()[len(show):]:
        ax.axis("off")
    for ax in axs[-1]:
        ax.set_xlabel(r"$d_{\rm MSA}$")
    for ax in axs[:, 0]:
        ax.set_ylabel("count")
    axs.ravel()[0].legend(handles=[
        plt.Line2D([], [], color=WITHIN, lw=6, alpha=.62, label="within"),
        plt.Line2D([], [], color=BETWEEN, lw=6, alpha=.62, label="between")],
        fontsize=7.6, loc="upper right")
    fig.suptitle(f"{REGION} D={D} · {mode} split · {r['convention']} · {r['n_seeds']} seeds · "
                 f"{len(show)} of {len(S)} pairs, ordered by days apart",
                 fontsize=9.6, y=1.005)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/fig_{out_prefix}_{mode}_grid.{ext}", dpi=150,
                    bbox_inches="tight", facecolor=SURF)
    print(f"WROTE figures/fig_{out_prefix}_{mode}_grid.pdf (+ .png)")


def make_summary_fig(S, mode, out_prefix):
    """All pairs at once: level, separation, and both against the calendar."""
    plt = _style()
    days = np.array([p["days"] for p in S], float)
    exc = np.array([p["excess"] for p in S])
    cles = np.array([p["cles"] for p in S])
    btw = np.array([p["between_mean"] for p in S])
    wth = np.array([p["within_mean"] for p in S])

    fig, axs = plt.subplots(1, 3, figsize=(ps.WIDTH, 2.35))

    ax = axs[0]
    ax.scatter(days, btw, s=26, color=BETWEEN, alpha=.7, edgecolor="none", label="between")
    ax.scatter(days, wth, s=26, color=WITHIN, alpha=.7, edgecolor="none",
               label="within (mean of the 2 sessions)")
    ax.set(xlabel="days between sessions", ylabel=r"$d_{\rm MSA}$",
           title="A · Level: both terms vs the calendar")
    # opaque frame: in units mode the cloud fills the panel and an unframed legend is
    # unreadable on top of it
    ax.legend(fontsize=7.8, loc="lower right", frameon=True, framealpha=.92,
              facecolor=SURF, edgecolor="none")

    ax = axs[1]
    ax.scatter(days, exc, s=28, color=INK, alpha=.72, edgecolor="none")
    z = np.polyfit(days, exc, 1)
    xs = np.linspace(0, days.max() * 1.04, 50)
    ax.plot(xs, np.polyval(z, xs), color=BETWEEN, lw=1.8)
    ax.axhline(0, color=MUTED, lw=1, ls=":")
    ax.set(xlabel="days between sessions", ylabel=r"between $-$ within",
           title="B · Excess over each pair's own floor")

    ax = axs[2]
    ax.scatter(days, cles, s=28, color=INK, alpha=.72, edgecolor="none")
    ax.axhline(0.5, color=MUTED, lw=1, ls=":")
    ax.axhline(1.0, color=MUTED, lw=1, ls="--")
    ax.set(xlabel="days between sessions", ylabel="CLES  P(between > within)",
           ylim=(0.4, 1.03), title="C · Separation vs the calendar")

    fig.suptitle(f"{REGION} D={D} · {mode} split · all {len(S)} session pairs", fontsize=9.6, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/fig_{out_prefix}_{mode}_summary.{ext}", dpi=150,
                    bbox_inches="tight", facecolor=SURF)
    print(f"WROTE figures/fig_{out_prefix}_{mode}_summary.pdf (+ .png)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--region", choices=("M1", "PMd"), default="M1")
    p.add_argument("--mode", choices=("trials", "units"), default="trials")
    p.add_argument("--D", type=int, default=None,
                   help="override the region default (M1 8, PMd 12); the fit store and outputs get a separate DN tag so configurations never mix")
    p.add_argument("--budget", type=float, default=None,
                   help="seconds; stop starting new cells after this, for shell chunking")
    p.add_argument("--n-seeds", type=int, default=N_SEEDS)
    p.add_argument("--seed-range", nargs=2, type=int, metavar=("START", "STOP"),
                   help="restrict this process to a half-open seed range, for running two "
                        "workers side by side; pair it with --shard-seeds so the workers "
                        "write disjoint shard files")
    p.add_argument("--shard-seeds", type=int, default=None)
    p.add_argument("--progress", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--root", default=".")
    a = p.parse_args()
    globals()["REGION"] = a.region
    globals()["D"] = a.D if a.D else D_REGION[a.region]
    globals()["N_SEEDS"] = a.n_seeds
    print(f"region={a.region}  D={globals()['D']}  mode={a.mode}  seeds={a.n_seeds}  "
          f"store={fields_glob()}", flush=True)
    if a.progress:
        return progress(a.mode, a.n_seeds)
    if a.report:
        return report(a.mode)
    run(a.mode, budget=a.budget, root=a.root, n_seeds=a.n_seeds)


if __name__ == "__main__":
    main()
