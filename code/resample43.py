"""resample43.py — trial-bootstrap and neuron-hold-out bands for the metric field.

Two resamplings, both refitting the model from scratch:

    trials   stratified bootstrap within each reach direction, with replacement, same trial
             count per direction, B replicates
    units    disjoint halves of the recorded units, drawn uniformly, R independent splits,
             both halves refitted

Quantities recorded are dimensionless: ratio_L = L(end)/L(onset), ratio_tt, ratio_qq and
cos alpha.

Config: the six monkey-C sessions of td.SESSIONS; D fixed at M1 = 8, PMd = 12 at every
resample; identity init, NIT = 80, unweighted, unit variance floor 0.01 / 0.05, matching
msa_cache.py; metric_convention.py (pcubic, lam = 0.3, pushforward), with lambda not
re-selected inside the loop; ring n_fine = 240, onset bin 5, T = 25 as build_ring_fields.py.

Writes one JSON line per fit to resample43.jsonl, so the run is resumable and chunkable and
nothing is held in memory across replicates.

Usage
    python resample43.py --mode full
    python resample43.py --mode trials --reps 0 100
    python resample43.py --mode units  --reps 0 10
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import tangent_data as td
import msa_cache as mc
import fitting as mr
import metric_convention as mcv
import geodesic_ring as gr

D_REGION = {"M1": 8, "PMd": 12}
NIT = 80

#: monkey M, the second animal. Its own operating point (co-smoothing knee at 4/6, not 8/12)
#: and its own tensors. Loaded exactly as build_subm_fields.py does, WITHOUT the unit-variance
#: floor that td.load_session_tensor applies, so the fits match the ones already in
#: subm_fields.npz and the rebuilt band lands on the curve already plotted.
SESSIONS_SUBM = ["20140203", "20140217", "20140218", "20140303",
                 "20140304", "20140306", "20140307"]
D_REGION_SUBM = {"M1": 4, "PMd": 6}
N_FINE = 240
ONSET = 5
REGIONS = ("M1", "PMd")
OUT = "resample43.jsonl"

#: seeds are derived, never drawn, so a rerun of replicate b is bit-identical
SEED_TRIALS = 10_000
SEED_UNITS = 20_000


# statistics
def entries(g):
    """(K,T,2,2) -> (sqrt g_tt, sqrt g_thth, cos alpha), each (T,), averaged over directions.

    Mean of the square root, not root of the mean -- matches mk_metric_entries.entries so the
    band lands on the curve that is already plotted.
    """
    gtt, gqq, gtq = g[..., 0, 0], g[..., 1, 1], g[..., 0, 1]
    cos = gtq / np.sqrt(np.maximum(gtt * gqq, 1e-30))
    return np.sqrt(gtt).mean(0), np.sqrt(gqq).mean(0), cos.mean(0)


def stats_of_fit(F):
    """One fit -> the scalar curves the section reports.  All at the settled convention."""
    xbar, theta, A, C = F["xbar"], F["theta"], F["A"], F["C"]
    g = mcv.metric(xbar, theta, A, C)                    # (K,T,2,2), pcubic/0.3/pushforward
    s_tt, s_qq, cos = entries(g)
    thf, radius = mcv.ring(xbar, theta, A, C, n_fine=N_FINE)     # radius (n,T)
    L = np.array([gr.arc_length(radius[:, t], thf)[0] for t in range(radius.shape[1])])
    return dict(L=L.tolist(), s_tt=s_tt.tolist(), s_qq=s_qq.tolist(), cos=cos.tolist())


# resamplings
def boot_index(targ, seed):
    """Trial indices resampled WITH replacement, stratified within reach direction.

    Stratifying keeps all eight directions populated at their original counts, which the
    periodic spline needs to stay well posed.
    """
    rng = np.random.default_rng(seed)
    out = []
    for d in np.unique(targ):
        idx = np.flatnonzero(targ == d)
        out.append(rng.choice(idx, size=len(idx), replace=True))
    return np.concatenate(out)


def unit_halves(M, seed):
    """Two disjoint, uniformly drawn halves of the recorded units."""
    p = np.random.default_rng(seed).permutation(M)
    return p[: M // 2], p[M // 2:]


# runner
def done_keys(path=OUT):
    if not os.path.exists(path):
        return set()
    keys = set()
    with open(path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add((r["mode"], r["rep"], r["sess"], r["region"], r["variant"]))
    return keys


def emit(rec, path=OUT):
    with open(path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def load_subm(sess):
    """Monkey M tensor, loaded as build_subm_fields.py loads it: trials with a finite
    direction label and finite rates, and NO unit-variance floor."""
    z = np.load(f"tensor_{sess}_sub-M.npz")
    out = {}
    for reg in REGIONS:
        y = np.asarray(z["y_" + reg], float)
        targ = np.asarray(z["target_" + reg], float)
        keep = np.isfinite(targ) & np.isfinite(y).all(axis=(1, 2))
        out[reg] = dict(y=y[keep], targ=targ[keep])
    return out


def run(mode, reps, sessions, regions, out=OUT, animal="subC"):
    have = done_keys(out)
    d_region = D_REGION if animal == "subC" else D_REGION_SUBM
    load = td.load_session_tensor if animal == "subC" else load_subm
    t_all = time.time()
    for rep in reps:
        for sess in sessions:
            S = load(sess)
            if S is None:
                print(f"  {sess}: tensor missing, skipped", flush=True)
                continue
            for reg in regions:
                y, targ = S[reg]["y"], S[reg]["targ"]
                D, rf = d_region[reg], mr.RF[reg]

                if mode == "full":
                    jobs = [("full", y, targ)]
                elif mode == "trials":
                    ix = boot_index(targ, SEED_TRIALS + 1000 * rep + hash(sess) % 997)
                    jobs = [("boot", y[ix], targ[ix])]
                elif mode == "units":
                    a, b = unit_halves(y.shape[2], SEED_UNITS + 1000 * rep + hash(sess) % 997)
                    jobs = [("hA", y[:, :, a], targ), ("hB", y[:, :, b], targ)]
                else:
                    raise SystemExit(f"unknown mode {mode}")

                for variant, yy, tt in jobs:
                    key = (mode, rep, sess, reg, variant)
                    if key in have:
                        continue
                    t0 = time.time()
                    F = mc.fit_fields(yy, tt, D, rf, nit=NIT, init="identity")
                    rec = dict(mode=mode, rep=rep, sess=sess, region=reg, variant=variant,
                               D=D, n_trials=int(yy.shape[0]), n_units=int(yy.shape[2]),
                               ll=float(np.asarray(F["lls"])[-1]),
                               secs=round(time.time() - t0, 1),
                               **stats_of_fit(F))
                    emit(rec, out)
                    print(f"  {mode} rep{rep} {sess} {reg:>3} {variant:>4} "
                          f"M={rec['n_units']:3d} N={rec['n_trials']:3d} "
                          f"L {rec['L'][ONSET]:6.2f}->{rec['L'][-1]:6.2f} "
                          f"ratio {rec['L'][-1] / rec['L'][ONSET]:.3f}  {rec['secs']:5.1f}s",
                          flush=True)
        print(f"[rep {rep} done, {(time.time() - t_all) / 60:.1f} min elapsed]", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "trials", "units"], required=True)
    ap.add_argument("--reps", nargs=2, type=int, default=[0, 1],
                    help="replicate range, half open")
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--regions", nargs="*", default=list(REGIONS))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--animal", choices=["subC", "subM"], default="subC")
    a = ap.parse_args()
    default_sess = list(td.SESSIONS) if a.animal == "subC" else SESSIONS_SUBM
    sess = default_sess if a.sessions is None else a.sessions
    out = a.out if a.out != OUT or a.animal == "subC" else "resample43_subm.jsonl"
    run(a.mode, range(*a.reps), sess, a.regions, out, animal=a.animal)
