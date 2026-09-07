"""msa_cache.py — fit once, cache the metric fields, so re-analysis needs no refitting.

Does all the EM fitting in one pass and writes every field to msa_fields.npz.

Stores, per session and region: full (identity init, all trials), h1/h2 (identity init on a
stratified within-direction trial split), alt (warm start on all trials) and shuf (direction
labels permuted). For each of those: A, C, g, g_push, g_lat, xbar, theta, lls, n, M.

  g        the neural-space metric, C applied
  g_lat    its latent-space counterpart
  g_push   the same metric with the direction tangent estimated once at t = 0 and carried
           forward by A^t rather than refitted by spline at each time bin. Both are always
           stored; --push only sets which one g aliases
  xbar     the condition-mean latent trajectory, from which g can be rebuilt with a different
           interpolator or lambda without refitting

D, EM iterations, the interpolator and both RNG seeds are written to msa_fields_meta.json. The
interpolator is passed explicitly rather than taken from pullback_metric.DEFAULT_METHOD.

Usage
    python msa_cache.py                 # everything, -> msa_fields.npz   (~13 min)
    python msa_cache.py --no-alt-init   # skip the different-init fits    (~10 min)
    python msa_cache.py --sessions 20160914 20160919
    python msa_cache.py --D-M1 5 --D-PMd 5 --out msa_fields_D5.npz

Then:
    from msa_cache import load
    F = load()                              # {(sess, region, variant): {...}}
    F[("20160914", "M1", "full")]["g"]      # (K, T, 2, 2)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import fitting as mr
import lds_em_highD as em
import pullback_metric as pb
import tangent_data as td

# defaults
#: co-smoothing knees, fixed per region so that
#: cross-session comparisons are like-for-like.
D_REGION = {"M1": 8, "PMd": 12}
NIT = 80
#: passed explicitly to pb.metric rather than inheriting pullback_metric.DEFAULT_METHOD
METHOD = "cubic"
SPLIT_SEED = 0      # td.split_halves — matches run_msa_floor.py's intent
SHUF_SEED = 1       # direction-label permutation — matches run_msa.py
OUT = "msa_fields.npz"

WEIGHTED = False   # True: condition-split loss, w_n = N/(K N_k). See lds_em_weighted.
LAM = 0.0          # roughness penalty; only meaningful for method="pcubic"
PUSH = False       # False: spline refitted at every time bin. True: fitted at t=0, A^t forward.

VARIANTS = ("full", "h1", "h2", "alt", "shuf")
FIELDS = ("A", "C", "g", "g_push", "g_lat", "xbar", "theta", "lls", "meta")


# fits
def fit_fields(y, targ, D, rf, nit=NIT, method=METHOD, init="identity",
               lam=LAM, push=PUSH, weighted=WEIGHTED):
    """One EM fit -> both metric fields plus the parameters they came from.

    init: "identity" (mr.fit_identity) or "warmstart" (mr.fit_warmstart).
    lam:  dimensionless roughness penalty, method="pcubic" only.
    push: estimate the direction tangent once at t=0 and push it forward by A^t
          (pb.metric_pushforward) instead of refitting the spline at every time bin.
    Returns a dict of plain numpy arrays.
    """
    if weighted:
        # condition-split loss: each reach direction contributes 1/K to the objective
        # regardless of its trial count. Reduces to the unweighted fit exactly when the
        # counts are balanced (lds_em_weighted.selftest).
        if init != "identity":
            raise NotImplementedError("weighted fits are identity-init only")
        import lds_em_weighted as lw
        params, lls, _ = lw.fit_identity_weighted(y, targ, D, rf, nit)
    else:
        fit = mr.fit_identity if init == "identity" else mr.fit_warmstart
        params, lls, _ = fit(y, D, rf, nit)

    A = np.asarray(params[0])
    C = np.asarray(params[1])
    xh = np.asarray(em.e_step_batch(y, *params)[0])
    xbar, theta = pb.condition_means(xh, targ)

    # Both constructions are always stored, in neural space (C applied):
    #   g       direction tangent refitted by spline at every time bin   ("direct")
    #   g_push  tangent estimated once at t=0 and carried by A^t         ("pushforward")
    # `push` selects which one the alias `g` points at; both arrays are present either way
    # and the choice is recorded in the meta json.
    g_direct, _, dth_x = pb.metric(xbar, A, C=C, theta=theta, method=method, lam=lam)
    g_push, _, _ = pb.metric_pushforward(xbar, A, C=C, theta=theta, method=method, lam=lam)
    g = g_push if push else g_direct
    # latent space — reuse the SAME tangent so the two differ only by C
    g_lat, _, _ = pb.metric(xbar, A, C=None, dth_x=dth_x, theta=theta)

    return dict(
        A=A,
        C=C,
        g=np.asarray(g, float),
        g_push=np.asarray(g_push, float),
        g_lat=np.asarray(g_lat, float),
        # condition-mean latents. Stored so that re-estimating the direction tangent —
        # a different interpolator, or a lambda sweep — costs no EM fits at all. Without
        # it every lambda on the grid is a full rebuild, which is what made the smoothing
        # spline work look expensive.
        xbar=np.asarray(xbar, float),
        theta=np.asarray(theta, float),
        lls=np.asarray(lls, float),
        # n trials, M units, D latents  — needed for the unit-count confound (item 4)
        meta=np.array([y.shape[0], y.shape[2], D], float),
    )


def shuffle_targ(targ, seed=SHUF_SEED):
    """Permute direction labels -> destroys the direction code, keeps everything else."""
    return np.random.default_rng(seed).permutation(np.asarray(targ))


# build
def build(out=OUT, sessions=None, regions=None, d_region=None, nit=NIT,
          method=METHOD, alt_init=True, root=".", resume=False, lam=LAM, push=PUSH,
          weighted=WEIGHTED):
    sessions = list(td.SESSIONS) if sessions is None else list(sessions)
    regions = list(td.REGIONS) if regions is None else list(regions)
    d_region = dict(D_REGION if d_region is None else d_region)

    store: dict[str, np.ndarray] = {}
    provenance = dict(D_region=d_region, nit=nit, method=method, lam=lam, push=push,
                      weighted=weighted,
                      split_seed=SPLIT_SEED, shuffle_seed=SHUF_SEED,
                      alt_init="warmstart" if alt_init else None,
                      sessions=sessions, regions=regions, entries=[])

    # resume: keep fields already computed, but ONLY if the config matches
    if resume and os.path.exists(out):
        prev = np.load(out)
        store = {k: prev[k] for k in prev.files}
        meta_path = out.replace(".npz", "_meta.json")
        if os.path.exists(meta_path):
            old = json.load(open(meta_path))
            clash = {k: (old.get(k), provenance[k])
                     for k in ("D_region", "nit", "method", "lam", "push", "weighted",
                               "split_seed", "shuffle_seed")
                     if old.get(k) != provenance[k]}
            if clash:
                raise SystemExit(
                    f"refusing to resume {out}: config differs from the cached run\n  "
                    + "\n  ".join(f"{k}: cached={a!r} now={b!r}" for k, (a, b) in clash.items())
                    + "\nUse a different --out, or delete the file to rebuild from scratch.")
            # carry forward the entries already recorded, dropping any that are recomputed
            provenance["entries"] = list(old.get("entries", []))
            provenance["sessions"] = sorted(set(old.get("sessions", [])) | set(sessions))
        done = {tuple(k.split("|")[:3]) for k in store}
        print(f"resuming {out}: {len(done)} fields already cached\n", flush=True)

    t_all = time.time()

    for sess in sessions:
        F = td.load_session_tensor(sess, root=root)
        if F is None:
            print(f"!! no tensor for {sess} "
                  f"(expected {td.TENSOR_PATH[sess]}) — skipped", flush=True)
            continue

        for region in regions:
            y = np.asarray(F[region]["y"], float)
            targ = np.asarray(F[region]["targ"], float)
            D, rf = d_region[region], mr.RF[region]
            m1, m2 = td.split_halves(targ, seed=SPLIT_SEED)

            jobs = [("full", y, targ, "identity"),
                    ("h1", y[m1], targ[m1], "identity"),
                    ("h2", y[m2], targ[m2], "identity")]
            if alt_init:
                jobs.append(("alt", y, targ, "warmstart"))
            # shuffled control on the canonical first session only — td.SESSIONS[0], not
            # sessions[0], so that chunked/resumed runs do not each add their own control
            if sess == td.SESSIONS[0]:
                jobs.append(("shuf", y, shuffle_targ(targ), "identity"))

            for variant, yy, tt, init in jobs:
                key = f"{sess}|{region}|{variant}"
                if f"{key}|g" in store:
                    print(f"  {sess} {region:<3} {variant:<4} cached, skipped", flush=True)
                    continue
                t0 = time.time()
                try:
                    d = fit_fields(yy, tt, D, rf, nit=nit, method=method, init=init,
                                   lam=lam, push=push, weighted=weighted)
                except Exception as exc:                      # keep going, record the gap
                    print(f"  !! {sess} {region} {variant}: {type(exc).__name__}: {exc}",
                          flush=True)
                    continue
                for name, arr in d.items():
                    store[f"{key}|{name}"] = arr
                provenance["entries"].append(
                    dict(session=sess, region=region, variant=variant, init=init,
                         D=D, r_floor=rf, n_trials=int(yy.shape[0]),
                         M_units=int(yy.shape[2]), ll_final=float(d["lls"][-1]),
                         eig_max=float(np.abs(np.linalg.eigvals(d["A"])).max()),
                         seconds=round(time.time() - t0, 1)))
                print(f"  {sess} {region:<3} {variant:<4} D={D:<3} "
                      f"n={yy.shape[0]:<4} M={yy.shape[2]:<4} "
                      f"g{d['g'].shape}  |eig|max {provenance['entries'][-1]['eig_max']:.4f}"
                      f"  [{time.time() - t0:.0f}s]", flush=True)

            np.savez_compressed(out, **store)                 # checkpoint per region

    np.savez_compressed(out, **store)
    # derive the session/region inventory from what is actually IN the cache, not from
    # this invocation's arguments — otherwise a chunked --resume run records only the
    # last chunk, and downstream consumers silently iterate a subset.
    keys = {tuple(k.split("|")[:3]) for k in store}
    provenance["sessions"] = sorted({s for s, _, _ in keys})
    provenance["regions"] = sorted({r for _, r, _ in keys})
    provenance["n_fields"] = len(keys)
    meta_path = out.replace(".npz", "_meta.json")
    json.dump(provenance, open(meta_path, "w"), indent=1)

    print(f"\nWROTE {out}  ({len(provenance['entries'])} fields, "
          f"{time.time() - t_all:.0f}s total)", flush=True)
    print(f"WROTE {meta_path}", flush=True)
    return out


# load
def load(path=OUT, construction=None):
    """-> {(session, region, variant): {A, C, g, g_direct, g_push, g_lat, theta, ...}}.

    CONSTRUCTION. Two metrics are stored for every entry and they are NOT interchangeable:

        g_direct   direction tangent refitted by spline at every time bin
        g_push     tangent estimated once at t=0 and carried by A^t

    On disk the key `g` is always the direct one. What `d["g"]` means in the RETURNED dict is
    set by the cache's declared default (`default_construction` in the meta json) or by this
    argument, and both arrays are always present under their own names so a caller can be
    explicit. The declared default is **push**.

    The two differ: d_SR barely, since it is scale-invariant (M1 floor 0.273 -> 0.282), but
    L(t) by about 18%, since that is a scale. Ratios are preserved. Pass
    construction="direct" for the other one.
    """
    z = np.load(path)
    out: dict[tuple[str, str, str], dict] = {}
    for k in z.files:
        sess, region, variant, name = k.split("|")
        d = out.setdefault((sess, region, variant), {})
        d[name] = z[k]

    if construction is None:
        try:
            construction = meta(path).get("default_construction", "direct")
        except FileNotFoundError:
            construction = "direct"
    if construction not in ("direct", "push"):
        raise ValueError(f"construction must be 'direct' or 'push', got {construction!r}")

    for d in out.values():
        if "meta" in d:
            n, M, D = d.pop("meta")
            d.update(n=int(n), M=int(M), D=int(D))
        if "g" in d:
            d["g_direct"] = d["g"]                       # on-disk `g` is always direct
        if "g_push" in d:
            d["g"] = d["g_push"] if construction == "push" else d["g_direct"]
        elif construction == "push":
            raise KeyError(f"{path} predates g_push; run migrate_cache_gpush.py")
        d["construction"] = construction
    return out


def meta(path=OUT):
    return json.load(open(path.replace(".npz", "_meta.json")))


# main
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=OUT)
    p.add_argument("--root", default=".", help="directory holding the tensor_*.npz files")
    p.add_argument("--sessions", nargs="*", default=None)
    p.add_argument("--regions", nargs="*", default=None)
    p.add_argument("--D-M1", type=int, default=D_REGION["M1"])
    p.add_argument("--D-PMd", type=int, default=D_REGION["PMd"])
    p.add_argument("--nit", type=int, default=NIT)
    p.add_argument("--method", default=METHOD, choices=list(td.INTERPOLATORS))
    p.add_argument("--lam", type=float, default=LAM,
                   help="dimensionless roughness penalty; requires --method pcubic")
    p.add_argument("--weighted", action="store_true",
                   help="condition-split loss: weight each trial by N/(K N_k) so every reach "
                        "direction contributes equally to the fit regardless of trial count")
    p.add_argument("--push", action="store_true",
                   help="estimate the direction tangent at t=0 and push forward by A^t "
                        "instead of refitting the spline at every time bin")
    p.add_argument("--no-alt-init", action="store_true",
                   help="skip the warm-start fits (drops the different-init floor)")
    p.add_argument("--resume", action="store_true",
                   help="keep fields already in --out and only fit what is missing; "
                        "refuses if D/nit/method/seeds differ from the cached run")
    a = p.parse_args()

    d_region = {"M1": a.D_M1, "PMd": a.D_PMd}
    if a.lam and a.method != "pcubic":
        raise SystemExit(f"--lam {a.lam} requires --method pcubic (got {a.method!r})")
    print(f"msa_cache -> {a.out}\n  D={d_region}  nit={a.nit}  method={a.method}  "
          f"lam={a.lam}  push={a.push}  weighted={a.weighted}  "
          f"alt_init={not a.no_alt_init}\n", flush=True)
    build(out=a.out, sessions=a.sessions, regions=a.regions, d_region=d_region,
          nit=a.nit, method=a.method, alt_init=not a.no_alt_init, root=a.root,
          resume=a.resume, lam=a.lam, push=a.push, weighted=a.weighted)


if __name__ == "__main__":
    main()
