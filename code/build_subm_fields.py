"""build_subm_fields.py — the sub-M analogue of msa_fields.npz, for the cross-animal panel.

Seven sessions of monkey M spanning 2014-02-03 to 2014-03-07, 33 days, chosen to match the
34-day span of the six monkey-C sessions used everywhere else.

D = 4 (M1) and 6 (PMd), which is where co-smoothing puts the knee for this animal rather than
the 8/12 used for monkey C. The reason is the array: sub-M yields 26 to 52 M1 units and 66 to
121 PMd units, against 55 to 95 and 114 to 258 in sub-C, so fewer latent dimensions are
supported. Fits are otherwise identical to msa_cache.py: identity initialisation, nit = 80,
the same per-region R floor, and the same stratified trial halves for the error bar.

Stores xbar, theta, A and C per session/region/variant, so g and g_push are recomputed from
the cache without refitting.

Usage: python3 build_subm_fields.py [--out subm_fields.npz]
"""
from __future__ import annotations
import argparse, json, time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import diag
import lds_em_highD as em
import fitting as mr
import pullback_metric as pb
import tangent_data as td
from neural_pca_init import identity_init

SESSIONS = ["20140203", "20140217", "20140218", "20140303",
            "20140304", "20140306", "20140307"]
D_REGION = {"M1": 4, "PMd": 6}          # sub-M co-smoothing knee, not sub-C's 8/12
NIT, SPLIT_SEED = 80, 0
_estep = jax.jit(em.e_step_batch)


def fit_one(y, targ, D, rf):
    params, lls, _ = diag.run_track(y, identity_init(y, D), NIT, rf)
    xh = np.asarray(_estep(*[jax.numpy.asarray(np.asarray(p)) for p in (y, *params)])[0])
    xbar, theta = pb.condition_means(xh, targ)
    A, C = np.asarray(params[0]), np.asarray(params[1])
    return dict(xbar=xbar, theta=theta, A=A, C=C,
                meta=np.array([y.shape[0], y.shape[2], D], float),
                ll=float(np.asarray(lls)[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="subm_fields.npz")
    a = ap.parse_args()
    store, entries, t0 = {}, [], time.time()
    for sess in SESSIONS:
        z = np.load(f"tensor_{sess}_sub-M.npz")
        for region in ("M1", "PMd"):
            y = np.asarray(z["y_" + region], float)
            targ = np.asarray(z["target_" + region], float)
            keep = np.isfinite(targ) & np.isfinite(y).all(axis=(1, 2))
            y, targ = y[keep], targ[keep]
            D, rf = D_REGION[region], mr.RF[region]
            m1, m2 = td.split_halves(targ, seed=SPLIT_SEED)
            for variant, yy, tt in (("full", y, targ), ("h1", y[m1], targ[m1]),
                                    ("h2", y[m2], targ[m2])):
                f = fit_one(yy, tt, D, rf)
                for k, v in f.items():
                    if k != "ll":
                        store[f"{sess}|{region}|{variant}|{k}"] = v
                entries.append(dict(session=sess, region=region, variant=variant, D=D,
                                    n_trials=int(yy.shape[0]), M_units=int(yy.shape[2]),
                                    ll_final=f["ll"]))
                print(f"  {sess} {region:3s} {variant:4s} D={D} N={yy.shape[0]:3d} "
                      f"M={yy.shape[2]:3d}  [{time.time()-t0:.0f}s]", flush=True)
        np.savez_compressed(a.out, **store)           # checkpoint per session
    np.savez_compressed(a.out, **store)
    json.dump(dict(subject="sub-M", sessions=SESSIONS, D_region=D_REGION, nit=NIT,
                   split_seed=SPLIT_SEED, init="identity", entries=entries),
              open(a.out.replace(".npz", "_meta.json"), "w"), indent=1)
    print(f"WROTE {a.out} ({len(store)} arrays) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
