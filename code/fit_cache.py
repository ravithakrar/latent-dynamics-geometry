"""fit_cache.py — fit the identity-init LDS once per session x region and cache it.

Same fit as fitting.fit_identity:
identity initialisation, operating D (M1 60, PMd 80, capped at M-5), 80 EM iterations,
per-region R floor. Cached so the simulation study does not refit twelve times.

Usage: python3 fit_cache.py [outfile.npz]
"""
from __future__ import annotations
import sys, time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import diag
from neural_pca_init import identity_init
import tangent_data as td

D_OP = {"M1": 60, "PMd": 80}          # legacy default; override on the command line
RF = {"M1": 0.01, "PMd": 0.05}
NIT = 80
OUT = "lds_fits.npz"


def fit_all(out=OUT, d_op=None):
    d_op = D_OP if d_op is None else d_op
    store = {}
    for sess in td.SESSIONS:
        F = td.load_session_tensor(sess)
        if F is None:
            print(f"!! no tensor for {sess}", flush=True); continue
        for region in td.REGIONS:
            y, targ = F[region]["y"], F[region]["targ"]
            N, T, M = y.shape
            D = min(d_op[region], M - 5)
            t0 = time.time()
            params, lls, _ = diag.run_track(y, identity_init(y, D), NIT, RF[region])
            key = f"{sess}_{region}"
            for nm, p in zip(("A", "C", "Q", "R", "mu0", "V0"), params):
                store[f"{key}_{nm}"] = np.asarray(p)
            store[f"{key}_lls"] = np.asarray(lls)
            store[f"{key}_targ"] = np.asarray(targ)
            store[f"{key}_meta"] = np.array([N, T, M, D], float)
            print(f"  {key}: D={D} M={M} N={N}  LL {lls[-1]:.1f}  "
                  f"|eig|max {np.abs(np.linalg.eigvals(np.asarray(params[0]))).max():.4f}"
                  f"  [{time.time()-t0:.0f}s]", flush=True)
            np.savez_compressed(out, **store)
    np.savez_compressed(out, **store)
    print("WROTE", out, flush=True)


def load(path=OUT):
    z = np.load(path)
    fits = {}
    for k in z.files:
        if k.endswith("_meta"):
            key = k[:-5]
            sess, region = key.rsplit("_", 1)
            N, T, M, D = z[k]
            fits[(sess, region)] = dict(
                params=tuple(z[f"{key}_{nm}"] for nm in ("A", "C", "Q", "R", "mu0", "V0")),
                targ=z[f"{key}_targ"], N=int(N), T=int(T), M=int(M), D=int(D),
                lls=z[f"{key}_lls"])
    return fits


if __name__ == "__main__":
    # usage: python3 fit_cache.py [out.npz] [D_M1 D_PMd]
    out = sys.argv[1] if len(sys.argv) > 1 else OUT
    dop = ({"M1": int(sys.argv[2]), "PMd": int(sys.argv[3])} if len(sys.argv) > 3 else None)
    print(f"fitting -> {out}  D = {dop or D_OP}", flush=True)
    fit_all(out, dop)
