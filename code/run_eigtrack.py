"""run_eigtrack.py — eig(A) at every EM iteration, session 20160914, both initialisations.

`diag.run_track` already records eig(A) per iteration; msa_cache.py throws it away and keeps
only the final A. This re-runs the same two fits that fig_init_floor plots the likelihood of
(identity init and warm start, D = 8 / 12, nit = 80) and keeps the eigenvalue track, so the
spectrum figure and the log-likelihood figure show the same two runs.

One (region, init) per invocation so each call fits inside the device shell's time limit;
results accumulate in eigtrack_20160914.npz.

Usage:  python3 run_eigtrack.py --region M1 --init identity
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import fitting as mr
import tangent_data as td

SESS = "20160914"
D_REGION = {"M1": 8, "PMd": 12}
NIT = 80
OUT = "eigtrack_20160914.npz"

ap = argparse.ArgumentParser()
ap.add_argument("--region", choices=["M1", "PMd"], required=True)
ap.add_argument("--init", choices=["identity", "warmstart"], required=True)
a = ap.parse_args()

store = dict(np.load(OUT)) if os.path.exists(OUT) else {}
key = f"{a.region}|{a.init}"
if key + "|eigs" in store:
    raise SystemExit(f"{key} already cached in {OUT}")

F = td.load_session_tensor(SESS)
y = np.asarray(F[a.region]["y"], float)
D, rf = D_REGION[a.region], mr.RF[a.region]

t0 = time.time()
fit = mr.fit_identity if a.init == "identity" else mr.fit_warmstart
params, lls, eigs = fit(y, D, rf, NIT)
store[key + "|eigs"] = np.asarray(eigs)          # (NIT+1, D) complex
store[key + "|lls"] = np.asarray(lls, float)
store[key + "|meta"] = np.array([y.shape[0], y.shape[2], D], float)
np.savez_compressed(OUT, **store)
print(f"{key}: eigs {eigs.shape}  final max|lam| {np.abs(eigs[-1]).max():.4f}  "
      f"ll {lls[-1]:.1f}  {time.time() - t0:.0f}s", flush=True)
