"""run_msa_direct.py — the same 91 session pairs under the DIRECT tangent construction.

Everything in section 4.4 is pushforward: the all-pairs run stored d_MSA under
pcubic_lam0.3_push and cubic_lam0_push, which differ in lambda and not in construction. The
6-session run found that the LEVEL of d_MSA is robust to the construction but the calendar TREND
is not (direct gave M1 p = 0.050 and PMd p = 0.099 against 0.039 and 0.044), so the slope has to
be checked before it is quoted.

No refits: the store holds xbar / A / C / theta per fit, so the direct field is rebuilt from what
is already there. Same aggregation as run_msa_dist_allpairs.report().

Usage:  python3 run_msa_direct.py --region M1 --mode units
Writes msa_direct_<region>.json, merged over calls.
"""
from __future__ import annotations

import argparse
import glob
import itertools as it
import json
import os
import time

import numpy as np

import metric_compare as met
import metric_convention as mc

SESSIONS = ["20160909", "20160912", "20160914", "20160915", "20160919", "20160921",
            "20160923", "20160929", "20161005", "20161006", "20161007", "20161011",
            "20161013", "20161021"]
TAG = {"M1": "", "PMd": "PMd_", "PMdD8": "PMdD8_"}
N_SEEDS = 50


def days_apart(a, b):
    import datetime as dt
    f = lambda s: dt.date(int(s[:4]), int(s[4:6]), int(s[6:]))
    return abs((f(a) - f(b)).days)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="M1", choices=list(TAG))
    ap.add_argument("--mode", default="units", choices=["trials", "units"])
    a = ap.parse_args()
    out_path = f"msa_direct_{a.region}.json"
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    if a.mode in out:
        raise SystemExit(f"{a.mode} already in {out_path}")

    store = {}
    for f in sorted(glob.glob(f"msa_allpairs_fields_{TAG[a.region]}{a.mode}_p*.npz")):
        z = np.load(f)
        store.update({k: z[k] for k in z.files})

    t0 = time.time()
    g = {}
    for seed in range(N_SEEDS):
        for sess in SESSIONS:
            for h in (1, 2):
                k = f"{a.mode}|{seed}|{sess}|h{h}"
                if k + "|xbar" in store:
                    g[(seed, sess, h)] = mc.metric(store[k + "|xbar"], store[k + "|theta"],
                                                   store[k + "|A"], store[k + "|C"],
                                                   push=False)
    print(f"{len(g)} direct fields built in {time.time() - t0:.0f}s", flush=True)

    seeds = sorted({s for s, _, _ in g})
    within = {s: [float(met.msa(g[(sd, s, 1)], g[(sd, s, 2)])["d_msa"]) for sd in seeds]
              for s in SESSIONS}
    pairs = []
    for p, q in it.combinations(SESSIONS, 2):
        v = [float(met.msa(g[(sd, p, i)], g[(sd, q, j)])["d_msa"])
             for sd in seeds for i in (1, 2) for j in (1, 2)]
        pairs.append(dict(a=p, b=q, days=days_apart(p, q), between_mean=float(np.mean(v)),
                          within_mean=float(np.mean(within[p] + within[q]))))
    out[a.mode] = dict(pairs=pairs, n_seeds=len(seeds))
    json.dump(out, open(out_path, "w"), indent=1)

    d = np.array([q["days"] for q in pairs], float)
    b = np.array([q["between_mean"] for q in pairs])
    w = np.array([q["within_mean"] for q in pairs])
    A = np.vstack([np.ones_like(d), d]).T
    beta = np.linalg.lstsq(A, b - w, rcond=None)[0]
    r = (b - w) - A @ beta
    se = np.sqrt(r @ r / (len(d) - 2) * np.linalg.inv(A.T @ A)[1, 1])
    print(f"{a.region} {a.mode} DIRECT: between {b.mean():.4f}  within {w.mean():.4f}  "
          f"excess {np.mean(b - w):+.4f}  slope {10 * beta[1]:+.4f} +- {10 * se:.4f} /10 d  "
          f"cross {-beta[0] / beta[1]:.1f} d  [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
