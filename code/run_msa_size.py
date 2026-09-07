"""run_msa_size.py — does the SIZE of the metric drift across sessions?

d_SR is scale-blind: two metric fields differing by an overall magnification are at distance
zero (eq. dsr). So d_MSA answers only whether the SHAPE of the metric is stable, and the size
has to be carried separately. This script does that on exactly the same 91 session pairs and the
same two nulls as run_msa_dist_allpairs.py.

Size of one fit:      s = mean over the cylinder of  log sqrt(g_theta theta)
Mismatch of a pair:   |s - s'|,  a dimensionless log ratio, so it is comparable between
                      regions fitted at different D.

Everything is recomputed from the stored xbar / A / C / theta at the settled convention
(metric_convention: pcubic, lambda = 0.3, pushforward). No EM fits.

Usage:  python3 run_msa_size.py --region M1 [--mode units]
Writes msa_size_<region>.json, merged over calls.
"""
from __future__ import annotations

import argparse
import glob
import itertools as it
import json
import os

import numpy as np

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


def log_size(z, k):
    """mean log sqrt(g_thth) over the cylinder, and the same PER UNIT.

    Ring size is a length in the space of that day's recorded units, so it grows with how many
    the array yielded (r = 0.92 with unit count, section 4.3). Two sessions with different array
    sizes therefore differ in size for a reason that has nothing to do with the geometry, while
    two disjoint halves of one session have the same unit count by construction and carry no
    such term. Subtracting 1/2 log M removes it, since the metric is a sum of M squared
    projections and so scales as sqrt(M).
    """
    C = z[k + "|C"]
    g = mc.metric(z[k + "|xbar"], z[k + "|theta"], z[k + "|A"], C)
    s = float(np.mean(0.5 * np.log(np.maximum(g[..., 1, 1], 1e-300))))
    return s, s - 0.5 * np.log(C.shape[0])


def summarise(s):
    """s: (seed, sess, half) -> log size. Returns the per-pair between/within mismatch."""
    within = {sess: [abs(s[(sd, sess, 1)] - s[(sd, sess, 2)])
                     for sd in range(N_SEEDS) if (sd, sess, 1) in s]
              for sess in SESSIONS}
    pairs = []
    for p, q in it.combinations(SESSIONS, 2):
        v = [abs(s[(sd, p, i)] - s[(sd, q, j)])
             for sd in range(N_SEEDS) for i in (1, 2) for j in (1, 2)
             if (sd, p, i) in s and (sd, q, j) in s]
        if not v:
            continue
        pairs.append(dict(a=p, b=q, days=days_apart(p, q),
                          between_mean=float(np.mean(v)),
                          within_mean=float(np.mean(within[p] + within[q]))))
    return dict(pairs=pairs,
                within={k: float(np.mean(v)) for k, v in within.items() if v})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="M1", choices=list(TAG))
    ap.add_argument("--mode", default=None, choices=["trials", "units"])
    a = ap.parse_args()
    out_path = f"msa_size_{a.region}.json"
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}

    for mode in ([a.mode] if a.mode else ["trials", "units"]):
        if mode in out:
            print(f"{mode}: cached, skipped", flush=True)
            continue
        store = {}
        for f in sorted(glob.glob(f"msa_allpairs_fields_{TAG[a.region]}{mode}_p*.npz")):
            z = np.load(f)
            store.update({k: z[k] for k in z.files})
        s = {}
        for seed in range(N_SEEDS):
            for sess in SESSIONS:
                for h in (1, 2):
                    k = f"{mode}|{seed}|{sess}|h{h}"
                    if k + "|xbar" in store:
                        s[(seed, sess, h)] = log_size(store, k)
        print(f"{mode}: {len(s)} fits sized", flush=True)

        for j, name in enumerate(["raw", "per_unit"]):
            sj = {k: v[j] for k, v in s.items()}
            out.setdefault(mode, {})[name] = summarise(sj)
        json.dump(out, open(out_path, "w"), indent=1)
        for name in ("raw", "per_unit"):
            r = out[mode][name]
            d = np.array([q["days"] for q in r["pairs"]], float)
            b = np.array([q["between_mean"] for q in r["pairs"]])
            w = np.array([q["within_mean"] for q in r["pairs"]])
            A = np.vstack([np.ones_like(d), d]).T
            beta = np.linalg.lstsq(A, b - w, rcond=None)[0]
            print(f"  {a.region} {mode} {name:9s}: between {b.mean():.4f}  "
                  f"within {w.mean():.4f}  excess {np.mean(b - w):+.4f}  "
                  f"slope {10 * beta[1]:+.5f}/10 d", flush=True)
        continue

        within = {sess: [abs(s[(sd, sess, 1)] - s[(sd, sess, 2)])
                         for sd in range(N_SEEDS) if (sd, sess, 1) in s]
                  for sess in SESSIONS}
        pairs = []
        for p, q in it.combinations(SESSIONS, 2):
            v = [abs(s[(sd, p, i)] - s[(sd, q, j)])
                 for sd in range(N_SEEDS) for i in (1, 2) for j in (1, 2)
                 if (sd, p, i) in s and (sd, q, j) in s]
            if not v:
                continue
            pairs.append(dict(a=p, b=q, days=days_apart(p, q),
                              between_mean=float(np.mean(v)),
                              within_mean=float(np.mean(within[p] + within[q]))))
        out[mode] = dict(pairs=pairs,
                         within={k: float(np.mean(v)) for k, v in within.items() if v})
        json.dump(out, open(out_path, "w"), indent=1)
        d = np.array([p["days"] for p in pairs], float)
        b = np.array([p["between_mean"] for p in pairs])
        w = np.array([p["within_mean"] for p in pairs])
        A = np.vstack([np.ones_like(d), d]).T
        beta = np.linalg.lstsq(A, b - w, rcond=None)[0]
        print(f"  {a.region} {mode}: between {b.mean():.4f}  within {w.mean():.4f}  "
              f"excess {np.mean(b - w):+.4f}  slope {10 * beta[1]:+.5f}/10 d", flush=True)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
