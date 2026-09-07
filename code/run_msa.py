"""run_msa.py — d_MSA between sessions, between regions and against surrogates.

Computes, over the native 8 directions x 25 time bins grid, both the uniform average of d_SR
and the average volume-weighted by sqrt(det G), together with a scale statistic and the
condition numbers (points above COND_MAX are reported).

Four comparisons: within-session across different inits, between sessions in one region,
between regions in one session, and a direction-shuffled surrogate. Latent dimension is fixed
per region at M1 = 8, PMd = 12.

Usage:  python run_msa.py
Output: msa.json
"""
from __future__ import annotations
import glob, itertools, json, os, time
import numpy as np
from scipy.linalg import eigh
import jax
jax.config.update("jax_enable_x64", True)

import fitting as mr
import lds_em_highD as em
import pullback_metric as pb

DATA = "data/dandi/000688-rich14/sub-C"
SESSIONS = ["20160909", "20160914", "20160919", "20160921", "20160929", "20161013"]
NPZ_FALLBACK = {"20160914": "y_tensors.npz"}
D_REGION = {"M1": 8, "PMd": 12}          # co-smoothing knees
NIT = 80
COND_MAX = 1e8
OUT = "msa.json"


def load_npz(path):
    z = np.load(path)
    return {r: dict(y=np.asarray(z["y_" + r], float), targ=np.asarray(z["target_" + r], float))
            for r in ("M1", "PMd")}


# the metric
def metric_field(y, targ, D, rf, shuffle_seed=None):
    """Neural-space pullback metric g(t,theta): (K,T,2,2), plus the fit used."""
    if shuffle_seed is not None:                       # surrogate: break the direction code
        targ = np.random.default_rng(shuffle_seed).permutation(targ)
    params, _, _ = mr.fit_identity(y, D, rf, NIT)
    A = np.asarray(params[0]); C = np.asarray(params[1])
    xh = np.asarray(em.e_step_batch(y, *params)[0])
    xbar, theta = pb.condition_means(xh, targ)
    g, _, _ = pb.metric(xbar, A, C=C, theta=theta)     # C given -> neural space, gauge-invariant
    return np.asarray(g), theta


# the MSA
def d_sr(G, Gp):
    """Spectral ratio distance, 1 - sqrt(lambda_min/lambda_max) of the pencil (G, Gp)."""
    lam = eigh(G, Gp, eigvals_only=True)
    lam = np.clip(lam, 1e-300, None)
    return float(1.0 - np.sqrt(lam.min() / lam.max()))


def msa(gA, gB):
    """Average d_SR over the (theta, t) grid. Uniform and volume-weighted."""
    K, T = gA.shape[:2]
    d = np.empty((K, T)); w = np.empty((K, T)); cond = np.empty((K, T))
    for k in range(K):
        for t in range(T):
            GA, GB = gA[k, t], gB[k, t]
            d[k, t] = d_sr(GA, GB)
            w[k, t] = np.sqrt(max(np.linalg.det(GA), 0)) + np.sqrt(max(np.linalg.det(GB), 0))
            cond[k, t] = max(np.linalg.cond(GA), np.linalg.cond(GB))
    good = cond < COND_MAX
    return dict(uniform=float(d[good].mean()),
                vol_weighted=float((d[good] * w[good]).sum() / w[good].sum()),
                frac_illcond=float(1 - good.mean()),
                d_by_time=[float(v) for v in d.mean(0)])


def scale_stat(gA, gB):
    """MSA is scale-blind by construction; report the scale difference separately.
    log-ratio of the direction-code length sqrt(g_thth), averaged over the grid."""
    a = np.sqrt(gA[..., 1, 1]); b = np.sqrt(gB[..., 1, 1])
    return float(np.mean(np.log(a / b)))


def main():
    t_all = time.time()
    fields = {}
    for sess in SESSIONS:
        hits = glob.glob(f"{DATA}/sub-C_ses-CO-{sess}_behavior+ecephys.nwb")
        if hits:
            F = mr.preprocess_session(hits[0])
        elif sess in NPZ_FALLBACK and os.path.exists(NPZ_FALLBACK[sess]):
            F = load_npz(NPZ_FALLBACK[sess])
        else:
            print(f"!! no data {sess}", flush=True); continue
        for region in ("M1", "PMd"):
            y = np.asarray(F[region]["y"], float); targ = np.asarray(F[region]["targ"], float)
            D, rf = D_REGION[region], mr.RF[region]
            t0 = time.time()
            g, _ = metric_field(y, targ, D, rf)
            fields[(sess, region)] = g
            # controls computed on the same data
            if sess == SESSIONS[0]:
                fields[(sess, region, "shuf")] = metric_field(y, targ, D, rf, shuffle_seed=1)[0]
            print(f"  {sess} {region} D={D}  metric {g.shape}  [{time.time()-t0:.0f}s]", flush=True)

    out = {"D_region": D_REGION, "pairs": [], "controls": []}

    # between-session, same region
    for region in ("M1", "PMd"):
        ss = [s for s in SESSIONS if (s, region) in fields]
        for a, b in itertools.combinations(ss, 2):
            m = msa(fields[(a, region)], fields[(b, region)])
            m.update(kind="between_session", region=region, a=a, b=b,
                     log_scale_ratio=scale_stat(fields[(a, region)], fields[(b, region)]))
            out["pairs"].append(m)
        vals = [p["uniform"] for p in out["pairs"] if p["region"] == region
                and p["kind"] == "between_session"]
        print(f"[{region}] between-session d_MSA: mean {np.mean(vals):.3f} "
              f"range {min(vals):.3f}-{max(vals):.3f}  (n={len(vals)} pairs)", flush=True)

    # between-region, same session
    for s in SESSIONS:
        if (s, "M1") in fields and (s, "PMd") in fields:
            m = msa(fields[(s, "M1")], fields[(s, "PMd")])
            m.update(kind="between_region", region="M1_vs_PMd", a=s, b=s,
                     log_scale_ratio=scale_stat(fields[(s, "M1")], fields[(s, "PMd")]))
            out["pairs"].append(m)
    vals = [p["uniform"] for p in out["pairs"] if p["kind"] == "between_region"]
    print(f"[M1 vs PMd] within-session d_MSA: mean {np.mean(vals):.3f} "
          f"range {min(vals):.3f}-{max(vals):.3f}", flush=True)

    # surrogate control: direction labels shuffled
    for region in ("M1", "PMd"):
        key = (SESSIONS[0], region, "shuf")
        if key in fields:
            m = msa(fields[(SESSIONS[0], region)], fields[key])
            m.update(kind="shuffled_control", region=region, a=SESSIONS[0], b="shuffled")
            out["controls"].append(m)
            print(f"[{region}] shuffled-direction control d_MSA: {m['uniform']:.3f}", flush=True)

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nWROTE {OUT}  ({time.time()-t_all:.0f}s total)\nDONE", flush=True)


if __name__ == "__main__":
    main()
