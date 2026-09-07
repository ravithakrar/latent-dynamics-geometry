"""run_metric_recovery_multi.py — the recovery validation across all sessions and both regions.

Extends run_metric_recovery.py (single session, 20160914 M1) to the full six-session set in
both regions. Same truth construction, simulation, pipeline and comparison; only the loop is
new, so every function that does real work is imported rather than reimplemented.

Both regions are the point. Every M1 session has D* = 8, so extra M1 sessions repeat the same
test, while PMd's operating dimension is 12. If the recovery-error minimum moves from 8 to 12
when the truth moves from an M1 fit to a PMd fit, the sweep is tracking the true dimension
rather than always returning 8.

The sweep grid is absolute, {4, 6, 8, 12, 20} in both regions, so M1 curves bottom out at 8 and
PMd curves at 12 on a shared x-axis. PMd therefore has only one grid point above its D* = 12,
so its over-fitting arm is sampled more thinly than M1's.

Output: metric_recovery_multi.json
"""
from __future__ import annotations
import argparse, json, time

import numpy as np

import fit_cache
import metric_compare as mc
import tangent_data as td
import run_metric_recovery as R1

SESSIONS = td.SESSIONS
REGIONS = ("M1", "PMd")
FIT_D = [4, 6, 8, 12, 20]
NSEED = 5
OUT = "metric_recovery_multi.json"


def one_truth(session, region, fit_d, nseed, verbose=True):
    """Full sweep for a single (session, region) truth. Mirrors run_metric_recovery.main()."""
    fits = fit_cache.load(R1.CACHE)
    fit = fits[(session, region)]
    F = td.load_session_tensor(session)
    if F is None:
        return None
    y, targ = F[region]["y"], F[region]["targ"]
    rf = fit_cache.RF[region]

    # STALE CACHE GUARD. lds_fits_D8_12.npz predates the flat-unit drop in
    # tangent_data.load_session_tensor, so its PMd emission matrices have 4-13 more rows than the
    # tensors now return; M1 drops none. Refit at the cached D with the current convention
    # rather than loading the old units back in.
    refit = False
    if np.asarray(fit["params"][1]).shape[0] != y.shape[2]:
        refit = True
        Dc = int(fit["D"])
        print(f"    ! cached C has {np.asarray(fit['params'][1]).shape[0]} rows, tensor has "
              f"{y.shape[2]} units -> refitting at D={Dc} with the current drop_flat rule",
              flush=True)
        params, lls, _ = R1.diag.run_track(y, R1.identity_init(y, Dc), R1.NIT, rf)
        fit = dict(params=tuple(np.asarray(p) for p in params), targ=targ,
                   N=y.shape[0], T=y.shape[1], M=y.shape[2], D=Dc,
                   lls=np.asarray(lls))

    truth = R1.build_truth(fit, y, targ)
    Dstar = int(truth["D"])
    Ltrue = R1.dir_length(truth["g_true"])[1]
    out = dict(session=session, region=region, D_true=Dstar,
               M=int(truth["C"].shape[0]), N=int(len(targ)), T=int(truth["T"]),
               var_explained=float(truth["var_explained"]),
               L_ratio_true=float(Ltrue[-1] / Ltrue[R1.ONSET]),
               refit_for_drop_flat=refit, per_fitD={})
    if verbose:
        print(f"  truth {session} {region}: D*={Dstar} M={out['M']} N={out['N']} "
              f"ring explains {100*out['var_explained']:.1f}%  L ratio {out['L_ratio_true']:.3f}",
              flush=True)

    for D in fit_d:
        t0 = time.time()
        runs, ghats = [], []
        for seed in range(nseed):
            ysim = R1.simulate_session(truth, targ, seed=1000 + seed)
            g_hat, ll = R1.fit_and_metric(ysim, targ, D, rf)
            r = R1.compare(truth["g_true"], g_hat)
            runs.append(r)
            ghats.append(g_hat)
        pw = [float(mc.msa(ghats[i], ghats[j], scheme="uniform")["d_msa"])
              for i in range(len(ghats)) for j in range(i + 1, len(ghats))]
        d = np.array([r["d_msa"] for r in runs])
        out["per_fitD"][str(D)] = dict(
            d_msa_mean=float(d.mean()), d_msa_sd=float(d.std(ddof=1)),
            same_truth_mean=float(np.mean(pw)), same_truth_sd=float(np.std(pw, ddof=1)),
            d_shape_mean=float(np.mean([r["d_shape"] for r in runs])),
            log_scale_mean=float(np.mean([r["log_scale"] for r in runs])
                                 if "log_scale" in runs[0] else
                                 np.mean([r["log_scale_mean"] for r in runs])),
            sqrt_g_corr_mean=float(np.mean([r["sqrt_g_corr"] for r in runs])),
            L_ratio_hat_mean=float(np.mean([r["L_ratio_hat"] for r in runs])),
            d_msa_all=[float(v) for v in d], same_truth_all=pw)
        if verbose:
            p = out["per_fitD"][str(D)]
            print(f"    D={D:2d}: vs truth {p['d_msa_mean']:.3f}+-{p['d_msa_sd']:.3f} | "
                  f"same-truth {p['same_truth_mean']:.3f} | "
                  f"logscale {p['log_scale_mean']:+.3f} | "
                  f"L {p['L_ratio_hat_mean']:.3f}  [{time.time()-t0:.0f}s]", flush=True)

    mu = {D: out["per_fitD"][str(D)]["d_msa_mean"] for D in fit_d}
    out["argmin_D"] = int(min(mu, key=mu.get))
    out["hits_true_D"] = bool(out["argmin_D"] == Dstar)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", default=SESSIONS)
    ap.add_argument("--regions", nargs="+", default=list(REGIONS))
    ap.add_argument("--fit-d", type=int, nargs="+", default=FIT_D)
    ap.add_argument("--nseed", type=int, default=NSEED)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    t0 = time.time()
    results = []
    for region in a.regions:
        for session in a.sessions:
            print(f"[{time.time()-t0:6.0f}s] {session} {region}", flush=True)
            r = one_truth(session, region, a.fit_d, a.nseed)
            if r is not None:
                results.append(r)
                json.dump(dict(config=dict(fit_D=a.fit_d, nseed=a.nseed, kmax=R1.KMAX,
                                           nit=R1.NIT, s_noise=R1.S_NOISE, onset_bin=R1.ONSET,
                                           cache=R1.CACHE, grid="absolute",
                                           construction="per-time-bin periodic cubic spline, "
                                                        "metric in neural space"),
                               results=results),
                          open(a.out, "w"), indent=1)

    print(f"\nDONE ({time.time()-t0:.0f}s)  wrote {a.out}")
    for r in results:
        print(f"  {r['session']} {r['region']}: D*={r['D_true']} argmin={r['argmin_D']} "
              f"{'HIT' if r['hits_true_D'] else 'MISS'}")


if __name__ == "__main__":
    main()
