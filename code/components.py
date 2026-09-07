"""components.py — save the FULL recovered metric field at D* so all three components of g
(neural speed g_tt, direction length g_thth, alignment cos alpha) can be compared to truth.
The sweep scripts keep only summary scalars, so this re-runs the D* arm and stores the fields.
"""
from __future__ import annotations
import argparse
import numpy as np

# fit_cache / run_metric_recovery pull in jax, which is needed only to RE-RUN the recovery.
# `comps` below is plain numpy and is what the plotting scripts import, so the heavy imports
# are deferred into run(): drawing the figure from the cached fields needs no jax.


def comps(g):
    """-> sqrt(g_tt), sqrt(g_thth), cos alpha, each (K,T)."""
    gtt = np.sqrt(np.maximum(g[..., 0, 0], 0))
    gthth = np.sqrt(np.maximum(g[..., 1, 1], 0))
    cosa = g[..., 0, 1] / np.maximum(gtt * gthth, 1e-30)
    return gtt, gthth, cosa


def run(session, region, nseed, D=None):
    import fit_cache, tangent_data as td, run_metric_recovery as R1
    fit = fit_cache.load(R1.CACHE)[(session, region)]
    F = td.load_session_tensor(session)
    y, targ = F[region]["y"], F[region]["targ"]
    rf = fit_cache.RF[region]
    if np.asarray(fit["params"][1]).shape[0] != y.shape[2]:
        Dc = int(fit["D"])
        params, lls, _ = R1.diag.run_track(y, R1.identity_init(y, Dc), R1.NIT, rf)
        fit = dict(params=tuple(np.asarray(p) for p in params), targ=targ, N=y.shape[0],
                   T=y.shape[1], M=y.shape[2], D=Dc, lls=np.asarray(lls))
    truth = R1.build_truth(fit, y, targ)
    D = int(truth["D"]) if D is None else D
    ghats = []
    for s in range(nseed):
        ysim = R1.simulate_session(truth, targ, seed=1000 + s)
        g, _ = R1.fit_and_metric(ysim, targ, D, rf)
        ghats.append(g)
    return truth, np.stack(ghats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="20160914")
    ap.add_argument("--region", default="M1")
    ap.add_argument("--nseed", type=int, default=5)
    a = ap.parse_args()
    truth, G = run(a.session, a.region, a.nseed)
    np.savez_compressed(f"components_{a.session}_{a.region}.npz",
                        g_true=truth["g_true"], g_hat=G, theta=truth["theta"])
    tt, th, ca = comps(truth["g_true"])
    ht, hh, hc = [np.stack(v) for v in zip(*[comps(g) for g in G])]
    print(f"{a.session} {a.region} D*={truth['D']}  K,T = {tt.shape}")
    print("\n  sqrt(g_tt)   true mean over dirs, first/last:  %.3f -> %.3f" % (tt.mean(0)[0], tt.mean(0)[-1]))
    print("               recovered:                          %.3f -> %.3f" % (ht.mean((0,1))[0], ht.mean((0,1))[-1]))
    print("               corr(true, recovered) over (K,T):   %.3f" % np.corrcoef(tt.ravel(), ht.mean(0).ravel())[0,1])
    print("\n  sqrt(g_thth) true:                              %.3f -> %.3f" % (th.mean(0)[0], th.mean(0)[-1]))
    print("               recovered:                          %.3f -> %.3f" % (hh.mean((0,1))[0], hh.mean((0,1))[-1]))
    print("               corr:                               %.3f" % np.corrcoef(th.ravel(), hh.mean(0).ravel())[0,1])
    print("\n  cos alpha    true  mean over dirs, first/last:  %+.3f -> %+.3f" % (ca.mean(0)[0], ca.mean(0)[-1]))
    print("               recov mean:                        %+.3f -> %+.3f" % (hc.mean((0,1))[0], hc.mean((0,1))[-1]))
    print("               true  RMS over dirs, first/last:    %.3f -> %.3f" % (np.sqrt((ca**2).mean(0))[0], np.sqrt((ca**2).mean(0))[-1]))
    print("               recov RMS:                          %.3f -> %.3f" % (np.sqrt((hc**2).mean(1)).mean(0)[0], np.sqrt((hc**2).mean(1)).mean(0)[-1]))
    print("               POINTWISE corr(true, recovered):    %.3f" % np.corrcoef(ca.ravel(), hc.mean(0).ravel())[0,1])
    print("               range of true cos alpha:            %+.3f .. %+.3f" % (ca.min(), ca.max()))
