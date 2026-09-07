"""run_metric_recovery.py — recovery of a known metric from simulated data.

Builds a ground-truth metric field, simulates observations from it, refits with the pipeline
used on the recordings, and compares the recovered field with the true one.

Truth is the 20160914 M1 identity-init fit at D = 8 (lds_fits_D8_12.npz). The ring
xbar(0, theta) is a kmax = 3 trig series fitted to that session's condition-mean initial
latents, so xbar(t) = A^t xbar(0) and its theta-derivative are closed form and no spline enters
the truth. The estimate refits by EM from the identity init at nit = 80 and then follows the
real-data path. Fit D is swept over {4, 6, 8, 12, 20} against the true D = 8, NSEED seeds each.

Also computes two reference scales under the same construction: the d_MSA between two fits of
the same data from different inits, and the d_MSA between the six sessions.

Usage:  python3 run_metric_recovery.py [--nseed 10] [--fit-d 4 6 8 12 20]
Output: metric_recovery.json, figures/fig_metric_recovery.png
"""
from __future__ import annotations
import argparse, json, os, time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import diag
import fit_cache
import lds_em_highD as em
import metric_compare as mc
import pullback_metric as pb
import tangent_data as td
import plotstyle as ps
from neural_pca_init import identity_init

SESSION, REGION = "20160914", "M1"
CACHE   = "lds_fits_D8_12.npz"
FIELDS  = "msa_fields.npz"          # cached xbar/theta/A/C for all 6 sessions, D=8/12
KMAX    = 3
NIT     = 80
S_NOISE = 1.0                       # 1.0 == recorded noise level
ONSET   = 5
N_FLOOR = 5                         # perturbed-init refits of the REAL data
OUT_JSON, OUT_FIG = "metric_recovery.json", "figures/fig_metric_recovery.png"

_estep = jax.jit(em.e_step_batch)


def _j(*p):
    import jax.numpy as jnp
    return [jnp.asarray(np.asarray(x)) for x in p]


def estep(y, params):
    return np.asarray(_estep(*_j(y, *params))[0])


# ground truth
def trig_design(theta, kmax, deriv=False):
    """a_0 + sum_j a_j cos(j th) + b_j sin(j th), or its d/dtheta. (K, 2*kmax+1)."""
    theta = np.asarray(theta, float)
    cols = [np.zeros_like(theta) if deriv else np.ones_like(theta)]
    for j in range(1, kmax + 1):
        cols += ([-j * np.sin(j * theta), j * np.cos(j * theta)] if deriv
                 else [np.cos(j * theta), np.sin(j * theta)])
    return np.stack(cols, axis=1)


def build_truth(fit, y, targ, kmax=KMAX):
    """Closed-form ring, its exact d_theta, and the true metric field g*."""
    A, C, Q, R, mu0, V0 = (np.asarray(p) for p in fit["params"])
    T, D = fit["T"], fit["D"]
    xh = estep(y, (A, C, Q, R, mu0, V0))
    theta = np.unique(targ)
    x0 = xh[:, 0, :]
    m0 = np.stack([x0[targ == d].mean(0) for d in theta])

    P = trig_design(theta, kmax)
    coef, *_ = np.linalg.lstsq(P, m0, rcond=None)
    x0_true = P @ coef
    d0_true = trig_design(theta, kmax, deriv=True) @ coef
    ve = 1.0 - ((m0 - x0_true) ** 2).sum() / ((m0 - m0.mean(0)) ** 2).sum()

    # residual initial-state covariance: the trial-to-trial spread the simulation must have
    idx = np.searchsorted(theta, targ)
    V_res = np.cov((x0 - m0[idx]).T) + 1e-9 * np.eye(D)

    xbar_true = td.transport_powers(x0_true, A, T)          # (K,T,D), exact
    dfield_true = td.transport_powers(d0_true, A, T)        # (K,T,D), exact
    g_true, _, _ = pb.metric(xbar_true, A, C=C, dth_x=dfield_true)
    return dict(A=A, C=C, Q=Q, R=R, mu0=mu0, V0=V0, T=T, D=D, theta=theta,
                x0_true=x0_true, d0_true=d0_true, V_res=V_res, g_true=g_true,
                var_explained=float(ve))


# simulation
def simulate(x0_by_trial, A, C, Q, R, T, s, rng):
    N, D = x0_by_trial.shape
    Lq = np.linalg.cholesky(Q + 1e-12 * np.eye(D))
    LR = np.linalg.cholesky(R + 1e-12 * np.eye(C.shape[0]))
    Y = np.empty((N, T, C.shape[0]))
    x = x0_by_trial.copy()
    for t in range(T):
        Y[:, t, :] = x @ C.T + s * (rng.standard_normal((N, C.shape[0])) @ LR.T)
        x = x @ A.T + s * (rng.standard_normal((N, D)) @ Lq.T)
    return Y


def simulate_session(truth, targ, seed, s=S_NOISE):
    """One synthetic recording with the real per-direction trial counts."""
    rng = np.random.default_rng(seed)
    theta = truth["theta"]
    idx = np.searchsorted(theta, targ)
    Lr = np.linalg.cholesky(truth["V_res"])
    x0 = truth["x0_true"][idx] + s * (rng.standard_normal((len(idx), truth["D"])) @ Lr.T)
    return simulate(x0, truth["A"], truth["C"], truth["Q"], truth["R"], truth["T"], s, rng)


# estimate
def metric_from_fit(y, targ, params):
    """The REAL pipeline: E-step -> condition means -> periodic cubic -> g in neural space."""
    A, C = np.asarray(params[0]), np.asarray(params[1])
    xh = estep(y, params)
    xbar, theta = pb.condition_means(xh, targ)
    g, _, _ = pb.metric(xbar, A, C=C, theta=theta)
    return g


def fit_and_metric(y, targ, D, rf, init=None, nit=NIT):
    init = identity_init(y, D) if init is None else init
    params, lls, _ = diag.run_track(y, init, nit, rf)
    return metric_from_fit(y, targ, params), float(np.asarray(lls)[-1])


# compare
def dir_length(g):
    """sqrt(g_thth) (K,T) and the ring circumference at the 8 knots, L(t) = mean * 2pi."""
    r = np.sqrt(np.maximum(g[..., 1, 1], 0.0))
    return r, r.mean(0) * 2 * np.pi


def compare(g_true, g_hat):
    r_t, L_t = dir_length(g_true)
    r_h, L_h = dir_length(g_hat)
    m = mc.msa(g_true, g_hat, scheme="uniform")
    return dict(
        d_msa=float(m["d_msa"]), d_shape=float(m["d_shape"]),
        log_scale_mean=float(m["log_scale_mean"]),
        frac_points_used=float(m["frac_points_used"]),
        sqrt_g_corr=float(np.corrcoef(r_t.ravel(), r_h.ravel())[0, 1]),
        sqrt_g_relerr=float(np.mean(np.abs(r_h - r_t) / (r_t + 1e-30))),
        L_ratio_true=float(L_t[-1] / L_t[ONSET]),
        L_ratio_hat=float(L_h[-1] / L_h[ONSET]),
        L_true=[float(v) for v in L_t], L_hat=[float(v) for v in L_h],
        sqrt_g_true_onset=[float(v) for v in r_t[:, ONSET]],
        sqrt_g_hat_onset=[float(v) for v in r_h[:, ONSET]],
    )


# reference scales
def between_session_scale(region=REGION, fields=FIELDS):
    """d_MSA between the six sessions' metrics, SAME construction, no refits."""
    z = np.load(fields)
    sess = sorted({k.split("|")[0] for k in z.files if k.endswith("|full|xbar")})
    G = {}
    for s in sess:
        k = f"{s}|{region}|full"
        g, _, _ = pb.metric(z[k + "|xbar"], z[k + "|A"], C=z[k + "|C"], theta=z[k + "|theta"])
        G[s] = g
    vals = [float(mc.msa(G[a], G[b], scheme="uniform")["d_msa"])
            for i, a in enumerate(sess) for b in sess[i + 1:]]
    return sess, vals


def init_floor(y, targ, D, rf, n=N_FLOOR):
    """d_MSA between the identity-init fit of the REAL data and n perturbed-init refits."""
    g_id, _ = fit_and_metric(y, targ, D, rf)
    out = []
    for s in range(n):
        g_alt, _ = fit_and_metric(y, targ, D, rf, init=diag.perturbed_init(y, D, seed=s))
        out.append(float(mc.msa(g_id, g_alt, scheme="uniform")["d_msa"]))
    return out


# figure
def make_figure(R, fn=OUT_FIG):
    ps.apply()
    fig, axs = plt.subplots(1, 3, figsize=(ps.WIDTH, 2.95))
    dstar = R["config"]["D_true"]
    ref = R["per_fitD"][str(dstar)]["examples"]
    theta = np.array(R["truth"]["theta"])

    ax = axs[0]
    ax.plot(theta, ref["sqrt_g_true_onset"], "o-", color=ps.INK, lw=2, label="true")
    ax.plot(theta, ref["sqrt_g_hat_onset"], "s--", color=ps.M1, lw=1.8, label="recovered")
    ax.set(title=r"$\sqrt{g_{\theta\theta}}$ at onset, $D=8$",
           xlabel=r"reach direction $\theta$ (rad)", ylabel=r"$\sqrt{g_{\theta\theta}}$")
    ax.grid(alpha=.7); ax.set_axisbelow(True); ax.legend(fontsize=6.5)

    ax = axs[1]
    Lt = np.array(ref["L_true"]); Lh = np.array(R["per_fitD"][str(dstar)]["L_hat_mean"])
    Ls = np.array(R["per_fitD"][str(dstar)]["L_hat_sd"])
    tt = np.arange(len(Lt))
    ax.plot(tt, Lt, color=ps.INK, lw=2.2, label="true")
    ax.plot(tt, Lh, color=ps.M1, lw=1.8, ls="--", label="recovered")
    ax.fill_between(tt, Lh - Ls, Lh + Ls, color=ps.M1, alpha=.18, lw=0)
    ax.axvline(ONSET, color=ps.MUTED, ls=":", lw=.9)
    ax.set(title="Ring circumference", xlabel="time bin (20 ms)", ylabel="$L(t)$")
    ax.grid(alpha=.7); ax.set_axisbelow(True); ax.legend(fontsize=8.5)

    ax = axs[2]
    Ds = R["config"]["fit_D"]
    mu = [R["per_fitD"][str(d)]["d_msa_mean"] for d in Ds]
    sd = [R["per_fitD"][str(d)]["d_msa_sd"] for d in Ds]
    st = [R["per_fitD"][str(d)]["same_truth_mean"] for d in Ds]
    stsd = [R["per_fitD"][str(d)]["same_truth_sd"] for d in Ds]
    ax.errorbar(range(len(Ds)), mu, yerr=sd, fmt="o-", color=ps.M1, lw=1.8, capsize=3,
                label="vs truth")
    ax.errorbar(range(len(Ds)), st, yerr=stsd, fmt="^--", color="#0f766e", lw=1.6, capsize=3,
                label="same truth, new data")
    fl = R["reference"]["init_floor_mean"]; bt = R["reference"]["between_median"]
    ax.axhline(fl, color=ps.MUTED, ls="--", lw=1.2)
    ax.text(0.02, fl, f" init floor {fl:.3f}", fontsize=6.5, color=ps.MUTED,
            va="bottom", transform=ax.get_yaxis_transform())
    ax.axhline(bt, color=ps.PMD, ls="-.", lw=1.4)
    ax.text(0.02, bt, f" between sessions {bt:.3f}", fontsize=6.5, color=ps.PMD,
            va="bottom", transform=ax.get_yaxis_transform())
    ax.axvline(Ds.index(dstar), color=ps.GRID, lw=6, zorder=0)
    ax.set_xticks(range(len(Ds))); ax.set_xticklabels([str(d) for d in Ds])
    ax.set(title=f"Recovery error vs fitted $D$",
           xlabel="latent dimension used for the fit", ylabel="$d_{MSA}$")
    ax.grid(alpha=.7); ax.set_axisbelow(True); ax.legend(fontsize=6.5, loc="upper left")

    fig.suptitle(rf"Recovery of a known metric: truth from {SESSION} {REGION}, "
                 rf"$D^*={dstar}$, $k_{{\max}}={KMAX}$, {R['config']['nseed']} seeds")
    fig.tight_layout(pad=.5, rect=(0, 0, 1, .94))
    ps.save(fig, os.path.splitext(fn)[0])


# main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nseed", type=int, default=10)
    ap.add_argument("--fit-d", type=int, nargs="+", default=[4, 6, 8, 12, 20])
    ap.add_argument("--out", default=OUT_JSON)
    ap.add_argument("--plot-only", action="store_true",
                    help="re-render the figure from an existing JSON, no fits")
    a = ap.parse_args()
    if a.plot_only:                     # restyling should never cost 55 EM fits
        make_figure(json.load(open(a.out)))
        return
    t0 = time.time()

    fits = fit_cache.load(CACHE)
    fit = fits[(SESSION, REGION)]
    F = td.load_session_tensor(SESSION)
    y, targ = F[REGION]["y"], F[REGION]["targ"]
    rf = fit_cache.RF[REGION]
    truth = build_truth(fit, y, targ)
    Dstar = truth["D"]
    print(f"truth: {SESSION} {REGION} D*={Dstar} T={truth['T']} N={len(targ)} "
          f"M={truth['C'].shape[0]} | kmax={KMAX} ring explains "
          f"{100*truth['var_explained']:.1f}% of the condition-mean initial latents",
          flush=True)

    # reference scales
    sess, between = between_session_scale()
    print(f"between-session d_MSA ({len(between)} pairs of {len(sess)} sessions): "
          f"median {np.median(between):.3f}, range {min(between):.3f}-{max(between):.3f}",
          flush=True)
    floor = init_floor(y, targ, Dstar, rf)
    print(f"same-data different-init floor: mean {np.mean(floor):.4f} "
          f"({', '.join(f'{v:.4f}' for v in floor)})  [{time.time()-t0:.0f}s]", flush=True)

    # the sweep
    per = {}
    for D in a.fit_d:
        runs, ghats = [], []
        for seed in range(a.nseed):
            ysim = simulate_session(truth, targ, seed=1000 + seed)
            g_hat, ll = fit_and_metric(ysim, targ, D, rf)
            r = compare(truth["g_true"], g_hat); r["ll"] = ll; r["seed"] = seed
            runs.append(r); ghats.append(g_hat)
        # SAME TRUTH, DIFFERENT DATA. Two estimates of an identical geometry, each from its own
        # synthetic recording at the real trial count. This is the proper null for a
        # between-session comparison: the thesis compares two ESTIMATES, so both carry this
        # error, and any observed between-session distance must clear this to mean anything.
        pw = [float(mc.msa(ghats[i], ghats[j], scheme="uniform")["d_msa"])
              for i in range(len(ghats)) for j in range(i + 1, len(ghats))]
        d = np.array([r["d_msa"] for r in runs])
        Lh = np.array([r["L_hat"] for r in runs])
        per[str(D)] = dict(
            runs=runs, d_msa_mean=float(d.mean()), d_msa_sd=float(d.std(ddof=1)),
            same_truth_pairwise=pw, same_truth_mean=float(np.mean(pw)),
            same_truth_sd=float(np.std(pw, ddof=1)),
            d_shape_mean=float(np.mean([r["d_shape"] for r in runs])),
            log_scale_mean=float(np.mean([r["log_scale_mean"] for r in runs])),
            sqrt_g_corr_mean=float(np.mean([r["sqrt_g_corr"] for r in runs])),
            sqrt_g_relerr_mean=float(np.mean([r["sqrt_g_relerr"] for r in runs])),
            L_ratio_hat_mean=float(np.mean([r["L_ratio_hat"] for r in runs])),
            L_hat_mean=[float(v) for v in Lh.mean(0)],
            L_hat_sd=[float(v) for v in Lh.std(0, ddof=1)],
            examples=runs[0])
        print(f"  fit D={D:2d}: d_MSA {d.mean():.4f} +- {d.std(ddof=1):.4f} | "
              f"same-truth {np.mean(pw):.4f} | "
              f"corr(sqrt g) {per[str(D)]['sqrt_g_corr_mean']:.3f} | "
              f"L end/onset true {runs[0]['L_ratio_true']:.3f} vs "
              f"recovered {per[str(D)]['L_ratio_hat_mean']:.3f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    R = dict(
        config=dict(session=SESSION, region=REGION, D_true=int(Dstar), fit_D=list(a.fit_d),
                    kmax=KMAX, nseed=a.nseed, nit=NIT, s_noise=S_NOISE, onset_bin=ONSET,
                    r_floor=rf, cache=CACHE, fields=FIELDS,
                    construction="per-time-bin periodic cubic spline, metric in neural space"),
        truth=dict(theta=[float(v) for v in truth["theta"]],
                   var_explained=truth["var_explained"],
                   L_ratio=float(dir_length(truth["g_true"])[1][-1]
                                 / dir_length(truth["g_true"])[1][ONSET])),
        reference=dict(between_sessions=between, between_median=float(np.median(between)),
                       between_sessions_list=sess,
                       init_floor=floor, init_floor_mean=float(np.mean(floor))),
        per_fitD=per)
    json.dump(R, open(a.out, "w"), indent=1)
    print(f"WROTE {a.out}", flush=True)
    make_figure(R)
    print(f"DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
