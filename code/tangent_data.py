"""tangent_data.py — shared helpers for the model-free (data-space) tangent analyses.

Everything here operates on the PREPROCESSED FIRING RATES y (N, T, M) directly.
No LDS, no EM, no Kalman smoother, no latent state.

Provided:
  * load_session_tensor   — the six sub-C session tensors from disk
  * condition_means_y     — trial-average the rates within each reach direction -> (K,T,M)
  * tangent(...)          — periodic akima / cubic / fourier d/dtheta across directions
                            (identical estimators to pullback_metric.py, re-exported so the
                            data-space and latent-space analyses cannot drift apart)
  * split_halves          — deterministic per-direction trial split, for the noise ceiling
  * pca_basis             — orthonormal basis of the condition-mean tensor (descriptive only)
  * ridge_transport       — least-squares/ridge one-step linear map fitted to condition means
  * transport_powers      — A^t applied to a tangent field
"""
from __future__ import annotations
import os
import numpy as np

import pullback_metric as pb   # re-use the exact interpolators used in the latent analysis

SESSIONS = ["20160909", "20160914", "20160919", "20160921", "20160929", "20161013"]
REGIONS = ("M1", "PMd")
#: 20 ms bins over (-0.10, 0.40) s around movement onset; bin 5 == onset
ONSET_BIN = 5
#: file layout: 20160914 lives in y_tensors.npz, the rest in tensor_<date>.npz
TENSOR_PATH = {s: (f"tensor_{s}.npz" if s != "20160914" else "y_tensors.npz") for s in SESSIONS}

#: "pcubic" is "cubic" with a roughness penalty; lam=0 reproduces "cubic" exactly.
INTERPOLATORS = ("akima", "cubic", "fourier", "pcubic")

#: a unit is dropped if its variance is below the region's observation-noise
#: floor (fitting.RF). Matches r_floor so the rule needs no new constant.
UNIT_VAR_FLOOR = {"M1": 0.01, "PMd": 0.05}


# loading
def load_session_tensor(sess, root=".", drop_flat=True):
    """-> {region: dict(y=(N,T,M), targ=(N,))}, non-finite direction labels dropped."""
    path = os.path.join(root, TENSOR_PATH[sess])
    if not os.path.exists(path):
        return None
    z = np.load(path)
    out = {}
    for r in REGIONS:
        y = np.asarray(z["y_" + r], float)
        targ = np.asarray(z["target_" + r], float)
        keep = np.isfinite(targ) & np.isfinite(y).all(axis=(1, 2))
        y, targ = y[keep], targ[keep]
        if drop_flat:
            # Drop units whose activity varies less than the observation-noise
            # floor for this region. Such a unit carries nothing the Gaussian
            # emission can represent, and an exactly constant one makes the
            # likelihood at the PCA initialisation unbounded above.
            # M1 loses none; PMd loses 4-13 per session.
            v = y.reshape(-1, y.shape[2]).var(0)
            live = v >= UNIT_VAR_FLOOR[r]
            y = y[:, :, live]
        out[r] = dict(y=y, targ=targ)
    return out


def condition_means_y(y, targ):
    """Trial-average the rates within each reach direction.
    y (N,T,M), targ (N,) -> ybar (K,T,M), theta (K,), counts (K,)."""
    dirs = np.unique(targ)
    ybar = np.stack([y[targ == d].mean(0) for d in dirs])
    counts = np.array([int((targ == d).sum()) for d in dirs])
    return ybar, dirs, counts


def split_halves(targ, seed=0):
    """Deterministic within-direction split into two disjoint trial sets.
    Returns two boolean masks over trials, balanced per direction."""
    rng = np.random.default_rng(seed)
    a = np.zeros(len(targ), bool)
    for d in np.unique(targ):
        idx = np.flatnonzero(targ == d)
        rng.shuffle(idx)
        a[idx[: len(idx) // 2]] = True
    return a, ~a


# interpolants
def tangent(ybar, theta, method="cubic", lam=0.0):
    """d ybar / d theta across reach directions, periodic. (K,T,F) -> (K,T,F).

    Identical code path as the latent analysis: pullback_metric.direction_tangent.
      akima   — nonlinear in the data, does NOT commute with a linear map
      cubic   — linear in the sampled values, commutes exactly
      fourier — band-limited trigonometric interpolant, commutes exactly
      pcubic  — cubic plus a roughness penalty `lam`; still linear, so still commutes.
                lam=0 reproduces cubic to machine precision.
    """
    return pb.direction_tangent(ybar, theta, method=method, lam=lam)


# cosines
def cos_KT(a, b, eps=1e-30):
    """Cosine per (direction, time) between two (K,T,F) fields."""
    num = np.sum(a * b, axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + eps
    return num / den


# descriptive linear transport
def pca_basis(ybar, d):
    """Orthonormal basis (d, M) of the condition-mean tensor, mean removed.

    Purely descriptive: an SVD of the (K*T, M) condition-mean matrix. Because the
    basis is ORTHONORMAL, cosines computed in these coordinates equal cosines in
    neural space, so the projection changes nothing about the reported quantity —
    it only conditions the regression below.
    Returns (U, mu) with U (d,M), mu (M,).
    """
    X = ybar.reshape(-1, ybar.shape[-1])
    mu = X.mean(0)
    Vt = np.linalg.svd(X - mu, full_matrices=False)[2]
    return Vt[:d], mu


def ridge_transport(z, lam=1e-3):
    """One-step linear map fitted by ridge regression to trajectories in `z`.

    z (n_traj, T, d) — either condition-mean trajectories (n_traj = K) or single-trial
    trajectories (n_traj = N). Fits A minimising
        sum_{i,t} || z(i,t+1) - A z(i,t) ||^2 + lam * ||A||_F^2 * scale
    with `scale` = mean squared magnitude of the regressors, so lam is dimensionless.
    Returns A (d,d).
    """
    X = z[:, :-1, :].reshape(-1, z.shape[-1])          # (n*(T-1), d)
    Y = z[:, 1:, :].reshape(-1, z.shape[-1])
    G = X.T @ X
    scale = np.trace(G) / max(X.shape[1], 1)
    A = np.linalg.solve(G + lam * scale * np.eye(X.shape[1]), X.T @ Y).T
    return A


def transport_powers(v0, A, T):
    """A^t v0 for t = 0..T-1.  v0 (K,d) -> (K,T,d)."""
    out = np.empty((v0.shape[0], T, A.shape[0]))
    cur = np.asarray(v0, float).copy()
    for t in range(T):
        out[:, t, :] = cur
        cur = cur @ A.T
    return out
