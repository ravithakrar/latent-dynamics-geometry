"""PCA-based initialisation of the LDS parameters for EM.

EM finds a local optimum of a non-convex likelihood, so the initialisation matters.
The emission  y_t = C x_t + noise  places the population activity near a
d-dimensional linear subspace, and PCA returns that subspace. PCA supplies C and a
first guess of the latents; the dynamics and the noise sizes follow in closed form,
and EM refines the set.

Recipe (matches the (A, C, Q, R, mu0, V0) layout of lds_em_highD):
    1. stack every time bin, PCA -> top-d directions U.  C_init = U   (M, d)
       latents  x_t = U^T y_t                                        (N, T, d)
    2. dynamics A: least-squares regress x_{t+1} on x_t   (within trials)
    3. Q, R: covariances of the dynamics / emission residuals
    4. mu0, V0: mean/cov of the first-bin latents x_1 across trials

Conventions (same as lds_em_highD / simulate_highD):
    N = trials, T = bins, M = neurons (observation dim), D = latent dim.
    C : (M, D)   A, Q, V0 : (D, D)   R : (M, M)   mu0 : (D,)
"""

from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


def pca_spectrum(ys: np.ndarray) -> np.ndarray:
    """Eigenvalue spectrum (variance per PC, descending) for a scree plot.

    Args:
        ys: (N, T, M) observations.
    Returns:
        (M,) eigenvalues of the neuron covariance, largest first.
    """
    Y = np.asarray(ys).reshape(-1, ys.shape[-1])
    Yc = Y - Y.mean(axis=0, keepdims=True)
    cov = (Yc.T @ Yc) / max(Yc.shape[0] - 1, 1)
    evals = np.linalg.eigvalsh(cov)[::-1]
    return np.clip(evals, 0.0, None)


def variance_explained(ys: np.ndarray, d: int) -> float:
    """Fraction of total variance captured by the top-d PCs."""
    evals = pca_spectrum(ys)
    return float(evals[:d].sum() / evals.sum())


def pca_init(ys: np.ndarray, d: int, jitter: float = 1e-4):
    """Initialise (A, C, Q, R, mu0, V0) from PCA + closed-form residual fits.

    Args:
        ys:     (N, T, M) preprocessed observations.
        d:      latent dimension D.
        jitter: small value added to covariance diagonals for numerical PD-ness.

    Returns:
        (A, C, Q, R, mu0, V0) as float64 jax arrays, ready for lds_em_highD.run_em.
    """
    ys = np.asarray(ys, dtype=float)
    N, T, M = ys.shape
    if d > M:
        raise ValueError(f"latent dim d={d} cannot exceed n_units M={M}")

    # 1. PCA: top-d directions U and the projected latents
    Y = ys.reshape(N * T, M)
    mu = Y.mean(axis=0, keepdims=True)
    Yc = Y - mu
    cov = (Yc.T @ Yc) / max(Yc.shape[0] - 1, 1)          # (M, M)
    evals, evecs = np.linalg.eigh(cov)                    # ascending
    U = evecs[:, ::-1][:, :d]                             # (M, d) top-d loadings
    C = U

    x = (ys - mu.reshape(1, 1, M)) @ U                    # (N, T, d) latents

    # 2. dynamics A: regress x_{t+1} on x_t (respect trial boundaries)
    Xp = x[:, :-1, :].reshape(-1, d)                      # x_t
    Xn = x[:, 1:, :].reshape(-1, d)                       # x_{t+1}
    Spp = Xp.T @ Xp + jitter * np.eye(d)
    A = np.linalg.solve(Spp, Xp.T @ Xn).T                 # (d, d)

    # 3. residual covariances Q (dynamics) and R (emission, diagonal)
    dyn_res = Xn - Xp @ A.T
    Q = (dyn_res.T @ dyn_res) / max(dyn_res.shape[0], 1) + jitter * np.eye(d)

    emis_res = (ys - mu.reshape(1, 1, M)) - x @ C.T       # (N, T, M)
    emis_res = emis_res.reshape(-1, M)
    R = np.diag(emis_res.var(axis=0) + jitter)            # (M, M) diagonal

    # 4. initial-state distribution from the first bin of each trial
    x1 = x[:, 0, :]                                       # (N, d)
    mu0 = x1.mean(axis=0)                                 # (d,)
    V0 = np.cov(x1, rowvar=False)
    V0 = np.atleast_2d(V0) + jitter * np.eye(d)           # (d, d)

    to_j = lambda a: jnp.asarray(a, dtype=jnp.float64)
    return (to_j(A), to_j(C), to_j(Q), to_j(R), to_j(mu0), to_j(V0))


def identity_init(ys: np.ndarray, d: int, jitter: float = 1e-4):
    """Identity-dynamics initialisation: A = I (pure integrator).

    Identical PCA subspace (C), emission noise (R) and initial-state (mu0, V0)
    as `pca_init`, but the dynamics start at the identity matrix instead of the
    1-step least-squares regressor. Starting at A = I means the state is, at
    init, a pure integrator with no deterministic drift, so EM has to *learn*
    the deviation A - I -- which is exactly the object of interest for the Schur
    / non-normality analysis.

    Q is initialised from the covariance of the latent first differences
    x_{t+1} - x_t, i.e. the one-step prediction residual implied by A = I
    (whereas pca_init uses the residual of the fitted regressor).

    Args:
        ys:     (N, T, M) preprocessed observations.
        d:      latent dimension D.
        jitter: small value added to covariance diagonals for numerical PD-ness.

    Returns:
        (A, C, Q, R, mu0, V0) as float64 jax arrays, ready for lds_em_highD.run_em.
    """
    # reuse PCA for the spatial half (C), emission noise (R), initial state (mu0, V0)
    _A_ls, C, _Q_ls, R, mu0, V0 = pca_init(ys, d, jitter)

    ys = np.asarray(ys, dtype=float)
    N, T, M = ys.shape
    mu = ys.reshape(N * T, M).mean(axis=0, keepdims=True)
    x = ((ys.reshape(N * T, M) - mu) @ np.asarray(C)).reshape(N, T, d)   # same latents as pca_init

    # A = I ; Q from first-difference residuals (the one-step residual when A = I)
    A = np.eye(d)
    dx = (x[:, 1:, :] - x[:, :-1, :]).reshape(-1, d)
    Q = dx.T @ dx / max(dx.shape[0], 1) + jitter * np.eye(d)

    to_j = lambda a: jnp.asarray(np.asarray(a), dtype=jnp.float64)
    return (to_j(A), to_j(C), to_j(Q), to_j(R), to_j(mu0), to_j(V0))
