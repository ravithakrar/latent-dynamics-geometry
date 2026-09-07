"""Diagnostics for the analyses #5, #6, #7 on the rich session
(sub-C_ses-CO-20160914, M1=95, PMd=258). Reuses the project's own EM
(lds_em_highD) and regularisation (neural_fit.run_em_reg) exactly.
"""
from __future__ import annotations
import functools
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import lds_em_highD as em
from neural_pca_init import pca_init

Q_FLOOR = 1e-4

# one regularised EM step (matches neural_fit.run_em_reg), JIT-compiled
@functools.partial(jax.jit, static_argnums=())
def _em_step(ys, A, C, Q, R, mu0, V0, r_floor):
    xhat, P, Pcross, loglik = em.e_step_batch(ys, A, C, Q, R, mu0, V0)
    ll = jnp.sum(loglik)
    A, C, Q, R, mu0, V0 = em.m_step(ys, xhat, P, Pcross)
    D = A.shape[0]
    Id = jnp.eye(D)
    Q = 0.5 * (Q + Q.T) + Q_FLOOR * Id
    R = jnp.diag(jnp.clip(jnp.diag(R), r_floor, None))
    V0 = 0.5 * (V0 + V0.T) + Q_FLOOR * Id
    return (A, C, Q, R, mu0, V0), ll


def run_track(ys, init, n_iters, r_floor):
    """EM that records LL and eig(A) at *every* iteration (pre-update A)."""
    ys = jnp.asarray(ys)
    params = tuple(jnp.asarray(p) for p in init)
    rf = jnp.asarray(float(r_floor))
    lls, eigs = [], []
    for _ in range(n_iters):
        eigs.append(np.linalg.eigvals(np.asarray(params[0])))
        params, ll = _em_step(ys, *params, rf)
        lls.append(float(ll))
    # final-state eig
    eigs.append(np.linalg.eigvals(np.asarray(params[0])))
    return params, np.asarray(lls), np.asarray(eigs)  # eigs: (n_iters+1, D)


def score_ll(ys, params):
    """Total one-step predictive LL over a batch, using given params."""
    ys = jnp.asarray(ys)
    return float(jnp.sum(em.e_step_batch(ys, *params)[3]))


def smoothed_latents(ys, params):
    return np.asarray(em.e_step_batch(jnp.asarray(ys), *params)[0])  # (N,T,D)


# random / perturbed initialisations for multi-start #6
def perturbed_init(y, d, seed, a_noise=0.03, c_noise=0.02):
    A, C, Q, R, mu0, V0 = pca_init(y, d=d)
    rng = np.random.default_rng(seed)
    A = np.asarray(A) + a_noise * rng.standard_normal((d, d))
    C = np.asarray(C) + c_noise * rng.standard_normal(np.asarray(C).shape)
    mu0 = np.asarray(mu0) + 0.1 * rng.standard_normal(d)
    to = lambda x: jnp.asarray(np.asarray(x), dtype=jnp.float64)
    return (to(A), to(C), to(Q), to(R), to(mu0), to(V0))


def random_init(y, d, seed, rho=0.9):
    """Fully random init: random stable A, random orthonormal C, identity noises."""
    N, T, M = y.shape
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    A = rho * A / (np.max(np.abs(np.linalg.eigvals(A))) + 1e-9)  # spectral radius rho
    C = np.linalg.qr(rng.standard_normal((M, d)))[0]             # orthonormal columns
    Q = 0.1 * np.eye(d); R = np.eye(M) * float(np.var(y));
    mu0 = np.zeros(d); V0 = np.eye(d)
    to = lambda x: jnp.asarray(np.asarray(x), dtype=jnp.float64)
    return (to(A), to(C), to(Q), to(R), to(mu0), to(V0))


# structured inits that fix high-D multimodality (see init bake-off)
from sklearn.decomposition import FactorAnalysis

def fa_init(y, d, jitter=1e-4, seed=0):
    """Factor-analysis init: separates shared vs private variance, so R starts
    diagonal (matches the model). Beats PCA at high D."""
    N, T, M = y.shape
    Y = y.reshape(N * T, M)
    fa = FactorAnalysis(n_components=d, max_iter=500, random_state=seed).fit(Y)
    C = fa.components_.T
    x = fa.transform(Y).reshape(N, T, d)
    Xp = x[:, :-1, :].reshape(-1, d); Xn = x[:, 1:, :].reshape(-1, d)
    A = np.linalg.solve(Xp.T @ Xp + jitter * np.eye(d), Xp.T @ Xn).T
    Q = (Xn - Xp @ A.T).T @ (Xn - Xp @ A.T) / max(len(Xn), 1) + jitter * np.eye(d)
    R = np.diag(fa.noise_variance_ + jitter)
    x1 = x[:, 0, :]; mu0 = x1.mean(0); V0 = np.cov(x1, rowvar=False) + jitter * np.eye(d)
    to = lambda a: jnp.asarray(np.asarray(a), dtype=jnp.float64)
    return tuple(to(v) for v in (A, C, Q, R, mu0, V0))

def _expand_init(y, prev, d2):
    A1, C1, Q1, R1, mu1, V1 = [np.asarray(p) for p in prev]
    d1 = A1.shape[0]; N, T, M = y.shape
    mu = y.reshape(N * T, M).mean(0, keepdims=True)
    x1 = (y.reshape(N * T, M) - mu) @ C1
    resid = (y.reshape(N * T, M) - mu) - x1 @ C1.T
    cov = resid.T @ resid / max(len(resid) - 1, 1)
    evec = np.linalg.eigh(cov)[1]; U = evec[:, ::-1][:, :(d2 - d1)]
    C2 = np.concatenate([C1, U], axis=1)
    A2 = np.zeros((d2, d2)); A2[:d1, :d1] = A1
    for i in range(d1, d2): A2[i, i] = 0.9
    Q2 = np.eye(d2) * 1e-2; Q2[:d1, :d1] = Q1
    mu2 = np.zeros(d2); mu2[:d1] = mu1
    V2 = np.eye(d2); V2[:d1, :d1] = V1
    to = lambda a: jnp.asarray(np.asarray(a), dtype=jnp.float64)
    return tuple(to(v) for v in (A2, C2, Q2, R1, mu2, V2))

def warmstart_fit(y, d_target, r_floor, schedule=(10, 20, 35, 50), n_iters=80):
    """Dimension annealing: fit at increasing D, embedding each solution into the
    next. Most reliable route to the best basin at high D; also stays monotone."""
    ds = [d for d in schedule if d < d_target] + [d_target]
    params = None
    for dk in ds:
        init = pca_init(y, d=dk) if params is None else _expand_init(y, params, dk)
        params, _, _ = run_track(y, init, n_iters, r_floor)
    return params
