"""lds_em_weighted.py — EM with per-trial weights.

lds_em_highD.m_step sums its sufficient statistics over trials, weighting each trial equally.
This module weights each trial by

    w_n = N / (K N_k(n))

so that every reach direction contributes 1/K of the objective whatever its trial count. The
weights sum to N, which keeps the 1/(N T) normalisers in the M-step correct and makes w_n == 1
for every trial when the counts are balanced, so this reduces exactly to the unweighted EM;
selftest() checks that.

run_track_weighted returns the weighted log-likelihood, which is the quantity that increases
monotonically. The E-step is imported from lds_em_highD unchanged.
"""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp

import lds_em_highD as em
from diag import Q_FLOOR


def condition_weights(targ):
    """w_n = N / (K N_k), summing to N; identically 1 when the counts are balanced."""
    targ = np.asarray(targ)
    dirs, inv = np.unique(targ, return_inverse=True)
    counts = np.bincount(inv, minlength=dirs.size).astype(float)
    N, K = targ.size, dirs.size
    return N / (K * counts[inv])


def m_step_weighted(ys, xhat, P, Pcross, w):
    """`lds_em_highD.m_step` with every trial-sum weighted by w (which must sum to N)."""
    N, T, M = ys.shape
    solve = jnp.linalg.solve

    sum_cross = jnp.einsum("n,ntij->ij", w, Pcross)
    sum_prev = jnp.einsum("n,ntij->ij", w, P[:, :-1])
    A_new = solve(sum_prev, sum_cross.T).T

    sum_P = jnp.einsum("n,ntij->ij", w, P)
    C_new = solve(sum_P, jnp.einsum("n,nti,ntj->ij", w, ys, xhat).T).T

    R_new = (1.0 / (N * T)) * (
        jnp.einsum("n,nti,ntj->ij", w, ys, ys)
        - C_new @ jnp.einsum("n,nti,ntj->ij", w, xhat, ys)
    )

    sum_P_curr = jnp.einsum("n,ntij->ij", w, P[:, 1:])
    Q_new = (1.0 / (N * (T - 1))) * (sum_P_curr - A_new @ sum_cross.T)

    # initial state: condition-weighted too, so the ring of initial states is not tilted
    # towards the better-sampled directions either
    mu0_new = jnp.einsum("n,nd->d", w, xhat[:, 0]) / N
    V0_new = jnp.einsum("n,nij->ij", w, P[:, 0]) / N - jnp.outer(mu0_new, mu0_new)

    return A_new, C_new, Q_new, R_new, mu0_new, V0_new


@jax.jit
def _em_step_weighted(ys, w, A, C, Q, R, mu0, V0, r_floor):
    """Mirrors diag._em_step: same Q floor, same diagonal-R clip, weighted M-step."""
    xhat, P, Pcross, loglik = em.e_step_batch(ys, A, C, Q, R, mu0, V0)
    ll = jnp.sum(w * loglik)                      # the WEIGHTED objective, i.e. the split loss
    A, C, Q, R, mu0, V0 = m_step_weighted(ys, xhat, P, Pcross, w)
    Id = jnp.eye(A.shape[0])
    Q = 0.5 * (Q + Q.T) + Q_FLOOR * Id
    R = jnp.diag(jnp.clip(jnp.diag(R), r_floor, None))
    V0 = 0.5 * (V0 + V0.T) + Q_FLOOR * Id
    return (A, C, Q, R, mu0, V0), ll


def run_track_weighted(ys, w, init, n_iters, r_floor):
    """EM under the split loss. Returns (params, weighted_lls, eigs), matching diag.run_track."""
    ys, w = jnp.asarray(ys), jnp.asarray(np.asarray(w, float))
    params = tuple(jnp.asarray(p) for p in init)
    rf = jnp.asarray(float(r_floor))
    lls, eigs = [], []
    for _ in range(n_iters):
        eigs.append(np.linalg.eigvals(np.asarray(params[0])))
        params, ll = _em_step_weighted(ys, w, *params, rf)
        lls.append(float(ll))
    eigs.append(np.linalg.eigvals(np.asarray(params[0])))
    return params, np.asarray(lls), np.asarray(eigs)


def fit_identity_weighted(y, targ, D, rf, nit):
    """Drop-in for fitting.fit_identity, under the condition-split loss."""
    from neural_pca_init import identity_init
    return run_track_weighted(y, condition_weights(targ), identity_init(y, D), nit, rf)


# selftest
def selftest(seed=0, N=64, T=12, M=20, D=4, nit=8):
    """Balanced counts => weights all 1 => must reproduce the unweighted fit exactly."""
    import diag
    rng = np.random.default_rng(seed)
    y = rng.standard_normal((N, T, M))
    targ = np.repeat(np.arange(8) * np.pi / 4, N // 8)          # balanced, 8 trials each
    w = condition_weights(targ)
    print(f"balanced: weights min {w.min():.6f} max {w.max():.6f} sum {w.sum():.3f} (N={N})")

    from neural_pca_init import identity_init
    init = identity_init(y, D)
    pa, la, _ = diag.run_track(y, init, nit, 1e-2)
    pb_, lb, _ = run_track_weighted(y, w, init, nit, 1e-2)
    worst = max(float(np.abs(np.asarray(a) - np.asarray(b)).max()) for a, b in zip(pa, pb_))
    print(f"balanced params   max |diff| = {worst:.3e}")
    print(f"balanced LL       max |diff| = {np.abs(la - lb).max():.3e}")

    # unbalanced: weights must depart from 1, and the fit must differ
    targ2 = np.concatenate([np.full(N - 28, 0.0), np.full(20, 1.0), np.full(8, 2.0)])
    w2 = condition_weights(targ2)
    print(f"unbalanced: weights {np.unique(np.round(w2, 3))} sum {w2.sum():.3f}")
    pc, lc, _ = run_track_weighted(y, w2, init, nit, 1e-2)
    diff = max(float(np.abs(np.asarray(a) - np.asarray(c)).max()) for a, c in zip(pa, pc))
    print(f"unbalanced params max |diff| from unweighted = {diff:.3e}  (should be >> 0)")
    print(f"weighted LL monotone increasing: {bool(np.all(np.diff(lc) > -1e-6))}")


if __name__ == "__main__":
    selftest()
