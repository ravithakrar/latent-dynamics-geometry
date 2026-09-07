"""EM inference for the 1-D linear dynamical system (LDS).

Pipeline:
    E-step (per trajectory): Kalman filter -> RTS smoother -> lag-one covariance,
        giving the smoothed expectations the M-step needs.
    M-step (over the batch):  closed-form updates for (a, b, q, r, mu0, V0).
    run_em:                   alternate E and M steps, tracking the log-likelihood.

Import these to run experiments and plot the results.
"""

import jax
jax.config.update("jax_enable_x64", True)   # float64: required for a strictly monotonic LL
import jax.numpy as jnp



# Forward pass: Kalman filter                                                  
def kalman_filter(y, A, C, Q, R, mu0, V0):
    """Run the Kalman filter over a single observation sequence y (shape (T,M)).

    Returns filtered means/vars, predicted means/vars, and the log-likelihood.
    """
    T = y.shape[0]
    M = y.shape[1]
    D = mu0.shape[0]

    solve = jnp.linalg.solve
    I = jnp.eye(D)

    # storage
    x_filt = jnp.zeros((T,D))
    V_filt = jnp.zeros((T,D,D))
    x_pred = jnp.zeros((T,D))
    V_pred = jnp.zeros((T,D,D))

    # t = 1: prediction is the prior
    xp, Vp = mu0, V0                      # predicted mean/var at t=1
    S = C @ Vp @ C.T + R                  # innovation covariance
    K = solve(S, (Vp @ C.T).T).T          # Kalman gain  (K = Vp Cᵀ S⁻¹)
    innov = y[0] - C @ xp                 # innovation
    xf = xp + K @ innov                   # corrected mean
    Vf = (I - K @ C) @ Vp                 # corrected


    sign, logdetS = jnp.linalg.slogdet(S)
    loglik = -0.5 * (M * jnp.log(2 * jnp.pi) + logdetS + innov @ solve(S, innov))

    x_pred = x_pred.at[0].set(xp)
    V_pred = V_pred.at[0].set(Vp)
    x_filt = x_filt.at[0].set(xf)
    V_filt = V_filt.at[0].set(Vf)

    # t = 2 -> T
    for t in range(1, T):
        # PREDICT from previous filtered estimate
        xp = A @ x_filt[t - 1]                  # A @ previous filtered mean
        Vp = A @ V_filt[t - 1] @ A.T + Q        #  A @ previous filtered var @ A.T + Q

        # CORRECT using y[t]
        S = C @ Vp @ C.T + R
        K = solve(S, (Vp @ C.T).T).T            # K = Vp Cᵀ S⁻¹
        innov = y[t] - C @ xp                   # fresh innovation
        xf = xp + K @ innov                     # xp + K * innovation
        Vf = (I - K @ C) @ Vp                   # (I - K @ C) @ Vp
        _, logdetS = jnp.linalg.slogdet(S)      # fresh log-det for this t
        loglik += -0.5 * (M * jnp.log(2 * jnp.pi) + logdetS + innov @ solve(S, innov))

        x_pred = x_pred.at[t].set(xp)
        V_pred = V_pred.at[t].set(Vp)
        x_filt = x_filt.at[t].set(xf)
        V_filt = V_filt.at[t].set(Vf)

    return x_filt, V_filt, x_pred, V_pred, loglik


# Backward pass: RTS smoother + lag-one covariance                             
def rts_smoother(x_filt, V_filt, x_pred, V_pred, A):
    """Rauch-Tung-Striebel backward smoother.

    Returns smoothed means/vars and the smoother gains J (needed for the
    lag-one cross-covariance).
    """
    T = x_filt.shape[0]
    D = x_filt.shape[1]

    solve = jnp.linalg.solve

    # storage for smoothed estimates
    x_smooth = jnp.zeros((T,D))
    V_smooth = jnp.zeros((T,D,D))
    J_store = jnp.zeros((T,D,D))        # need J's later for the cross-cov

    # initialize at t = T: smoothed = filtered
    x_smooth = x_smooth.at[T - 1].set(x_filt[T - 1])
    V_smooth = V_smooth.at[T - 1].set(V_filt[T - 1])

    # walk backward: t = T-1 down to 0
    for t in range(T - 2, -1, -1):
        J = solve(V_pred[t + 1], (V_filt[t] @ A.T).T).T   # J = V_filt Aᵀ V_pred⁻¹
        xs = x_filt[t] + J @ (x_smooth[t + 1] - A @ x_filt[t])
        Vs = V_filt[t] + J @ (V_smooth[t + 1] - V_pred[t + 1]) @ J.T

        J_store = J_store.at[t].set(J)
        x_smooth = x_smooth.at[t].set(xs)
        V_smooth = V_smooth.at[t].set(Vs)

    return x_smooth, V_smooth, J_store


def lag_one_cov(V_smooth, J_store):
    """Lag-one smoothed cross-covariance: Vcross[t] = V_smooth[t] @ J[t-1].T.

    Each Vcross[t] is the (D, D) smoothed covariance between state t and
    state t-1, needed by the M-step for the dynamics update. Vcross[0] is
    left as zero (no previous state at t=0).
    """
    T = V_smooth.shape[0]
    D = V_smooth.shape[1]                       # latent dimension

    Vcross = jnp.zeros((T, D, D))               # (T, D, D), one cross-cov per step
                                                # Vcross[t] = V_smooth[t] @ J[t-1].T, vectorised over t via aligned slices:
                                                #   J_store[:-1] = J[0..T-2]  (the t-1 terms),  V_smooth[1:] = V[1..T-1] (the t terms)
                                                # Order matters in >1D: the lag-one cov is V^s_t J_{t-1}^T, not J_{t-1} V^s_t.
    Vcross = Vcross.at[1:].set(
        jnp.einsum("tij,tkj->tik", V_smooth[1:], J_store[:-1])
    )
    return Vcross


# E-step                                                                       
def e_step_single(y, A, C, Q, R, mu0, V0):
    """E-step for a single trajectory: returns the expectations the M-step needs.

        For a D-dimensional latent state x_t and M-dimensional observation y_t:
        xhat   = E[x_t | y_1:T]                         (T, D)
        P      = E[x_t x_t.T | y_1:T]                   (T, D, D)
        Pcross = E[x_t x_{t-1}.T | y_1:T], t = 1..T-1   (T-1, D, D)

    Also returns the per-trajectory log-likelihood.
    """
    # forward pass
    x_filt, V_filt, x_pred, V_pred, loglik = kalman_filter(y, A, C, Q, R, mu0, V0)
    # backward pass
    x_smooth, V_smooth, J_store = rts_smoother(x_filt, V_filt, x_pred, V_pred, A)
    # lag-one cross-covariance
    Vcross = lag_one_cov(V_smooth, J_store)

    xhat = x_smooth                                                      # E[x_t | y_1:T], shape (T, D)
    P = V_smooth + jnp.einsum("ti,tj->tij", xhat, xhat)                  # E[x_t x_t.T | y_1:T], shape (T, D, D)
    Pcross = Vcross[1:] + jnp.einsum("ti,tj->tij", xhat[1:], xhat[:-1])  # E[x_t x_{t-1}.T | y_1:T], shape (T-1, D, D)

    return xhat, P, Pcross, loglik


# Vectorise the E-step across the batch (axis 0 = trajectories).
e_step_batch = jax.vmap(e_step_single, in_axes=(0, None, None, None, None, None, None))



# M-step                                                                       
def m_step(ys, xhat, P, Pcross):
    """Closed-form M-step updates for (A, C, Q, R, mu0, V0).

    Args:
        ys:     (N, T, M) observations.
        xhat:   (N, T, D) smoothed means.
        P:      (N, T, D, D) smoothed second moments.
        Pcross: (N, T-1, D, D) smoothed lag-one cross moments.
    """
    N, T, M = ys.shape

    solve = jnp.linalg.solve

    # dynamics a: sum(Pcross) / sum(P over t=1..T-1)

    sum_cross = jnp.sum(Pcross, axis=(0,1))
    sum_prev = jnp.sum(P[:,:-1], axis=(0,1))
    A_new = solve(sum_prev, sum_cross.T).T   # A = sum_cross sum_prev⁻¹  (denominator excludes last column)

    # emission C: (sum_t y_t x_tᵀ) (sum_t x_t x_tᵀ)⁻¹  via solve
    sum_P = jnp.sum(P, axis=(0,1))
    C_new = solve(sum_P, jnp.einsum("nti,ntj -> ij", ys, xhat).T).T   # C = (sum y xᵀ) (sum x xᵀ)⁻¹

    # observation noise R: (1/NT) * sum( y yᵀ - C_new x yᵀ )
    R_new = (1.0 / (N * T)) * (
        jnp.einsum("nti,ntj->ij", ys, ys)
        - C_new @ jnp.einsum("nti,ntj->ij", xhat, ys)
    )

   # process noise Q: (1/(N(T-1))) * sum(P[t] - A_new @ Pcross^T) over t=2..T
    sum_cross = jnp.sum(Pcross, axis=(0, 1))
    sum_P_curr = jnp.sum(P[:, 1:], axis=(0, 1))
    Q_new = (1.0 / (N * (T - 1))) * (sum_P_curr - A_new @ sum_cross.T)

    # initial mean & covariance (use only t=1, i.e. column 0)
    mu0_new = jnp.mean(xhat[:, 0], axis=0)
    V0_new = jnp.mean(P[:, 0], axis=0) - jnp.outer(mu0_new, mu0_new)

    return A_new, C_new, Q_new, R_new, mu0_new, V0_new


def run_em(ys, init, n_iters=60):
    """Alternate E and M steps.

    Args:
        ys:      (N, T, M) observations.
        init:    (A, C, Q, R, mu0, V0) starting parameters.
        n_iters: number of EM iterations.

    Returns:
        params: recovered (A, C, Q, R, mu0, V0).
        lls:    (n_iters,) log-likelihood at each iteration (should be increasing).
    """
    A, C, Q, R, mu0, V0 = init
    lls = []
    for _ in range(n_iters):
        xhat, P, Pcross, loglik = e_step_batch(ys, A, C, Q, R, mu0, V0)
        lls.append(float(jnp.sum(loglik)))
        A, C, Q, R, mu0, V0 = m_step(ys, xhat, P, Pcross)
    return (A, C, Q, R, mu0, V0), jnp.array(lls)
