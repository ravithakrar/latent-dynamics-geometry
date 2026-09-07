"""Pullback metric g(t, theta) of the reach representation, from a fitted LDS.

Task variables are time t and reach direction theta (8 centre-out directions); the manifold is
x(t, theta), the latent state, or y = C x, the neural activity. The Jacobian columns are

    d_t x  = (A - I) x            discrete-time velocity, from the fitted dynamics
    d_th x = periodic spline tangent across the 8 directions

and the metric is g = J^T J, a 2x2 at every (t, theta). It is computed in latent space (the
literal J^T J) and in neural space (J_y = C J).

Also provides the pushforward d_th x(t) = A^t d_th x(0), which builds the tangent once at t = 0
and carries it forward instead of resplining at every time bin.

direction_tangent defaults to a periodic cubic spline (method="pcubic" for its penalised form);
method="akima" and method="fourier" are the alternatives.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import Akima1DInterpolator, CubicSpline

#: interpolator used when a caller does not name one explicitly
DEFAULT_METHOD = "cubic"


def condition_means(xhat, targ):
    """Average smoothed latents within each reach direction.
    xhat (N,T,D), targ (N,) -> xbar (K,T,D), theta (K,) sorted-as-given unique dirs."""
    dirs = np.unique(targ)
    xbar = np.stack([np.asarray(xhat)[targ == d].mean(0) for d in dirs])
    return xbar, dirs


# interpolators
def _tangent_akima(xbar, theta, pad=3):
    """Periodic Akima spline. NONLINEAR in the data -> does not commute with A^t."""
    theta = np.asarray(theta, float)
    order = np.argsort(theta)
    th, xb = theta[order], xbar[order]
    th_ext = np.concatenate([th[-pad:] - 2 * np.pi, th, th[:pad] + 2 * np.pi])
    xb_ext = np.concatenate([xb[-pad:], xb, xb[:pad]], axis=0)
    dspline = Akima1DInterpolator(th_ext, xb_ext, axis=0).derivative()
    return dspline(th)[np.argsort(order)]


def _tangent_cubic(xbar, theta):
    """Periodic cubic spline. LINEAR in the sampled values -> commutes with A^t exactly."""
    theta = np.asarray(theta, float)
    order = np.argsort(theta)
    th, xb = theta[order], xbar[order]
    th_c = np.concatenate([th, [th[0] + 2 * np.pi]])          # close the loop
    xb_c = np.concatenate([xb, xb[:1]], axis=0)
    d = CubicSpline(th_c, xb_c, axis=0, bc_type="periodic").derivative()(th)
    return d[np.argsort(order)]


def _tangent_fourier(xbar, theta):
    """Trigonometric interpolant via FFT; exact for equally spaced knots on the circle.
    d/dtheta of sum_k c_k e^{i k theta} = sum_k (i k) c_k e^{i k theta}.
    The Nyquist coefficient is zeroed so the derivative stays real. Linear in the data."""
    theta = np.asarray(theta, float)
    order = np.argsort(theta)
    th, xb = theta[order], xbar[order]
    K = xb.shape[0]
    spacing = np.diff(np.concatenate([th, [th[0] + 2 * np.pi]]))
    if not np.allclose(spacing, spacing[0]):
        raise ValueError("fourier tangent requires equally spaced directions")
    F = np.fft.fft(xb, axis=0)
    k = np.fft.fftfreq(K, d=1.0 / K)
    if K % 2 == 0:
        k[K // 2] = 0.0
    d = np.real(np.fft.ifft(1j * k[:, None, None] * F, axis=0))
    return d[np.argsort(order)]


# penalised (smoothing) spline
# "pcubic" solves
#
#     f_hat = argmin_f  sum_k || xbar_k - f(theta_k) ||^2
#                       + lambda * INTEGRAL_0^2pi || f''(theta) ||^2 dtheta
#
# in the cardinal basis L_j, with L_j(theta_k) = delta_jk, which makes the design matrix the
# identity:
#
#     c_hat = (I + lambda_eff R)^{-1} xbar,      R_ij = INTEGRAL L_i'' L_j'' dtheta
#
# so lambda = 0 returns the interpolating spline of _tangent_cubic; test_pcubic_matches_cubic
# checks that to machine precision. lambda is made dimensionless by multiplying the penalty by
# K / trace(R), so it is comparable across fits.
_OPS_CACHE: dict[bytes, tuple] = {}


def _cardinal_splines(th_sorted):
    """K periodic cubic splines L_j with L_j(theta_k) = delta_jk. `th_sorted` ascending."""
    th = np.asarray(th_sorted, float)
    K = th.size
    th_c = np.concatenate([th, [th[0] + 2 * np.pi]])
    out = []
    for j in range(K):
        e = np.zeros(K)
        e[j] = 1.0
        out.append(CubicSpline(th_c, np.concatenate([e, e[:1]]), bc_type="periodic"))
    return out


def _spline_ops(th_sorted):
    """(L, R, Dm) for ascending angles, cached on the angle grid.

    R_ij = INTEGRAL L_i'' L_j'' dtheta   (roughness penalty, exact)
    Dm_kj = L_j'(theta_k)                (so f'(theta_k) = (Dm @ c)_k)
    """
    th = np.ascontiguousarray(np.asarray(th_sorted, float))
    key = th.tobytes()
    if key in _OPS_CACHE:
        return _OPS_CACHE[key]

    L = _cardinal_splines(th)
    K = th.size
    knots = np.concatenate([th, [th[0] + 2 * np.pi]])
    # L_j'' is piecewise LINEAR, so L_i'' L_j'' is piecewise quadratic and 2-point
    # Gauss-Legendre is exact on each knot span. No quadrature error.
    gx = np.array([-1.0, 1.0]) / np.sqrt(3.0)
    R = np.zeros((K, K))
    for a, b in zip(knots[:-1], knots[1:]):
        half = 0.5 * (b - a)
        q = 0.5 * (a + b) + half * gx
        S = np.stack([Lj.derivative(2)(q) for Lj in L])          # (K, 2)
        R += half * (S @ S.T)
    R = 0.5 * (R + R.T)                                          # symmetrise round-off
    Dm = np.stack([Lj.derivative(1)(th) for Lj in L], axis=1)    # (K, K), column j
    _OPS_CACHE[key] = (L, R, Dm)
    return L, R, Dm


def smoother_matrix(th_sorted, lam=0.0):
    """H with c_hat = H @ xbar, for ascending angles. lam is dimensionless."""
    _, R, _ = _spline_ops(th_sorted)
    K = R.shape[0]
    if lam <= 0:
        return np.eye(K)
    scale = K / max(float(np.trace(R)), 1e-300)
    return np.linalg.solve(np.eye(K) + lam * scale * R, np.eye(K))


def effective_dof(theta, lam=0.0):
    """trace of the smoother = effective number of parameters. K at lam=0, -> 1 as lam -> inf."""
    th = np.sort(np.asarray(theta, float))
    return float(np.trace(smoother_matrix(th, lam)))


def _tangent_pcubic(xbar, theta, lam=0.0):
    """Periodic cubic SMOOTHING spline tangent. lam=0 == _tangent_cubic exactly."""
    theta = np.asarray(theta, float)
    order = np.argsort(theta)
    th, xb = theta[order], np.asarray(xbar, float)[order]
    _, _, Dm = _spline_ops(th)
    c = np.tensordot(smoother_matrix(th, lam), xb, axes=(1, 0))
    d = np.tensordot(Dm, c, axes=(1, 0))
    return d[np.argsort(order)]


def interp_curve(xbar, theta, theta_q, lam=0.0, deriv=0):
    """Evaluate the (optionally penalised) periodic cubic through xbar at theta_q.

    The single entry point for interpolating the direction ring onto a fine grid.

    Args:
        xbar:    (K,...) values at the measured angles.
        theta:   (K,) measured angles, any order.
        theta_q: (Q,) query angles, wrapped into the measured period.
        lam:     dimensionless roughness penalty; 0 -> interpolation.
        deriv:   0 for the curve, 1 for d/dtheta, 2 for the second derivative.

    Returns:
        (Q,...) values.
    """
    theta = np.asarray(theta, float)
    theta_q = np.asarray(theta_q, float)
    order = np.argsort(theta)
    th, xb = theta[order], np.asarray(xbar, float)[order]
    L, _, _ = _spline_ops(th)
    c = np.tensordot(smoother_matrix(th, lam), xb, axes=(1, 0))
    thq = np.mod(theta_q - th[0], 2 * np.pi) + th[0]
    E = np.stack([(Lj.derivative(deriv)(thq) if deriv else Lj(thq)) for Lj in L], axis=1)
    return np.tensordot(E, c, axes=(1, 0))


_METHODS = {"akima": _tangent_akima, "cubic": _tangent_cubic, "fourier": _tangent_fourier}
#: methods that accept a roughness penalty
_PENALISED = ("pcubic",)


def direction_tangent(xbar, theta, method=None, pad=3, lam=0.0):
    """d x / d theta via a PERIODIC spline across reach directions.

    Args:
        xbar:   (K,T,D) condition-averaged latents.
        theta:  (K,) direction angles in radians.
        method: "cubic" (default, commutes with A^t), "pcubic" (the same estimator
                with a roughness penalty), "akima" (legacy, does not commute), or
                "fourier". None -> DEFAULT_METHOD.
        pad:    wrap-padding used by the Akima path only.
        lam:    dimensionless roughness penalty, "pcubic" only. lam=0 reproduces
                "cubic" exactly.

    Returns:
        (K,T,D) tangents aligned to xbar/theta.
    """
    method = DEFAULT_METHOD if method is None else method
    if method in _PENALISED:
        return _tangent_pcubic(xbar, theta, lam=lam)
    if method not in _METHODS:
        raise ValueError(
            f"unknown method {method!r}; choose from {sorted(_METHODS) + list(_PENALISED)}")
    if lam:
        # never silently ignore a penalty the caller asked for
        raise ValueError(f"method={method!r} takes no penalty; use method='pcubic' for lam>0")
    if method == "akima":
        return _tangent_akima(xbar, theta, pad=pad)
    return _METHODS[method](xbar, theta)


# metric
def _gram(dt, dth):
    """Assemble the 2x2 metric from two tangent fields (...,F)."""
    gtt = np.sum(dt * dt, axis=-1)
    gthth = np.sum(dth * dth, axis=-1)
    gtth = np.sum(dt * dth, axis=-1)
    top = np.stack([gtt, gtth], axis=-1)
    bot = np.stack([gtth, gthth], axis=-1)
    return np.stack([top, bot], axis=-2)               # (...,2,2)


def metric(xbar, A, C=None, dth_x=None, theta=None, method=None, lam=0.0):
    """Metric along the trajectory. If C given -> neural (gauge-invariant) space.
    Returns g (K,T,2,2) plus the latent tangents used."""
    D = A.shape[0]
    if dth_x is None:
        dth_x = direction_tangent(xbar, theta, method=method, lam=lam)
    dt_x = xbar @ (A - np.eye(D)).T                    # (K,T,D) = (A-I)x
    if C is not None:
        dt, dth = dt_x @ np.asarray(C).T, dth_x @ np.asarray(C).T
    else:
        dt, dth = dt_x, dth_x
    return _gram(dt, dth), dt_x, dth_x


def pushforward_tangent(dth_x0, A, T):
    """A^t d_th x(0): push the initial direction-tangent forward through A.
    dth_x0 (K,D) -> (K,T,D)."""
    A = np.asarray(A)
    out = np.empty((dth_x0.shape[0], T, A.shape[0]))
    cur = np.asarray(dth_x0, float).copy()
    for t in range(T):
        out[:, t, :] = cur
        cur = cur @ A.T                                # advance one step: A cur
    return out


def metric_pushforward(xbar, A, C=None, theta=None, method=None, lam=0.0):
    """Metric with the direction tangent estimated once at t = 0 and pushed forward.

        d_th x(t) = A^t d_th x(0)

    The tangent is built once at t = 0 and carried forward by A^t rather than resplined
    at each time bin. lam applies once, to that single estimate.

    Returns g (K,T,2,2) plus the latent tangents used, matching `metric`'s signature.
    """
    T = xbar.shape[1]
    dth_x0 = direction_tangent(xbar[:, :1, :], theta, method=method, lam=lam)[:, 0, :]
    dth_x = pushforward_tangent(dth_x0, A, T)
    return metric(xbar, A, C=C, dth_x=dth_x, theta=theta)


def alignment(g, eps=1e-30):
    """cos(angle) between the time and direction tangents = normalised off-diagonal."""
    gtt, gthth, gtth = g[..., 0, 0], g[..., 1, 1], g[..., 0, 1]
    return gtth / np.sqrt(gtt * gthth + eps)
