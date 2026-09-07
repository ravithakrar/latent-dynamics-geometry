"""metric_compare.py — Metric Similarity Analysis (MSA) kernel.

Compares two pullback metric fields built by pullback_metric.py. At each manifold point it
solves the generalised eigenproblem G v = lambda G' v and, for lambda_1 >= ... >= lambda_m,
returns

    d_SR(G, G') = 1 - sqrt(lambda_m / lambda_1)   in [0, 1]
    d_MSA       = (1 / Vol(M)) INT_M d_SR(G_p, G'_p) dvol(p)

d_SR is invariant to scale, so every reduction also returns the scale terms it discards:
log_scale = mean(l_i)/2 and d_shape = ||l - mean(l)|| for l_i = log(lambda_i), composing as
d_airm^2 = m (2 log_scale)^2 + d_shape^2.

Metric fields are (..., m, m) with the manifold grid on the leading axes, e.g. (K, T, 2, 2).
Non-finite and non-SPD points are masked, and every reducing function reports the fraction
dropped.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "gen_eigvals", "d_sr", "d_airm", "scale_shape", "volume_form",
    "spd_mask", "weights_for", "aggregate", "msa", "msa_matrix", "self_test",
]

#: relative ridge added to both metrics before the Cholesky, purely for
#: numerical conditioning.  1e-12 is ~4 orders below float64 resolution of a
#: well-scaled 2x2 and does not move any reported number.
RIDGE = 1e-12

#: grid points with condition number above this are treated as degenerate and
#: dropped (with the drop fraction reported).  A 2x2 pullback metric becomes
#: singular when the time tangent and the direction tangent become collinear,
#: or when one of them vanishes -- both happen late in the reach as the
#: autonomous latent decays, so this guard is not hypothetical.
COND_MAX = 1e10


# internals
def _sym(G):
    """Symmetrise, killing float asymmetry from J^T J accumulation."""
    G = np.asarray(G, float)
    return 0.5 * (G + np.swapaxes(G, -1, -2))


def _ridged(G, ridge):
    """G + ridge * diag(diag(G))  -- a per-coordinate relative ridge.

    A diagonal ridge scales with each coordinate and is exactly covariant under a
    coordinate rescaling S = diag(a, b). An isotropic ridge, ridge * (tr G / m) * I,
    is not.
    """
    G = _sym(G)
    if not ridge:
        return G
    m = G.shape[-1]
    d = np.einsum("...ii->...i", G)[..., None] * np.eye(m)
    return G + ridge * d


def _batched_cholesky(G):
    """Cholesky over leading axes, returning (L, ok) with ok False where it fails."""
    G = np.ascontiguousarray(G)
    flat = G.reshape(-1, G.shape[-2], G.shape[-1])
    L = np.empty_like(flat)
    ok = np.ones(flat.shape[0], bool)
    for i in range(flat.shape[0]):
        try:
            L[i] = np.linalg.cholesky(flat[i])
        except np.linalg.LinAlgError:
            L[i] = np.eye(G.shape[-1])
            ok[i] = False
    return L.reshape(G.shape), ok.reshape(G.shape[:-2])


# generalised spectrum
def gen_eigvals(G, Gp, ridge=RIDGE):
    """Generalised eigenvalues of  G v = lambda Gp v,  sorted DESCENDING.

    Computed by whitening with the Cholesky factor of Gp,
        Gp = L L^T,   S = L^-1 G L^-T   (symmetric),   lambda = eig(S),
    which is the numerically stable route and never forms Gp^-1 G explicitly.

    Args:
        G, Gp: (..., m, m) SPD metric fields, broadcastable against each other.
        ridge: relative ridge for conditioning (see RIDGE).

    Returns:
        lam: (..., m) real, positive, descending.  NaN at points where the
             Cholesky failed (i.e. Gp not numerically SPD).
    """
    G, Gp = np.broadcast_arrays(_ridged(G, ridge), _ridged(Gp, ridge))
    L, ok = _batched_cholesky(Gp)
    Linv = np.linalg.inv(L)
    S = _sym(Linv @ G @ np.swapaxes(Linv, -1, -2))
    lam = np.linalg.eigvalsh(S)                       # ascending
    lam = lam[..., ::-1]                              # -> descending
    lam = np.where(ok[..., None], lam, np.nan)
    # eigvalsh of an SPD matrix is positive up to rounding; clamp the rounding.
    return np.where(lam > 0, lam, np.nan)


def d_sr(G, Gp, ridge=RIDGE):
    """Spectral-ratio distance  1 - sqrt(lambda_min / lambda_max)  in [0, 1].

    0 iff G' = c G for some c > 0 (identical shape, any scale).
    """
    lam = gen_eigvals(G, Gp, ridge=ridge)
    return 1.0 - np.sqrt(lam[..., -1] / lam[..., 0])


def d_airm(G, Gp, ridge=RIDGE):
    """Affine-invariant Riemannian distance  ||log lambda||_2  in [0, inf).

    Unbounded, and NOT scale-invariant -- included as the complement to d_SR.
    """
    lam = gen_eigvals(G, Gp, ridge=ridge)
    return np.sqrt(np.sum(np.log(lam) ** 2, axis=-1))


def scale_shape(G, Gp, ridge=RIDGE):
    """Split the disagreement into a scale part and a shape part.

    Returns a dict of (...)-shaped arrays:
        log_scale: mean(log lambda)/2  -- mean log length ratio of G over Gp.
                   > 0 means G is the larger metric. d_SR discards this quantity.
        d_shape:   ||log lambda - mean||_2 -- anisotropy disagreement,
                   scale-invariant, 0 iff G' = cG.
        d_airm:    ||log lambda||_2, with d_airm^2 = m*(2*log_scale)^2 + d_shape^2.
        d_sr:      the paper's bounded measure, for convenience.
    """
    lam = gen_eigvals(G, Gp, ridge=ridge)
    l = np.log(lam)
    m = l.shape[-1]
    lbar = np.mean(l, axis=-1)
    return dict(
        log_scale=lbar / 2.0,
        d_shape=np.sqrt(np.sum((l - lbar[..., None]) ** 2, axis=-1)),
        d_airm=np.sqrt(np.sum(l ** 2, axis=-1)),
        d_sr=1.0 - np.sqrt(lam[..., -1] / lam[..., 0]),
        m=m,
    )


# volume + conditioning
def volume_form(G, ridge=0.0):
    """sqrt(det G) -- the Riemannian volume element of the metric field."""
    return np.sqrt(np.maximum(np.linalg.det(_ridged(G, ridge)), 0.0))


def spd_mask(G, cond_max=COND_MAX):
    """True where G is finite, positive definite and better conditioned than cond_max."""
    G = _sym(G)
    finite = np.isfinite(G).all(axis=(-2, -1))
    ev = np.linalg.eigvalsh(np.where(finite[..., None, None], G, np.eye(G.shape[-1])))
    pos = ev[..., 0] > 0
    cond = ev[..., -1] / np.where(ev[..., 0] > 0, ev[..., 0], np.nan)
    return finite & pos & (cond <= cond_max)


def weights_for(G1, G2, scheme="uniform"):
    """Quadrature weights over the manifold grid.

    The MSA integral is against dvol_g of a metric g on the INPUT manifold. The task
    manifold (t, theta) carries no canonical metric, so the weighting is a declared
    choice.

        "uniform"   flat metric on (t, theta): every grid point counts equally. The
                    default; the comparison does not depend on either metric compared.
        "geometric" sqrt( sqrt(det G1) * sqrt(det G2) ): weight by the geometric
                    mean of the two volume elements.  Symmetric in 1 <-> 2, and
                    downweights regions where either representation is
                    degenerate.  NOT scale-invariant across sessions.
        "vol1"/"vol2"  weight by sqrt(det G1) or sqrt(det G2) alone.  Asymmetric;
                    provided for completeness, breaks d(1,2) = d(2,1).

    Note "uniform" is also the only scheme that preserves the coordinate
    invariance of the *aggregate*: dvol picks up |det S| under a coordinate
    change, which cancels against Vol(M) only for a global rescaling.
    """
    if scheme == "uniform":
        return np.ones(np.broadcast_shapes(G1.shape[:-2], G2.shape[:-2]))
    if scheme == "vol1":
        return volume_form(G1)
    if scheme == "vol2":
        return volume_form(G2)
    if scheme == "geometric":
        return np.sqrt(volume_form(G1) * volume_form(G2))
    raise ValueError(f"unknown weighting scheme {scheme!r}")


def aggregate(values, weights, mask=None):
    """Weighted mean of `values` over the grid, ignoring masked/non-finite points.

    Returns (value, frac_used).
    """
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    good = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    if mask is not None:
        good &= np.asarray(mask, bool)
    if not good.any():
        return float("nan"), 0.0
    w = np.where(good, weights, 0.0)
    if w.sum() <= 0:
        return float("nan"), 0.0
    return float(np.sum(np.where(good, values, 0.0) * w) / w.sum()), float(good.mean())


# top level
def msa(G1, G2, scheme="uniform", cond_max=COND_MAX, ridge=RIDGE):
    """Volume-weighted MSA distance between two metric FIELDS on a shared grid.

    Args:
        G1, G2:   (..., m, m) metric fields sampled at the SAME manifold points.
        scheme:   quadrature weighting, see `weights_for`.
        cond_max: drop grid points where either metric is worse conditioned.
        ridge:    relative ridge for the generalised eigenproblem.

    Returns:
        dict with d_msa (the headline), plus the scale/shape companions
        aggregated the same way, the pointwise arrays, and bookkeeping.
    """
    G1 = _sym(G1)
    G2 = _sym(G2)
    if G1.shape[-1] != G2.shape[-1]:
        raise ValueError(f"metric dimension mismatch: {G1.shape[-1]} vs {G2.shape[-1]}")
    parts = scale_shape(G1, G2, ridge=ridge)
    mask = spd_mask(G1, cond_max) & spd_mask(G2, cond_max)
    w = weights_for(G1, G2, scheme)

    d_msa, frac = aggregate(parts["d_sr"], w, mask)
    out = dict(
        d_msa=d_msa,
        similarity=1.0 - d_msa,
        frac_points_used=frac,
        n_points=int(np.prod(np.broadcast_shapes(G1.shape[:-2], G2.shape[:-2]))),
        scheme=scheme,
        m=int(G1.shape[-1]),
    )
    for key in ("d_shape", "d_airm"):
        out[key], _ = aggregate(parts[key], w, mask)
    # scale is signed; report the mean (bias: is G1 systematically bigger?) and
    # the mean absolute (magnitude disagreement regardless of sign).
    out["log_scale_mean"], _ = aggregate(parts["log_scale"], w, mask)
    out["log_scale_absmean"], _ = aggregate(np.abs(parts["log_scale"]), w, mask)
    out["_pointwise"] = {k: parts[k] for k in ("d_sr", "d_shape", "log_scale", "d_airm")}
    out["_mask"] = mask
    return out


def msa_matrix(fields, labels=None, scheme="uniform", cond_max=COND_MAX, ridge=RIDGE,
               key="d_msa"):
    """Pairwise MSA dissimilarity matrix over a list of metric fields.

    Args:
        fields: sequence of (..., m, m) arrays, all on the same grid.
        labels: optional names, returned alongside.
        key:    which scalar to tabulate ("d_msa", "d_shape", "log_scale_mean", ...).

    Returns:
        (M, dict) with M the (S, S) matrix (diagonal exactly 0 for distances)
        and dict carrying labels and the full per-pair records.
    """
    S = len(fields)
    M = np.zeros((S, S))
    records = {}
    signed = key.startswith("log_scale") and not key.endswith("absmean")
    for i in range(S):
        for j in range(i + 1, S):
            r = msa(fields[i], fields[j], scheme=scheme, cond_max=cond_max, ridge=ridge)
            r.pop("_pointwise"); r.pop("_mask")
            records[f"{i}|{j}"] = r
            M[i, j] = r[key]
            M[j, i] = -r[key] if signed else r[key]
    return M, dict(labels=list(labels) if labels is not None else list(range(S)),
                   key=key, scheme=scheme, pairs=records)


# self-test
def self_test(verbose=True):
    """Property checks on the kernel.  Returns a dict of max errors."""
    rng = np.random.default_rng(0)
    shape = (8, 25)                                   # (K, T) like the real grid
    m = 2

    def rand_spd(sh, m=m, cond=None):
        Jf = rng.standard_normal(sh + (m + 6, m))     # J is (features, m)
        G = np.einsum("...fi,...fj->...ij", Jf, Jf)
        if cond is not None:
            G = G + cond * np.eye(m)
        return G

    G1, G2 = rand_spd(shape), rand_spd(shape)
    err = {}

    # identity / separation
    err["identity_d_sr"] = float(np.nanmax(np.abs(d_sr(G1, G1))))

    # scale invariance:  d_SR(G, cG) = 0, and log_scale recovers log c / ... exactly
    c = 7.3
    err["scale_invariance_d_sr"] = float(np.nanmax(np.abs(d_sr(G1, c * G1))))
    ss = scale_shape(G1, c * G1)
    err["log_scale_recovery"] = float(np.nanmax(np.abs(ss["log_scale"] + 0.5 * np.log(c))))
    err["scale_invariance_d_shape"] = float(np.nanmax(np.abs(ss["d_shape"])))

    # congruence / coordinate invariance:  G -> S^T G S for invertible S
    S = np.array([[3.0, 0.0], [0.0, 0.01]])           # rescale t and theta units
    err["congruence_diag"] = float(np.nanmax(np.abs(
        d_sr(S.T @ G1 @ S, S.T @ G2 @ S) - d_sr(G1, G2))))
    Sg = rng.standard_normal((m, m)) + 2 * np.eye(m)  # general invertible chart change
    err["congruence_general"] = float(np.nanmax(np.abs(
        d_sr(Sg.T @ G1 @ Sg, Sg.T @ G2 @ Sg) - d_sr(G1, G2))))

    # state-space rotation invariance:  J -> QJ leaves G = J^T J untouched
    Jf = rng.standard_normal(shape + (12, m))
    Q = np.linalg.qr(rng.standard_normal((12, 12)))[0]
    Gj = np.einsum("...fi,...fj->...ij", Jf, Jf)
    Jq = np.einsum("gf,...fi->...gi", Q, Jf)
    Gq = np.einsum("...fi,...fj->...ij", Jq, Jq)
    err["rotation_invariance"] = float(np.nanmax(np.abs(Gj - Gq)) / np.nanmax(np.abs(Gj)))

    # symmetry
    err["symmetry"] = float(np.nanmax(np.abs(d_sr(G1, G2) - d_sr(G2, G1))))

    # triangle inequality on d_SR (Prop 2.4), pointwise
    G3 = rand_spd(shape)
    tri = d_sr(G1, G3) - (d_sr(G1, G2) + d_sr(G2, G3))
    err["triangle_violation"] = float(np.nanmax(tri))          # must be <= 0

    # AIRM decomposition identity: d_airm^2 = m*(2*log_scale)^2 + d_shape^2
    ss12 = scale_shape(G1, G2)
    lhs = ss12["d_airm"] ** 2
    rhs = m * (2 * ss12["log_scale"]) ** 2 + ss12["d_shape"] ** 2
    err["airm_decomposition"] = float(np.nanmax(np.abs(lhs - rhs)))

    # d_SR as a function of the log-spread:  d_SR = 1 - exp(-(l1 - lm)/2)
    lam = gen_eigvals(G1, G2)
    err["d_sr_logspread"] = float(np.nanmax(np.abs(
        d_sr(G1, G2) - (1 - np.exp(-(np.log(lam[..., 0]) - np.log(lam[..., -1])) / 2)))))

    # closed form for 2x2 against the Cholesky route
    a = np.linalg.solve(G2, G1)
    tr = np.trace(a, axis1=-2, axis2=-1)
    det = np.linalg.det(a)
    disc = np.sqrt(np.maximum(tr ** 2 - 4 * det, 0))
    lp, lm_ = (tr + disc) / 2, (tr - disc) / 2
    err["closed_form_2x2"] = float(np.nanmax(np.abs(d_sr(G1, G2) - (1 - np.sqrt(lm_ / lp)))))

    # aggregate: uniform weighting of a constant field returns the constant
    r = msa(G1, c * G1)
    err["aggregate_constant"] = abs(r["d_msa"])

    # bounds
    dd = d_sr(G1, G2)
    err["bounds"] = float(max(0.0 - np.nanmin(dd), np.nanmax(dd) - 1.0, 0.0))

    if verbose:
        width = max(len(k) for k in err)
        for k, v in err.items():
            flag = "ok " if (v <= 1e-8 if k != "triangle_violation" else v <= 1e-12) else "!! "
            print(f"  {flag}{k:<{width}}  {v:.3e}")
    return err


if __name__ == "__main__":
    print("metric_compare self-test")
    e = self_test()
    bad = {k: v for k, v in e.items()
           if (v > 1e-12 if k == "triangle_violation" else v > 1e-8)}
    print("\nFAILED:" if bad else "\nall properties hold to ~1e-8", bad or "")
