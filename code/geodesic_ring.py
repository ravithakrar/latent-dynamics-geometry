"""geodesic_ring.py -- arc-length geometry of the direction ring and its 2-D embedding.

Single source of truth for the arc-length arithmetic, used by build_ring_fields.py,
mk43_geodesic.py, mk_L_constructions.py and resample43.py.

Restricting the pullback metric to theta gives a 1-D metric on the ring of reach directions,

    ds = sqrt(g_thth) dtheta,      L(t) = CLOSED-INTEGRAL sqrt(g_thth) dtheta

so the ring has a circumference L(t) and a local radius sqrt(g_thth)(theta), and the distance
between two directions is the arc length the shorter way round. classical_mds embeds that
K x K arc-length matrix in two dimensions.

align_ring fixes the reflection first, from the sign of the polygon area taken in
ascending-theta order, and only then solves for the rotation: numpy.linalg.eigh fixes
eigenvector signs arbitrarily, so without that step about half the time bins come out mirrored.

circle_stress_floor(K) returns the embedding stress for K equally spaced points on a perfect
circle (0.187 at K = 8), which is the floor any closed ring is measured against.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "arc_length", "arc_matrix", "classical_mds", "align_ring",
    "stress1", "circle_stress_floor", "embed_ring",
]


# arc length
def arc_length(radius, thf):
    """Circumference and cumulative arc length round the ring.

    radius  (n,)  the LOCAL RADIUS sqrt(g_thth), already square-rooted, at angles thf
    thf     (n,)  ascending angles spanning one full turn

    Returns (L, s) with s (n,) the cumulative arc length from thf[0].
    """
    radius = np.asarray(radius, float)
    dth = np.gradient(np.asarray(thf, float))
    L = float(np.sum(radius * dth))
    s = np.concatenate([[0.0], np.cumsum(radius[:-1] * dth[:-1])])
    return L, s


def arc_matrix(radius, thf, theta_knots):
    """Pairwise geodesic distance between the measured reach directions.

    The shorter of the two ways round, evaluated at the fine-grid index nearest each knot
    angle. Returns (D, idx) with D (K,K) and idx the fine indices used.
    """
    thf = np.asarray(thf, float)
    L, s = arc_length(radius, thf)
    idx = [int(np.argmin(np.abs(np.angle(np.exp(1j * (thf - t))))))
           for t in np.asarray(theta_knots, float)]
    K = len(idx)
    D = np.zeros((K, K))
    for a in range(K):
        for b in range(a + 1, K):
            d = abs(s[idx[a]] - s[idx[b]])
            D[a, b] = D[b, a] = min(d, L - d)
    return D, np.asarray(idx)


# MDS
def classical_mds(D, dim=2):
    """Classical MDS. Deterministic; no optimiser, no seed.

    Double-centre the squared distances, B = -1/2 J D^2 J, and keep the top `dim` eigenpairs
    scaled by sqrt(eigenvalue). Returns (coords (n,dim), eigenvalues (n,) descending, with
    negatives retained so the caller can see how non-Euclidean D was).
    """
    D = np.asarray(D, float)
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh((B + B.T) / 2)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    coords = V[:, :dim] * np.sqrt(np.maximum(w[:dim], 0.0))
    return coords, w


def align_ring(xy, ang):
    """Put an MDS configuration into a frame comparable across time bins and sessions.

    Classical MDS pins the configuration down only up to a rotation and a REFLECTION. Both
    must be removed before two bins can be overlaid, and the reflection must go first: on a
    mirrored configuration the rotation fit below is degenerate.

    xy   (K,2)  MDS coordinates
    ang  (K,)   the true reach angle of each point

    Returns (xy_aligned, resultant). resultant in [0,1] is the circular resultant of
    (ang - embedded angle) AFTER the reflection fix: 1.0 means the embedded points sit at
    their true angles; near 0 means the configuration is not ring-like and the rotation is
    not meaningful.
    """
    xy = np.array(xy, float)
    ang = np.asarray(ang, float)

    # reflection: signed polygon area traversed in ascending true angle
    e = xy[np.argsort(ang)]
    area = 0.5 * np.sum(e[:, 0] * np.roll(e[:, 1], -1) - np.roll(e[:, 0], -1) * e[:, 1])
    if area < 0:
        xy[:, 1] = -xy[:, 1]

    # rotation: circular mean of the angular residual
    emb_ang = np.arctan2(xy[:, 1], xy[:, 0])
    z = np.mean(np.exp(1j * (ang - emb_ang)))
    rot = float(np.angle(z))
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    return xy @ R.T, float(abs(z))


# stress
def stress1(D, xy):
    """Kruskal stress-1 of a configuration against the distance matrix it came from."""
    D = np.asarray(D, float)
    Dh = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    iu = np.triu_indices(D.shape[0], 1)
    return float(np.sqrt(((D[iu] - Dh[iu]) ** 2).sum() / (D[iu] ** 2).sum()))


def circle_stress_floor(K=8):
    """Stress-1 for K equally spaced points on a PERFECT circle.

    The irreducible cost of flattening a closed loop: arcs are longer than chords, so no
    planar configuration reproduces arc lengths exactly. 0.1865 at K=8.
    """
    th = np.arange(K) * 2 * np.pi / K
    d = np.abs(th[:, None] - th[None, :])
    d = np.minimum(d, 2 * np.pi - d)
    xy, _ = classical_mds(d, 2)
    return stress1(d, xy)


# convenience
def embed_ring(D, ang):
    """arc-length matrix -> aligned 2-D ring, with its diagnostics.

    Returns dict(xy, resultant, stress, floor, neg_frac); neg_frac is the negative
    eigenvalue mass as a fraction of the positive mass, i.e. how non-Euclidean D is
    (0.31 for a perfect 8-point circle).
    """
    xy, w = classical_mds(D, 2)
    xy, res = align_ring(xy, ang)
    pos = w[w > 0].sum()
    return dict(xy=xy, resultant=res, stress=stress1(D, xy),
                floor=circle_stress_floor(len(ang)),
                neg_frac=float(abs(w[w < 0]).sum() / pos) if pos > 0 else float("nan"))
