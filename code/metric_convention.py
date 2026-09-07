"""metric_convention.py — one place that fixes how the metric is built.

The settled convention:

    method   = "pcubic"      periodic cubic SMOOTHING spline
    lam      = 0.3           chosen by the trial-split CV, both regions
    push     = True          tangent estimated once at t = 0, carried by A^t

`msa_fields.npz` and `subm_fields.npz` hold `g` and `g_push` computed at
method="cubic", lam=0.0, which is NOT this convention. They also store xbar, theta, A and C,
so everything here is recomputed from the cache at no cost in EM fits. Use these helpers rather
than reading a stored `g` field, so that a change of convention is one edit.
"""
from __future__ import annotations

import numpy as np

import pullback_metric as pb
import tangent_data as td

METHOD, LAM, PUSH = "pcubic", 0.3, True


def label():
    return f"{METHOD}, $\\lambda={LAM}$, " + ("pushforward" if PUSH else "direct")


def metric(xbar, theta, A, C, push=None):
    """(K,T,2,2) metric in neural space at the settled convention."""
    push = PUSH if push is None else push
    fn = pb.metric_pushforward if push else pb.metric
    return fn(xbar, A, C=C, theta=theta, method=METHOD, lam=LAM)[0]


def ring(xbar, theta, A, C, n_fine=240, push=None):
    """The ring on a FINE angular grid: local radius sqrt(g_thth) and the grid itself.

    Under the pushforward the tangent is taken once at t = 0 on the fine grid and carried
    forward by A^t, which is the fine-grid form of pb.metric_pushforward.
    Returns (thf (n,), radius (n,T)).
    """
    push = PUSH if push is None else push
    th0 = np.sort(np.asarray(theta, float))[0]
    thf = np.linspace(th0, th0 + 2 * np.pi, n_fine, endpoint=False)
    T = xbar.shape[1]
    if push:
        d0 = pb.interp_curve(xbar, theta, thf, lam=LAM, deriv=1)[:, 0, :]   # (n,D) at t=0
        d = td.transport_powers(d0, np.asarray(A), T)                       # (n,T,D)
    else:
        d = pb.interp_curve(xbar, theta, thf, lam=LAM, deriv=1)             # (n,T,D)
    radius = np.sqrt(np.maximum(np.sum((d @ np.asarray(C).T) ** 2, axis=-1), 0.0))
    return thf, radius
