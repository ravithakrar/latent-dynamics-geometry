"""fitting.py — the fitting entry points every analysis script shares.

One place for the preprocessing and the two EM initialisations, so that every
script in this repository fits the data the same way.

    preprocess_session   NWB file -> {region: {y, targ, N, T, M}}, movement-onset
                         aligned, 20 ms bins, 40 ms smoothing, soft per-unit z-score
    fit_identity         EM from the identity initialisation, the operating choice
    fit_warmstart        EM up a PCA warm-start staircase to D, used as the
                         different-initialisation control

RF is the per-region floor on the observation-noise variance, which stops a
near-silent unit driving R to zero.
"""
from __future__ import annotations

import numpy as np

import diag
from neural_pca_init import identity_init
from neural_data_io import load_session, movement_onset_times, binned_counts
from neural_preprocess import preprocess

BIN, SIG, WIN = 0.02, 0.04, (-0.10, 0.40)
ONSET_BIN = 5
RF = {"M1": 0.01, "PMd": 0.05}
SCHED_FULL = (10, 20, 35, 55)


def preprocess_session(nwb):
    sess = load_session(nwb)
    onsets = movement_onset_times(sess, speed_frac=0.2)
    out = {}
    for region in ["M1", "PMd"]:
        counts, info = binned_counts(sess, region=region, event_times=onsets,
                                     t_start=WIN[0], t_stop=WIN[1], bin_size=BIN)
        y, _ = preprocess(counts, bin_size=BIN, sigma=SIG, softening_hz=5.0)
        out[region] = dict(y=np.asarray(y, float), targ=np.asarray(info["target_dir"], float),
                           N=y.shape[0], T=y.shape[1], M=y.shape[2])
    return out


def fit_warmstart(y, D, rf, nit):
    """PCA warm-start staircase to the fixed operating dimension D, tracking LL and
    eig(A) on the final stage."""
    sched = tuple(d for d in SCHED_FULL if d < D)
    params = None; lls = eigs = None
    for dk in list(sched) + [D]:
        init = diag.pca_init(y, d=dk) if params is None else diag._expand_init(y, params, dk)
        params, lls, eigs = diag.run_track(y, init, nit, rf)
    return params, lls, eigs


def fit_identity(y, D, rf, nit):
    """EM from the identity initialisation. -> (params, lls, eigs)"""
    return diag.run_track(y, identity_init(y, D), nit, rf)
