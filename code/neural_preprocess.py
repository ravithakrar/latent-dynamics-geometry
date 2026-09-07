"""Turn binned spike counts into Gaussian-LDS-ready observations.

    counts --(smooth)--> --(/bin -> rate)--> --(soft per-unit z-score)--> y

    1. Gaussian smoothing along time, per trial and per unit.
    2. Divide by the bin width, giving a firing rate in Hz. This cancels under the
       z-score and does not change the fit.
    3. Soft per-unit z-score, (r - mean) / (std + softening), so that no one unit
       dominates C and near-silent units are not amplified by a tiny std.

anscombe=True inserts a variance-stabilising 2*sqrt(n + 3/8) before the smoothing.

Fit the statistics on train trials with `fit_stats`, then `apply` to any split, so held-out
data is transformed with train-derived means and stds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass
class NormStats:
    """Per-unit normalisation constants, estimated on the training split."""
    mean: np.ndarray   # (n_units,)
    std: np.ndarray    # (n_units,)
    softening: float


def anscombe(counts: np.ndarray) -> np.ndarray:
    """Anscombe transform 2*sqrt(x + 3/8): stabilises Poisson variance to ~1."""
    return 2.0 * np.sqrt(counts + 0.375)


def gaussian_smooth(counts: np.ndarray, bin_size: float, sigma: float) -> np.ndarray:
    """Smooth along the time axis with a Gaussian kernel.

    Args:
        counts:   (n_trials, T, n_units).
        bin_size: bin width in seconds.
        sigma:    kernel width in seconds (converted to bins internally).

    The convolution is applied per trial and per unit (axis=1 only), never
    mixing neurons or bleeding across trial boundaries. `mode="reflect"` keeps
    the trial edges from being dragged toward zero.
    """
    if sigma <= 0:
        return counts.astype(float)
    sigma_bins = sigma / bin_size
    return gaussian_filter1d(counts.astype(float), sigma=sigma_bins,
                             axis=1, mode="reflect")


def to_rate(smoothed_counts: np.ndarray, bin_size: float) -> np.ndarray:
    """Convert smoothed counts to firing rate in Hz (divide by bin width)."""
    return smoothed_counts / bin_size


def fit_stats(rates: np.ndarray, softening_hz: float = 5.0) -> NormStats:
    """Estimate per-unit mean/std of the rates across trials and time.

    `softening_hz` is added to each unit's std so quiet neurons are not amplified.
    """
    flat = rates.reshape(-1, rates.shape[-1])          # (n_trials*T, n_units)
    return NormStats(
        mean=flat.mean(axis=0),
        std=flat.std(axis=0),
        softening=softening_hz,
    )


def apply(rates: np.ndarray, stats: NormStats) -> np.ndarray:
    """Soft per-unit z-score using pre-fitted stats: (r - mean)/(std + soft)."""
    return (rates - stats.mean) / (stats.std + stats.softening)


def preprocess(
    counts: np.ndarray,
    bin_size: float,
    sigma: float = 0.04,
    softening_hz: float = 5.0,
    anscombe_transform: bool = False,
    stats: NormStats | None = None,
) -> tuple[np.ndarray, NormStats]:
    """Full pipeline: (optional sqrt) -> smooth -> rate -> soft z-score.

    Args:
        counts:             (n_trials, T, n_units) integer counts.
        bin_size:           bin width, seconds.
        sigma:              Gaussian kernel width, seconds (default 40 ms).
        softening_hz:       soft-normalisation constant for the z-score.
        anscombe_transform: if True, apply 2*sqrt(x + 3/8) before smoothing.
        stats:              if given, reuse these (e.g. train stats on test data);
                            otherwise fit fresh stats on `counts`.

    Returns:
        y:     (n_trials, T, n_units) float observations for the LDS.
        stats: the NormStats used (fit here unless supplied).
    """
    x = anscombe(counts) if anscombe_transform else counts.astype(float)
    smoothed = gaussian_smooth(x, bin_size=bin_size, sigma=sigma)
    rates = to_rate(smoothed, bin_size=bin_size)
    if stats is None:
        stats = fit_stats(rates, softening_hz=softening_hz)
    return apply(rates, stats), stats
