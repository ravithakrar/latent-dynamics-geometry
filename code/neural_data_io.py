"""Load DANDI:000688 NWB sessions into binned spike tensors.

This is the *data* half of the real-data pipeline. It turns an NWB file into a
count tensor with the same layout the EM code already expects:

    counts : (N_trials, T_bins, U_units)   ==   ys : (N, T, M)   in lds_em_highD

The three jobs here are:
    1. split units by brain region (M1 vs PMd) using the electrodes table,
    2. epoch the continuous recording into trials aligned to an event,
    3. bin each unit's spike_times inside each trial window.

Everything downstream (smoothing, z-scoring, PCA init, EM) operates on the
tensor produced by `binned_counts`.

Not every subject in the dandiset carries both arrays, so which regions a session
yields has to be read from its electrodes table rather than assumed.

Example:
    from neural_data_io import load_session, binned_counts
    sess = load_session("data/dandi/000688/sub-C/"
                        "sub-C_ses-CO-20160914_behavior+ecephys.nwb")
    counts, info = binned_counts(sess, region="M1")   # (n_trials, T, n_units)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pynwb import NWBHDF5IO


# region name -> the substring that appears in the electrodes `location` column
REGION_LOCATIONS = {
    "M1": "Primary Motor Cortex",
    "PMd": "Dorsal Premotor Cortex",
}


@dataclass
class Session:
    """A minimal, in-memory view of one NWB session.

    Attributes:
        spike_times: list (len = n_units) of 1-D np.ndarrays of spike times (s).
        unit_region: (n_units,) array of region labels, e.g. "M1" / "PMd".
        trials:      dict of column_name -> np.ndarray (one row per trial).
        speed:       (n_samples,) cursor speed = |velocity|, or None.
        speed_t:     (n_samples,) timestamps for `speed`, or None.
        path:        source file path.
    """
    spike_times: list[np.ndarray]
    unit_region: np.ndarray
    trials: dict[str, np.ndarray]
    speed: np.ndarray | None
    speed_t: np.ndarray | None
    path: str


def _unit_region_labels(units: Any) -> np.ndarray:
    """Map each unit to a region string via its electrodes' `location` column.

    A unit that sits on electrodes from a single location is labelled with the
    matching short name ("M1"/"PMd"); anything else is "other"/"unknown".
    """
    labels: list[str] = []
    for i in range(len(units)):
        electrodes = units["electrodes"][i]
        location = "unknown"
        if hasattr(electrodes, "empty") and not electrodes.empty:
            locs = sorted(set(np.asarray(electrodes["location"]).astype(str)))
            if len(locs) == 1:
                loc = locs[0]
                location = next(
                    (short for short, full in REGION_LOCATIONS.items() if full == loc),
                    loc,  # fall back to the raw location string
                )
            else:
                location = "mixed"
        labels.append(location)
    return np.asarray(labels)


def load_session(path: str | Path) -> Session:
    """Read an NWB file into a `Session` (spike times, unit regions, trials)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with NWBHDF5IO(path, mode="r", load_namespaces=True) as io:
        nwb = io.read()

        units = nwb.units
        spike_times = [np.asarray(units["spike_times"][i], dtype=float)
                       for i in range(len(units))]
        unit_region = _unit_region_labels(units)

        trials_tbl = nwb.trials
        trials = {col: np.asarray(trials_tbl[col][:]) for col in trials_tbl.colnames}

        # cursor speed (for movement-onset alignment), if a Velocity interface exists
        speed = speed_t = None
        if nwb.processing and "behavior" in nwb.processing:
            interfaces = nwb.processing["behavior"].data_interfaces
            if "Velocity" in interfaces:
                series = interfaces["Velocity"].time_series
                vel = series.get("cursor_vel", next(iter(series.values())))
                ts = getattr(vel, "timestamps", None)
                if ts is not None:
                    v = np.asarray(vel.data[:], dtype=float)
                    speed = np.linalg.norm(v, axis=1) if v.ndim == 2 else np.abs(v)
                    speed_t = np.asarray(ts[:], dtype=float)

    return Session(
        spike_times=spike_times,
        unit_region=unit_region,
        trials=trials,
        speed=speed,
        speed_t=speed_t,
        path=str(path),
    )


def movement_onset_times(sess: Session, speed_frac: float = 0.2,
                         from_col: str = "go_cue_time",
                         stop_col: str = "stop_time") -> np.ndarray:
    """Per-trial movement-onset times (absolute, seconds).

    For each trial, onset is the first moment in [go_cue, stop] at which cursor
    speed exceeds `speed_frac` x (99th-percentile speed of the whole session).
    Trials with no detectable onset (or missing bounds) return NaN.

    Returns:
        (n_trials,) array of onset times, aligned to the trials table.
    """
    if sess.speed is None:
        raise ValueError("session has no cursor velocity; cannot detect movement onset")
    go = np.asarray(sess.trials[from_col], dtype=float)
    stop = np.asarray(sess.trials[stop_col], dtype=float)
    thr = speed_frac * np.nanpercentile(sess.speed, 99)

    onsets = np.full(go.shape, np.nan)
    for i, (g, s) in enumerate(zip(go, stop)):
        if not (np.isfinite(g) and np.isfinite(s)):
            continue
        w = (sess.speed_t >= g) & (sess.speed_t <= s)
        idx = np.flatnonzero(w)
        crossed = idx[sess.speed[idx] > thr]
        if crossed.size:
            onsets[i] = sess.speed_t[crossed[0]]
    return onsets


def region_unit_indices(sess: Session, region: str) -> np.ndarray:
    """Indices of units belonging to `region` (e.g. "M1", "PMd")."""
    return np.flatnonzero(sess.unit_region == region)


def select_trials(
    sess: Session,
    align: str = "go_cue_time",
    result_keep: str | None = "R",
) -> np.ndarray:
    """Row indices of trials to keep.

    Keeps successful trials (`result == result_keep`, default "R") that have a
    finite alignment time. Pass `result_keep=None` to keep all results.
    """
    n = len(sess.trials[align])
    mask = np.isfinite(np.asarray(sess.trials[align], dtype=float))
    if result_keep is not None and "result" in sess.trials:
        mask &= np.asarray(sess.trials["result"]).astype(str) == result_keep
    return np.flatnonzero(mask[:n] if mask.shape[0] >= n else mask)


def binned_counts(
    sess: Session,
    region: str,
    align: str = "go_cue_time",
    event_times: np.ndarray | None = None,
    t_start: float = -0.2,
    t_stop: float = 0.5,
    bin_size: float = 0.02,
    result_keep: str | None = "R",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Bin spikes into a (n_trials, T_bins, n_units) count tensor for one region.

    Each kept trial defines a window [event + t_start, event + t_stop) around the
    alignment event; spikes from each selected unit are histogrammed into fixed
    `bin_size` bins inside that window.

    Args:
        sess:        loaded Session.
        region:      "M1" or "PMd".
        align:       trials column to align to (default "go_cue_time"). Ignored
                     if `event_times` is given.
        event_times: optional (n_trials,) absolute event times to align to, e.g.
                     from `movement_onset_times`. Trials with NaN are dropped.
        t_start:     window start relative to the event, seconds.
        t_stop:      window end relative to the event, seconds.
        bin_size:    bin width, seconds.
        result_keep: trial result to keep ("R" = rewarded/successful).

    Returns:
        counts: (n_trials, T_bins, n_units) integer counts.
        info:   dict with unit indices, trial indices, bin edges, params.
    """
    unit_idx = region_unit_indices(sess, region)
    if unit_idx.size == 0:
        raise ValueError(f"No units found for region {region!r} in {sess.path}")

    # alignment vector: explicit event_times override the named column
    if event_times is not None:
        align_vec = np.asarray(event_times, dtype=float)
        align_label = "movement_onset"
    else:
        align_vec = np.asarray(sess.trials[align], dtype=float)
        align_label = align

    mask = np.isfinite(align_vec)
    if result_keep is not None and "result" in sess.trials:
        mask &= np.asarray(sess.trials["result"]).astype(str) == result_keep
    trial_idx = np.flatnonzero(mask)
    if trial_idx.size == 0:
        raise ValueError(f"No trials kept for align={align_label!r} result={result_keep!r}")

    # shared bin edges relative to the alignment event (same length for every trial)
    edges = np.arange(t_start, t_stop + 1e-9, bin_size)
    T = edges.size - 1
    events = align_vec[trial_idx]

    counts = np.zeros((trial_idx.size, T, unit_idx.size), dtype=np.int32)
    for ti, event in enumerate(events):
        lo, hi = event + t_start, event + t_stop
        for ui, u in enumerate(unit_idx):
            st = sess.spike_times[u]
            st = st[(st >= lo) & (st < hi)]
            if st.size:
                counts[ti, :, ui] = np.histogram(st - event, bins=edges)[0]

    info = {
        "region": region,
        "unit_idx": unit_idx,
        "trial_idx": trial_idx,
        "bin_edges": edges,
        "bin_centers": 0.5 * (edges[:-1] + edges[1:]),
        "bin_size": bin_size,
        "align": align_label,
        "t_start": t_start,
        "t_stop": t_stop,
        "target_dir": (np.asarray(sess.trials["target_dir"], dtype=float)[trial_idx]
                       if "target_dir" in sess.trials else None),
    }
    return counts, info


def available_regions(sess: Session) -> dict[str, int]:
    """Map region label -> unit count for this session."""
    labels, counts = np.unique(sess.unit_region, return_counts=True)
    return {str(k): int(v) for k, v in zip(labels, counts)}
