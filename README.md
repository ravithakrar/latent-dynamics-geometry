# Inferring the geometry of latent dynamics from neural time series

Code and cached results for an MSc dissertation on the **pullback metric** induced on a
task domain by a fitted latent linear dynamical system, applied to motor cortex (M1) and
dorsal premotor cortex (PMd) recordings during a centre-out reaching task.

The idea in one line: fit a linear dynamical system to the population activity, treat the
map from (time, reach direction) to latent state as a chart, and pull the ambient metric
back through it. What you get is a 2x2 metric field `g(t, theta)` over the task domain,
and its structure — how the direction code contracts through the reach, how stable it is
across sessions and across years — is what the figures below measure.

Every figure in the dissertation is reproduced here from cached intermediates. **No
refitting and no data download is required to redraw them**.

## Data

The recordings are public: **[DANDI:000688](https://dandiarchive.org/dandiset/000688)**,
*Long-term recordings of motor and premotor cortical spiking activity during reaching in
monkeys*, Perich, Miller, Azabou & Dyer. This work uses the centre-out sessions from
monkeys C and M.
The raw NWB files are ~12 GB and are not in this repository. What is here instead are the
fitted models and derived quantities (`*.npz`, `*.json`, `*.jsonl` in `code/`, 37 MB
total), which is everything the figures actually read.

## Running it

```bash
pip install -r requirements.txt
cd code
python3 mk44_msa.py --null units        # writes figures/fig44_msa.{pdf,png}
```

Every figure script is run **from `code/`** and writes into `code/figures/`.

## The figures

| Figure | Command (from `code/`) | What it shows |
|---|---|---|
| `fig_task_behavior` | `python3 make_task_behavior.py` | the task, trial timing, and that the animal's reaching behaviour is unchanged over three years |
| `fig_demo_*` | `python3 demo_synthetic.py` | EM recovers a known linear dynamical system: likelihood, parameter recovery, smoothed latents |
| `fig_pullback_schematic` | `python3 pullback_schematic.py` | the construction itself: task domain, latent manifold, observation manifold, and the metric ellipses induced on the domain |
| `fig_bcv_blocks` | `python3 mk_bcv_blocks.py` | the held-out blocks used by bi-cross-validated PCA |
| `fig_smoothing_spectrum` | `python3 mk_smoothing_spectrum.py` | how temporal smoothing inflates the apparent dimensionality of the population |
| `fig_metric_recovery` | `python3 plot_recovery_v2.py` | the pipeline recovers a known metric, and the recovery error is minimised at the true latent dimension |
| `fig_dimension_selection` | `python3 make_dimension_selection.py` | why the operating dimension is 8 in M1 and 12 in PMd (co-smoothing knee and bi-cross-validated PCA) |
| `fig42_ring_trajectories` | `python3 mk_ring_trajectories.py` | the fitted latents: trajectories fan out by reach direction from a direction-ordered ring of initial states |
| `fig42_eigenspectrum` | `python3 mk42_eigenspectrum.py` | where the fitted dynamics land over EM iterations, from two initialisations |
| `fig42_init_floor` | `python3 mk_init_floor.py` | how much of the fit is determined by the recording and how much by the initialisation |
| `fig43_metric_matrix` | `python3 make_metric_matrix.py --regions M1 --out figures/fig43_metric_matrix` | the metric matrix itself at six points of the (time x direction) cylinder |
| `fig43_metric_entries` | `python3 mk_metric_entries.py` | the three entries of `g` through the reach, monkey C |
| `fig43_metric_entries_subm` | `python3 mk_metric_entries_subm.py` | the same for monkey M |
| `fig43_circumference`, `fig43_mds` | `python3 mk43_geodesic.py --animal subC` | the direction ring measured in the pullback metric contracts through the reach |
| `fig43_circumference_subm`, `fig43_mds_subm` | `python3 mk43_geodesic.py --animal subM` | the same for monkey M |
| `fig43_L_constructions` | `python3 mk_L_constructions.py` | the contraction is not an artefact of how the tangent is built or of the smoothing parameter |
| `fig43_resampling` | `python3 mk43_resampling.py` | it survives a trial bootstrap and disjoint halves of the recorded units |
| `fig44_msa` | `python3 mk44_msa.py --null units` | metric similarity between sessions, against a disjoint-neuron null |
| `fig44_msa_trials` | `python3 mk44_msa.py --null trials` | the same, against a split-trial null |
| `fig44_msa_subm` | `python3 mk44_msa.py --subject subM` | the same for monkey M |
| `fig44_slopes` | `python3 mk44_slopes.py` | how fast the between-minus-within excess grows with days apart, with a delete-one-session jackknife error bar |
| `fig45_levels`, `fig45_levels_trials`, `fig45_drift` | `python3 mk45_multiyear.py` | M1 over three years and 1,326 session pairs, with unit count and trial count matched |

## Rebuilding the caches

Not needed for any figure, and slow. The builders are included as the record of how the
cached results were made:

| Cache | Built by | Cost |
|---|---|---|
| `msa_fields.npz`, `subm_fields.npz` | `msa_cache.py`, `build_subm_fields.py` | EM fits, minutes per session |
| `ring_fields_push.npz` | `build_ring_fields.py` | seconds, from `msa_fields.npz` |
| `msa_allpairs_*_summary.json` | `run_msa_dist_allpairs.py` | hours: every session pair x many seeds |
| `resample43.jsonl` | `resample43.py` | hours: 100 bootstrap refits + 20 half-arrays per session |
| `metric_recovery_multi_*.json` | `run_metric_recovery_multi.py` | hours: simulate from a fit, refit, sweep the dimension |
| `eigtrack_20160914.npz` | `run_eigtrack.py` | minutes |

The per-session spike tensors those builders read (`y_tensors.npz`, `tensor_<session>.npz`)
are rebuilt from the NWB files by `neural_data_io.py` and `neural_preprocess.py`; only
`tensor_20160914.npz`, which a figure needs directly, is included here.

These need `pip install -r requirements-refit.txt` (adds JAX, scikit-learn, pynwb) and the
raw NWB files from DANDI.

## Layout

Everything lives in one directory, `code/`, because the scripts read their caches from the
working directory. `code/figures/` is where output lands.

- **figure scripts** — `mk*.py`, `make_*.py`, `plot_recovery_v2.py`, `pullback_schematic.py`,
  `demo_synthetic.py`
- **the method** — `pullback_metric.py` (the metric), `metric_convention.py` (the single
  place fixing how it is built), `geodesic_ring.py` (arc length on the direction ring),
  `metric_compare.py` (the between-metric distance)
- **fitting** — `fitting.py` (the shared entry points), `lds_em_highD.py`,
  `lds_em_weighted.py`, `neural_pca_init.py`, `fit_cache.py`, `diag.py`
- **analysis runners** — `run_msa*.py`, `resample43.py`, `run_metric_recovery*.py`,
  `run_dt_regression.py`
- **cached results** — the `*.npz`, `*.json` and `*.jsonl` files

The dissertation itself is not in this repository.

## Licence

MIT, see `LICENSE`. The recordings are from DANDI:000688 and carry their own terms.
