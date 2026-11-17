# Data

## `raw/ENGEO_Gokceoglu_et_al_TBM_2022.xlsx`

5,334 records of the 8 km TBM-driven section of the Bahce–Nurdagi twin
tunnels, one record per 1.5 m of advance, in excavation order. Released as
supplementary material of Bardhan et al. (2024),
[doi:10.1016/j.engappai.2024.108997](https://doi.org/10.1016/j.engappai.2024.108997),
and redistributed here unmodified.

| Column | Meaning |
|---|---|
| `ROP (mm/m)` | rate of penetration, the target |
| `RPM (rotate per minute)` | cutterhead rotation speed |
| `CHD (kNm)` | cutterhead torque |
| `THRUST (kN)` | total thrust |
| `ALFA (degree)` | angle between the tunnel axis and the discontinuity planes |
| `WatInflow (l/m)` | water inflow |
| `WEATHERING` | weathering degree |
| `CAI` | Cerchar abrasivity index |
| `UCS (MPa)` | uniaxial compressive strength |
| `UCS_Weat` | strength divided by weathering degree |

Two column headings disagree with the text of the source publications, and the
paper resolves both in favour of the text and the descriptive tables. The
target is in **mm/min**, not mm/m: the headings say mm/m while both source
papers quote the same range in mm/min. Water inflow is in **l/s**, not l/m:
the headings say l/m while both papers quote inflows of up to 50 l/s. Neither
reading changes a model or a result, since both are monotone relabelings of the
same column.

The file names no lithological unit, but UCS takes exactly eight distinct
values, one per unit of the site investigation, each occupying contiguous runs
of records. Table S1 of the paper's supplementary sheet gives the mapping.

## `processed/`

One directory per run. Every CSV is written with a `.meta.json` sidecar
carrying its shape, a SHA-256 prefix and the stage that produced it.

| Directory | What it holds |
|---|---|
| `sig_a/` | the reported configuration: 53-column feature table, the same table with five Kalman-filtered prediction series appended, and the level-2 predictions of the tournament |
| `sig_a_off/` | the same pipeline with the filter switched off, which is configuration 2 of Table 4 and the source of the single-learner scores of configuration 1 |
| `a/`, `a_off/` | per-record predictions of the strictly nested construction, read by `run/pipeline/significance_nested.py` |
| `qs_raw/` | the unfiltered prediction series of the sweep, produced once with the consensus hyperparameters |
| `qs_<Q>/` | level-2 predictions after refiltering `qs_raw` with that process noise |

File roles inside a run directory:

- `features.csv` — 53 engineered features and the target, in excavation order.
  The 9 raw variables are joined by 12 physics-informed indices, 16 causal
  temporal statistics, 1 ground-regime cluster label and 15 degree-2
  interactions.
- `features_kalman.csv` — the same table plus one column per base learner,
  `rop_kf_<learner>`, holding that learner's out-of-fold prediction series
  after filtering. With `kalman.mode=none` the columns hold the raw series.
- `meta_features.csv` — the out-of-fold prediction of every level-2 candidate
  over all 5,334 records, together with the target.
- `nested_predictions.csv` — the same, for the strictly nested construction in
  which each fold's prediction columns are rebuilt from outer-training records
  alone.
