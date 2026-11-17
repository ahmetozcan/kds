# KDS — Kalman-denoised stacking for TBM rate of penetration

Reference implementation and reproduction package for *Kalman-denoised stacking
for tunnel boring machine penetration rate prediction in the Bahce–Nurdagi
tunnel*.

KDS is a two-level ensemble. Five heterogeneous base learners produce
out-of-fold (OOF) predictions for every record of the drive; those predictions
are treated as noisy signals ordered along the excavation and denoised with a
**causal** Kalman filter before a second-level learner integrates them together
with the original features. The filtering step is the contribution: appending
the *raw* OOF series lowers accuracy, appending the *filtered* series raises it.

| Configuration | Integrator | R² | MAE | RMSE |
|---|---|---|---|---|
| 1. Best individual learner, no prediction signals | TabPFN | 0.8611 | 3.96 | 5.84 |
| 2. Ensemble with unfiltered prediction signals | TabPFN | 0.8566 | 3.99 | 5.93 |
| **3. Ensemble with Kalman-filtered signals (KDS)** | TabPFN | **0.8849** | **3.48** | **5.32** |

Errors in mm/min. Every row uses default hyperparameters, the shared-fold
construction and process noise *Q* = 0.1, over 5,334 records in five shuffled
folds.

Every number above, and every other figure in the paper, can be recomputed from
the artifacts in this repository without refitting a model:

```bash
uv sync
uv run python run/pipeline/paper_numbers.py
```

## Repository layout

```
src/tbm_rop/          the package: features, folds, Kalman filter, stacking, tuning
run/conf/config.yaml  the single configuration file; every stage reads it
run/pipeline/         analysis scripts (figures, significance tests, Q sweep, …)
tests/                unit tests for the filter, the folds, the features, tuning
data/raw/             the published dataset, as distributed by its authors
data/processed/       the feature tables and OOF predictions of the reported runs
outputs/              one JSON report per run: scores, selections, environment
artifacts/            fold assignment and selected hyperparameters, for reference
```

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/) and pinned by
`uv.lock`.

```bash
uv sync                 # runtime dependencies, Python 3.12
uv sync --extra dev     # plus pytest, ruff, mypy
uv run pytest
```

TabPFN downloads its pretrained checkpoint on first use. Version 7.x is pinned
deliberately: 8.x requires an API token for the weights.

## Reproducing the paper

### From the stored artifacts (seconds)

```bash
uv run python run/pipeline/paper_numbers.py        # every quoted accuracy figure
uv run python run/pipeline/significance_tests.py   # Table 6, shared folds
uv run python run/pipeline/significance_nested.py  # Section 5.3.4, nested columns
uv run python run/pipeline/make_figures.py         # Figures 2, 3, 4 and S1
uv run python run/pipeline/lithology_numbers.py    # Table S1, the lithological units
```

These read `data/processed/` and `outputs/` and refit nothing.

### From the raw data (about ten minutes per run)

Each line below writes its own processed table and its own report, so runs never
overwrite one another. The timings here and in the sections that follow are from
the machine that produced the reported runs: an Intel Core i9-14900HX (24 cores,
32 threads), 32 GB of RAM visible to the WSL2 guest, and an NVIDIA RTX 5000 Ada
Generation Laptop GPU (16 GB, CUDA 13.0). TabPFN and the MLP are placed on the
GPU when one is present and fall back to the CPU, which is slower; the tree
ensembles are CPU-bound either way.

```bash
# Configuration 3, the reported result
uv run tbm-rop all kalman.mode=causal kalman.process_noise=0.1 \
    paths.processed=data/processed/sig_a paths.outputs=outputs/sig_a

# Configuration 2, the same pipeline with the filter switched off
uv run tbm-rop all kalman.mode=none \
    paths.processed=data/processed/sig_a_off paths.outputs=outputs/sig_a_off

# Nested construction of the prediction columns (Table 9), ~7 min each
uv run tbm-rop nested kalman.mode=causal \
    paths.processed=data/processed/a paths.outputs=outputs/nested_a
uv run tbm-rop nested kalman.mode=none \
    paths.processed=data/processed/a_off paths.outputs=outputs/nested_a_off
```

Configuration 1 needs no separate run: a base learner never sees the augmented
table, so its score is the TabPFN entry of the `denoise` stage report.

### Hyperparameter optimization (Table 7, about 2.5 hours)

```bash
uv run tbm-rop tune paths.processed=data/processed/a paths.outputs=outputs/a
```

The stage runs two studies. The *nested* one reports honest out-of-fold scores
for every competitor, 30 Optuna trials inside each outer fold. The *flat* one
produces the single consensus configuration consumed by the stacking stage,
written to `outputs/a/tuned_params.json` and reproduced in `artifacts/
hyperparameters.json`. Its internal best value is selection-biased and is never
reported as a result.

### Process-noise sweep (Figure 4)

The sweep reuses one set of raw OOF series, so only the cheap stages repeat:

```bash
uv run tbm-rop prepare \
    paths.processed=data/processed/qs_raw paths.outputs=outputs/qs_raw
uv run tbm-rop denoise kalman.mode=none base_params_file=outputs/a/tuned_params.json \
    paths.processed=data/processed/qs_raw paths.outputs=outputs/qs_raw

for Q in 0.01 0.03 0.1 0.3 1.0 3.0; do
  uv run python run/pipeline/q_filter.py \
      --source data/processed/qs_raw/features_kalman.csv --q $Q --out data/processed/qs_$Q
  uv run tbm-rop stack stack_params_file=outputs/a/tuned_params.json \
      paths.processed=data/processed/qs_$Q paths.outputs=outputs/qs_$Q
done
```

Both levels of the sweep carry the consensus configurations, which is why its
cell at *Q* = 0.1 reads 0.8847 against the 0.8849 of the reported run, where
every learner keeps its library defaults.

## What is in `artifacts/`

- `folds.csv` — the validation fold of each of the 5,334 records, written by
  `run/pipeline/export_folds.py`. The folds are deterministic given seed 42,
  so this file is a convenience for checking a split or aligning an external
  re-implementation.
- `hyperparameters.json` — the configurations selected by the flat Optuna
  studies, consumed by the stacking stage and listed in Table 8 of the paper.

No fitted model binaries are distributed. Every learner is refit from the seed
in minutes, pickled estimators break across library versions, and a serialized
forest of 831 trees is larger than the entire dataset. What determines a result
here is the data, the seed, the folds and the hyperparameters, and all four are
in the repository.

## Reproducibility

- One seed, 42, set for Python, NumPy and Torch at the start of every stage.
- Every artifact is written with a `.meta.json` sidecar carrying its shape, a
  SHA-256 prefix and the stage that produced it.
- Every run report records the interpreter, the platform and the versions of
  NumPy, pandas, scikit-learn, LightGBM, CatBoost, Torch and TabPFN.
- Tree ensembles and the MLP reproduce bit-for-bit on the same platform.
  TabPFN is a transformer evaluated in float32 and may move in the fourth
  decimal across hardware, which is why every report carries the environment
  it was produced in. `tests/test_reproducibility.py` asserts that the seed
  switches cuDNN to deterministic kernels and that each report embeds that
  environment.
- Checking a number in the paper needs no refitting at all: the out-of-fold
  predictions of every configuration are in `data/processed/`, so each score
  is recomputed from the stored columns.

## Data

`data/raw/ENGEO_Gokceoglu_et_al_TBM_2022.xlsx` is the dataset released as
supplementary material of Bardhan et al. (2024),
[doi:10.1016/j.engappai.2024.108997](https://doi.org/10.1016/j.engappai.2024.108997),
covering 5,334 records of an 8 km TBM-driven section of the Bahce–Nurdagi twin
tunnels at 1.5 m spacing. It is redistributed here unmodified for
reproducibility; the rights remain with its authors and publisher. See
`data/README.md` for the column inventory and the two unit readings that the
paper resolves.

## Method in one page

1. **Features.** 53 columns from 9 raw variables: physics-informed indices
   (specific power, thrust-to-torque ratio, weathering-adjusted strength),
   causal temporal statistics over the preceding records, a ground-regime
   cluster label, and degree-2 interactions over five engineered variables.
2. **Level 1.** Random forest, LightGBM, CatBoost, an MLP and TabPFN each
   produce an OOF prediction series over the whole drive.
3. **Filter.** Each series is passed through a local-level Kalman filter in
   excavation order, with the observation noise of learner *m* set adaptively
   to its own mean squared error, estimated per fold on training records only.
   The filter is causal: record *t* uses records 1…*t* and no later one.
4. **Level 2.** The 53 features and the 5 denoised series form the augmented
   table. A tournament over the same five families selects the integrator.

## Citation

The manuscript is under review and carries no author names in this package.
A citable entry, with the DOI, is added here on acceptance.

## License

Code is released under the MIT License (`LICENSE`). The dataset under
`data/raw/` is not covered by that license and remains subject to the terms of
its original publication, which `NOTICE` states.
