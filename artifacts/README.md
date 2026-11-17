# Artifacts

Small files that pin down the two things a reader most often needs to check
against an own implementation: which records were in which fold, and which
hyperparameters the tuned models carried.

## `folds.csv`

One row per record, in excavation order.

| Column | Meaning |
|---|---|
| `record` | position in the released file, 0 to 5333 |
| `fold` | validation fold, `KFold(5, shuffle=True, random_state=42)` |

The folds hold 1,067, 1,067, 1,067, 1,067 and 1,066 records. They are
deterministic given the seed, so this file duplicates what the code produces;
it exists so that a split can be inspected, or matched, without running the
pipeline. Regenerate it with:

```bash
uv run python run/pipeline/export_folds.py
```

## `hyperparameters.json`

The configurations selected by the flat Optuna studies, 30 trials per model
over the search spaces of Table 3 of the paper. They are consumed by the
stacking stage when `stack_params_file` or `base_params_file` points at them,
and they are the configurations behind the process-noise sweep. A copy lives at
`outputs/a/tuned_params.json`, which is where the tuning stage writes it;
`outputs/a/tuning_report.json` holds the per-trial record.

TabPFN has no search space and is used with its defaults by design, so it does
not appear in the file.

## Why there are no model binaries

A result here is determined by the data, the seed, the folds and the
hyperparameters, and all four are in this repository. Fitted estimators are
not, for three reasons: they are reproduced in minutes from the artifacts
above, pickled scikit-learn and boosting objects fail to load across library
versions, and a serialized forest of 831 trees is larger than the dataset it
was fitted on. TabPFN carries no fitted weights of ours at all, being a
pretrained model used through in-context learning.
