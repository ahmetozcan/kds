"""Pipeline stages: prepare -> denoise -> stack.

Each stage reads/writes artifacts under ``cfg.paths`` so intermediate outputs
are inspectable and the pipeline can resume at any point.
"""

import json
import logging
import re
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import mean_squared_error

from .cv import CVConfig, make_folds
from .data import TARGET_COLUMN, load_raw
from .evaluation import regression_metrics, save_report
from .features import build_features
from .kalman import denoise_oof_foldwise
from .nested import nested_stack_evaluation
from .oof import generate_oof_predictions
from .stacking import build_meta_features, run_meta_tournament
from .tuning import (
    TuningConfig,
    load_tuned_params,
    merge_model_params,
    nested_oof_predictions,
    tune_on_folds,
)
from .utils import save_artifact, set_seed

logger = logging.getLogger(__name__)

FEATURES_FILE = "features.csv"
DENOISED_FILE = "features_kalman.csv"


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make column names safe for LightGBM (alphanumerics + underscore)."""
    df = df.copy()
    df.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", str(c)).strip("_") for c in df.columns]
    return df


def _split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def _model_params(cfg_section: DictConfig) -> dict[str, dict]:
    """Convert an OmegaConf model section into plain name -> params dicts."""
    section = OmegaConf.to_container(cfg_section, resolve=True) or {}
    return {name: (params or {}) for name, params in section.items()}


def stage_prepare(cfg: DictConfig) -> Path:
    """Load raw data and build the engineered feature table."""
    set_seed(cfg.seed)
    df = load_raw(cfg.paths.raw)

    steps = list(cfg.features.steps)
    step_params = OmegaConf.to_container(cfg.features.get("step_params", {}), resolve=True) or {}
    df = build_features(df, steps=steps, step_params=step_params)

    if cfg.features.include_row_index:
        # Chainage-position proxy. Off by default: with shuffled CV it leaks
        # neighbourhood target information. Kept as an option for replicating
        # the original pipeline and for the leakage ablation.
        df["row_index"] = range(len(df))

    df = _sanitize_columns(df)
    out = Path(cfg.paths.processed) / FEATURES_FILE
    return save_artifact(
        df,
        out,
        meta={
            "stage": "prepare",
            "source": str(cfg.paths.raw),
            "feature_steps": steps,
            "include_row_index": bool(cfg.features.include_row_index),
            "seed": int(cfg.seed),
        },
    )


def stage_denoise(cfg: DictConfig) -> Path:
    """Add Kalman-denoised OOF prediction features for each base model."""
    set_seed(cfg.seed)
    df = pd.read_csv(Path(cfg.paths.processed) / FEATURES_FILE)
    X, y = _split_xy(df)

    folds = make_folds(len(df), CVConfig(**cfg.cv))
    base_models = _model_params(cfg.base_models)
    if cfg.get("base_params_file"):
        tuned = load_tuned_params(cfg.base_params_file)
        base_models = merge_model_params(base_models, tuned)
        logger.info("Merged tuned params from %s into denoise base models: %s",
                    cfg.base_params_file, sorted(tuned))

    scores: dict[str, dict] = {}
    for name, params in base_models.items():
        logger.info("=== Base model '%s' ===", name)
        oof = generate_oof_predictions(X, y, name, params, folds)
        raw_metrics = regression_metrics(y, oof)

        # Observation noise is estimated per fold from training rows only, so
        # that a row's own target never reaches the filter applied to it.
        denoised, r_per_fold = denoise_oof_foldwise(
            oof,
            y.to_numpy(dtype=float),
            folds,
            process_noise=cfg.kalman.process_noise,
            mode=cfg.kalman.mode,
        )
        df[f"rop_kf_{name}"] = denoised

        scores[name] = {
            "raw_oof": raw_metrics,
            "kalman": regression_metrics(y, denoised),
            "observation_noise_per_fold": r_per_fold,
            "observation_noise_full_data": float(mean_squared_error(y, oof)),
        }
        logger.info(
            "%s: raw R2=%.4f, kalman(%s) R2=%.4f",
            name,
            raw_metrics["r2"],
            cfg.kalman.mode,
            scores[name]["kalman"]["r2"],
        )

    out = Path(cfg.paths.processed) / DENOISED_FILE
    save_artifact(
        df,
        out,
        meta={
            "stage": "denoise",
            "cv": OmegaConf.to_container(cfg.cv),
            "kalman_mode": str(cfg.kalman.mode),
            "seed": int(cfg.seed),
        },
    )
    save_report(
        {"stage": "denoise", "cv": OmegaConf.to_container(cfg.cv),
         "kalman_mode": str(cfg.kalman.mode), "scores": scores},
        Path(cfg.paths.outputs) / "denoise_report.json",
    )
    return out


def stage_tune(cfg: DictConfig) -> Path:
    """Optuna tuning on the denoised feature table.

    Two protocols per model:
    - nested: per-outer-fold selection on inner folds -> honest OOF metrics;
    - flat: one study on the standard folds -> consensus params for the stack
      stage (best value is selection-biased and labelled as such).
    """
    set_seed(cfg.seed)
    df = pd.read_csv(Path(cfg.paths.processed) / DENOISED_FILE)
    X, y = _split_xy(df)
    X_arr, y_arr = X.to_numpy(dtype=float), y.to_numpy(dtype=float)

    cv_cfg = CVConfig(**cfg.cv)
    folds = make_folds(len(df), cv_cfg)
    tune_cfg = TuningConfig(
        n_trials=int(cfg.tuning.n_trials),
        inner_splits=int(cfg.tuning.inner_splits),
        seed=int(cfg.seed),
    )

    report: dict[str, dict] = {}
    consensus: dict[str, dict] = {}
    for name in cfg.tuning.models:
        logger.info("=== Tuning '%s' (nested) ===", name)
        nested = nested_oof_predictions(X_arr, y_arr, name, folds, cv_cfg, tune_cfg)
        if pd.isna(nested.oof).any():
            raise ValueError(f"Nested OOF for {name!r} left rows unpredicted.")
        nested_metrics = regression_metrics(y, nested.oof)

        logger.info("=== Tuning '%s' (flat, for stack params) ===", name)
        flat_params, flat_value = tune_on_folds(X_arr, y_arr, name, folds, tune_cfg)
        consensus[name] = flat_params

        report[name] = {
            "nested_oof": nested_metrics,
            "nested_fold_params": nested.fold_params,
            "nested_fold_inner_r2": nested.fold_inner_scores,
            "flat_best_params": flat_params,
            "flat_best_r2_biased": flat_value,
        }
        logger.info("%s tuned: nested OOF R2=%.4f (honest), flat best R2=%.4f (biased)",
                    name, nested_metrics["r2"], flat_value)

    params_path = Path(cfg.paths.outputs) / "tuned_params.json"
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.write_text(json.dumps(consensus, indent=2, default=str))
    save_report(
        {"stage": "tune", "cv": OmegaConf.to_container(cfg.cv),
         "kalman_mode": str(cfg.kalman.mode),
         "tuning": {"n_trials": tune_cfg.n_trials, "inner_splits": tune_cfg.inner_splits},
         "models": report},
        Path(cfg.paths.outputs) / "tuning_report.json",
    )
    return params_path


def stage_stack(cfg: DictConfig) -> Path:
    """Run the two-level stacking tournament on the denoised feature table."""
    set_seed(cfg.seed)
    df = pd.read_csv(Path(cfg.paths.processed) / DENOISED_FILE)
    X, y = _split_xy(df)

    folds = make_folds(len(df), CVConfig(**cfg.cv))
    base_models = _model_params(cfg.stack_base_models)
    if cfg.get("stack_params_file"):
        tuned = load_tuned_params(cfg.stack_params_file)
        base_models = merge_model_params(base_models, tuned)
        logger.info("Merged tuned params from %s for models: %s",
                    cfg.stack_params_file, sorted(tuned))
    meta_models = _model_params(cfg.meta_models)

    meta_features = build_meta_features(X, y, base_models, folds)
    individual = {
        name: regression_metrics(y, meta_features[name]) for name in meta_features.columns
    }
    tournament = run_meta_tournament(meta_features, y, meta_models, folds)

    ranking = sorted(
        [(f"stacking_meta_{k}", v) for k, v in tournament.items()]
        + [(k, v["r2"]) for k, v in individual.items()],
        key=lambda item: item[1],
        reverse=True,
    )

    report_path = save_report(
        {
            "stage": "stack",
            "cv": OmegaConf.to_container(cfg.cv),
            "kalman_mode": str(cfg.kalman.mode),
            "individual_base_models": individual,
            "meta_tournament_r2": tournament,
            "ranking": ranking,
            "champion": {"name": ranking[0][0], "r2": ranking[0][1]},
        },
        Path(cfg.paths.outputs) / "stack_report.json",
    )
    save_artifact(
        meta_features.assign(**{TARGET_COLUMN: y}),
        Path(cfg.paths.processed) / "meta_features.csv",
        meta={"stage": "stack", "seed": int(cfg.seed)},
    )
    return report_path



def stage_nested(cfg: DictConfig) -> Path:
    """Strictly nested evaluation of the augmented table (see ``nested.py``).

    The augmented table is rebuilt inside each outer fold from the
    outer-training rows only, so the level-2 learners are scored without any
    path from a validation target back into their training features. Reported
    alongside the standard ``stack`` metrics as the stricter axis.
    """
    set_seed(cfg.seed)
    df = pd.read_csv(Path(cfg.paths.processed) / FEATURES_FILE)
    X, y = _split_xy(df)

    cv_cfg = CVConfig(**cfg.cv)
    folds = make_folds(len(df), cv_cfg)

    base_models = _model_params(cfg.base_models)
    if cfg.get("base_params_file"):
        base_models = merge_model_params(base_models, load_tuned_params(cfg.base_params_file))
    super_models = _model_params(cfg.stack_base_models)
    if cfg.get("stack_params_file"):
        super_models = merge_model_params(super_models, load_tuned_params(cfg.stack_params_file))

    inner_splits = int(cfg.nested.inner_splits) if cfg.get("nested") else 5

    result = nested_stack_evaluation(
        X,
        y,
        base_models=base_models,
        super_models=super_models,
        outer_folds=folds,
        cv_cfg=cv_cfg,
        inner_splits=inner_splits,
        process_noise=float(cfg.kalman.process_noise),
        mode=str(cfg.kalman.mode),
    )

    metrics = {
        name: regression_metrics(y, preds) for name, preds in result.predictions.items()
    }
    ranking = sorted(
        ((name, m["r2"]) for name, m in metrics.items()), key=lambda item: item[1], reverse=True
    )

    save_artifact(
        pd.DataFrame(result.predictions).assign(**{TARGET_COLUMN: y}),
        Path(cfg.paths.processed) / "nested_predictions.csv",
        meta={"stage": "nested", "seed": int(cfg.seed)},
    )
    return save_report(
        {
            "stage": "nested",
            "cv": OmegaConf.to_container(cfg.cv),
            "kalman_mode": str(cfg.kalman.mode),
            "inner_splits": inner_splits,
            "level2_nested": metrics,
            "ranking": ranking,
            "champion": {"name": ranking[0][0], "r2": ranking[0][1]},
            "observation_noise_per_outer_fold": result.observation_noise,
        },
        Path(cfg.paths.outputs) / "nested_report.json",
    )


STAGES = {
    "prepare": stage_prepare,
    "denoise": stage_denoise,
    "tune": stage_tune,
    "stack": stage_stack,
    "nested": stage_nested,
}

# "all" deliberately excludes the expensive tune stage; run it explicitly.
DEFAULT_STAGES = ("prepare", "denoise", "stack")
