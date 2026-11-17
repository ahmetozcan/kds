"""Tests for the strictly nested evaluation of the augmented table."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from tbm_rop.cv import CVConfig, make_folds
from tbm_rop.models import create_model, register_model
from tbm_rop.nested import nested_stack_evaluation
from tbm_rop.stacking import build_meta_features

# A cheap deterministic learner: unlike the production models it costs
# microseconds, and because it uses X its predictions react to any change in
# the training targets, which is what the leakage tests probe.
register_model("linreg")(lambda params: LinearRegression(**(params or {})))


def _toy(n: int = 120, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)}
    )
    y = pd.Series(2.0 * X["a"] - X["b"] + rng.normal(scale=0.3, size=n), name="rop")
    return X, y


def _run_nested(X: pd.DataFrame, y: pd.Series, mode: str = "causal") -> dict[str, np.ndarray]:
    cv_cfg = CVConfig(scheme="shuffled", n_splits=4, random_state=42)
    folds = make_folds(len(X), cv_cfg)
    result = nested_stack_evaluation(
        X,
        y,
        base_models={"linreg": {}},
        super_models={"linreg": {}},
        outer_folds=folds,
        cv_cfg=cv_cfg,
        inner_splits=3,
        process_noise=0.1,
        mode=mode,
    )
    return result.predictions


def test_every_row_is_scored():
    X, y = _toy()
    preds = _run_nested(X, y)
    assert set(preds) == {"linreg"}
    assert not np.isnan(preds["linreg"]).any()
    assert len(preds["linreg"]) == len(y)


def test_nested_prediction_is_independent_of_own_target():
    """The core guarantee: y[i] must not influence the prediction for row i."""
    X, y = _toy()
    base = _run_nested(X, y)["linreg"]

    for row in (7, 53, 118):
        perturbed = y.copy()
        perturbed.iloc[row] += 500.0
        after = _run_nested(X, perturbed)["linreg"]
        assert np.isclose(base[row], after[row], atol=1e-9), (
            f"row {row}: own target leaked into its own nested prediction"
        )


def test_nested_is_stricter_than_shared_fold_stacking():
    """The shared-fold path reacts to a held-out target; the nested one does not."""
    X, y = _toy()
    cv_cfg = CVConfig(scheme="shuffled", n_splits=4, random_state=42)
    folds = make_folds(len(X), cv_cfg)

    def shared_fold_prediction(target: pd.Series) -> np.ndarray:
        meta = build_meta_features(X, target, {"linreg": {}}, folds)
        augmented = pd.concat([X, meta.add_prefix("rop_kf_")], axis=1)
        out = np.full(len(target), np.nan)
        for train_idx, val_idx in folds:
            model = create_model("linreg", {})
            model.fit(augmented.iloc[train_idx], target.iloc[train_idx])
            out[val_idx] = model.predict(augmented.iloc[val_idx])
        return out

    row = 53
    perturbed = y.copy()
    perturbed.iloc[row] += 500.0

    shared_before = shared_fold_prediction(y)[row]
    shared_after = shared_fold_prediction(perturbed)[row]
    assert not np.isclose(shared_before, shared_after, atol=1e-6), (
        "expected the shared-fold path to be sensitive to the held-out target"
    )

    nested_before = _run_nested(X, y)["linreg"][row]
    nested_after = _run_nested(X, perturbed)["linreg"][row]
    assert np.isclose(nested_before, nested_after, atol=1e-9)


def test_observation_noise_logged_per_outer_fold():
    X, y = _toy()
    cv_cfg = CVConfig(scheme="shuffled", n_splits=4, random_state=42)
    folds = make_folds(len(X), cv_cfg)
    result = nested_stack_evaluation(
        X, y,
        base_models={"linreg": {}},
        super_models={"linreg": {}},
        outer_folds=folds,
        cv_cfg=cv_cfg,
        inner_splits=3,
        process_noise=0.1,
        mode="causal",
    )
    assert len(result.observation_noise) == len(folds)
    assert all(entry["linreg"] > 0 for entry in result.observation_noise)


def test_blocked_scheme_runs_and_scores_every_row():
    X, y = _toy()
    cv_cfg = CVConfig(scheme="blocked", n_splits=4, gap=2)
    folds = make_folds(len(X), cv_cfg)
    result = nested_stack_evaluation(
        X, y,
        base_models={"linreg": {}},
        super_models={"linreg": {}},
        outer_folds=folds,
        cv_cfg=cv_cfg,
        inner_splits=3,
        process_noise=0.1,
        mode="causal",
    )
    assert not np.isnan(result.predictions["linreg"]).any()
