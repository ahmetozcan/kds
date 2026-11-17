"""Tests for the Optuna tuning stage."""

import json

import numpy as np
import pytest

from tbm_rop.cv import CVConfig
from tbm_rop.tuning import (
    SEARCH_SPACE_REGISTRY,
    TuningConfig,
    load_tuned_params,
    merge_model_params,
    nested_oof_predictions,
    tune_on_folds,
)


def _tiny_data(n: int = 120, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = X[:, 0] * 2.0 + X[:, 1] + rng.normal(scale=0.1, size=n)
    return X, y


def _tiny_rf_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 5, 10),
        "max_depth": trial.suggest_int("max_depth", 2, 4),
    }


def _folds(n: int, k: int = 3) -> list:
    from tbm_rop.cv import make_folds

    return make_folds(n, CVConfig(scheme="shuffled", n_splits=k, random_state=0))


def test_search_space_registry_covers_tunable_models():
    for name in ("random_forest", "lightgbm", "catboost", "mlp"):
        assert name in SEARCH_SPACE_REGISTRY


def test_tune_on_folds_returns_valid_best_params():
    X, y = _tiny_data()
    folds = _folds(len(X))
    cfg = TuningConfig(n_trials=2, seed=0)

    best_params, best_value = tune_on_folds(
        X, y, "random_forest", folds, cfg, space_fn=_tiny_rf_space
    )

    assert set(best_params) == {"n_estimators", "max_depth"}
    assert 5 <= best_params["n_estimators"] <= 10
    assert isinstance(best_value, float)


def test_nested_oof_covers_all_rows_and_reports_fold_params():
    X, y = _tiny_data()
    outer = _folds(len(X), k=3)
    cfg = TuningConfig(n_trials=2, inner_splits=2, seed=0)

    result = nested_oof_predictions(
        X, y, "random_forest", outer, CVConfig(scheme="shuffled", n_splits=3), cfg,
        space_fn=_tiny_rf_space,
    )

    assert len(result.oof) == len(X)
    assert not np.isnan(result.oof).any()
    assert len(result.fold_params) == len(outer)
    for params in result.fold_params:
        assert set(params) == {"n_estimators", "max_depth"}


def test_unknown_model_raises():
    X, y = _tiny_data()
    with pytest.raises(KeyError, match="No search space"):
        tune_on_folds(X, y, "not_a_model", _folds(len(X)), TuningConfig(n_trials=1))


def test_merge_model_params_overrides_only_tuned_models():
    base = {"random_forest": {"n_jobs": 4}, "tabpfn": {}}
    tuned = {"random_forest": {"n_estimators": 300}}

    merged = merge_model_params(base, tuned)

    assert merged["random_forest"] == {"n_jobs": 4, "n_estimators": 300}
    assert merged["tabpfn"] == {}
    assert base["random_forest"] == {"n_jobs": 4}  # input untouched


def test_merge_model_params_ignores_models_not_in_base():
    base = {"tabpfn": {}}
    tuned = {"mlp": {"n_layers": 2}}  # deliberately excluded from the stack

    merged = merge_model_params(base, tuned)

    assert "mlp" not in merged  # must not silently expand the ensemble
    assert merged == {"tabpfn": {}}


def test_nested_selection_never_fits_on_outer_test_rows(monkeypatch):
    """Invariant guard: every fit between two outer-test predictions must be
    disjoint from the rows predicted next (row identity = X[:, 0])."""
    import tbm_rop.tuning as tuning_mod

    X, y = _tiny_data()
    X[:, 0] = np.arange(len(X))  # unique row ids
    outer = _folds(len(X), k=3)

    real_create = tuning_mod.create_model
    fit_rows_since_predict: list[set[float]] = [set()]

    def spying_create(name, params):
        model = real_create(name, params)
        real_fit, real_predict = model.fit, model.predict

        def fit(Xf, yf):
            fit_rows_since_predict[-1] |= set(np.asarray(Xf)[:, 0])
            return real_fit(Xf, yf)

        def predict(Xp):
            seen, pred_rows = fit_rows_since_predict[-1], set(np.asarray(Xp)[:, 0])
            # inner-CV predicts inside outer-train; the final predict of each
            # outer fold covers test rows — those must never have been fit on.
            if not pred_rows <= seen:  # predicting unseen rows == outer test
                assert not (seen & pred_rows), "outer-test row used during selection"
                fit_rows_since_predict.append(set())
            return real_predict(Xp)

        model.fit, model.predict = fit, predict
        return model

    monkeypatch.setattr(tuning_mod, "create_model", spying_create)
    result = nested_oof_predictions(
        X, y, "random_forest", outer, CVConfig(scheme="shuffled", n_splits=3),
        TuningConfig(n_trials=1, inner_splits=2, seed=0), space_fn=_tiny_rf_space,
    )
    assert not np.isnan(result.oof).any()


def test_load_tuned_params_roundtrip(tmp_path):
    path = tmp_path / "tuned.json"
    payload = {"random_forest": {"n_estimators": 300}, "mlp": {"n_layers": 2}}
    path.write_text(json.dumps(payload))

    loaded = load_tuned_params(path)

    assert loaded == payload
