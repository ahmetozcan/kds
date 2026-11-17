"""Tests for the scalar Kalman filter/smoother."""

import numpy as np
import pytest

from tbm_rop.kalman import KalmanConfig, denoise_oof_foldwise, denoise_series


def _cfg(mode: str) -> KalmanConfig:
    return KalmanConfig(process_noise=0.1, observation_noise=1.0, mode=mode)


def test_constant_series_is_preserved():
    obs = np.full(50, 3.7)
    for mode in ("causal", "smoother"):
        out = denoise_series(obs, _cfg(mode))
        np.testing.assert_allclose(out, obs, atol=1e-9)


def test_causal_filter_ignores_future():
    rng = np.random.default_rng(0)
    obs = rng.normal(size=100)
    modified = obs.copy()
    modified[60:] += 100.0  # perturb only the future

    base = denoise_series(obs, _cfg("causal"))
    perturbed = denoise_series(modified, _cfg("causal"))
    np.testing.assert_allclose(base[:60], perturbed[:60], atol=1e-12)


def test_smoother_uses_future():
    rng = np.random.default_rng(0)
    obs = rng.normal(size=100)
    modified = obs.copy()
    modified[60:] += 100.0

    base = denoise_series(obs, _cfg("smoother"))
    perturbed = denoise_series(modified, _cfg("smoother"))
    assert not np.allclose(base[:60], perturbed[:60])


def test_smoother_reduces_noise_variance():
    rng = np.random.default_rng(1)
    truth = np.sin(np.linspace(0, 8 * np.pi, 500))
    obs = truth + rng.normal(scale=0.5, size=500)

    smoothed = denoise_series(obs, _cfg("smoother"))
    assert np.mean((smoothed - truth) ** 2) < np.mean((obs - truth) ** 2)


def test_none_mode_is_identity():
    rng = np.random.default_rng(2)
    obs = rng.normal(size=100)
    out = denoise_series(obs, _cfg("none"))
    np.testing.assert_array_equal(out, obs)
    assert out is not obs  # must be a copy, not a view on the input


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="Unknown Kalman mode"):
        denoise_series(np.ones(5), KalmanConfig(0.1, 1.0, mode="banana"))


def test_empty_series_raises():
    with pytest.raises(ValueError, match="empty"):
        denoise_series(np.array([]), _cfg("causal"))


def _simple_folds(n: int, k: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    blocks = np.array_split(np.arange(n), k)
    return [(np.setdiff1d(np.arange(n), b), b) for b in blocks]


def test_foldwise_r_excludes_own_target():
    """A row's own target must not reach the filter that produces its value."""
    rng = np.random.default_rng(3)
    n = 80
    y = rng.normal(size=n)
    oof = y + rng.normal(scale=0.4, size=n)
    folds = _simple_folds(n)

    base, _ = denoise_oof_foldwise(oof, y, folds, process_noise=0.1)
    for row in (5, 33, 70):
        perturbed = y.copy()
        perturbed[row] += 1000.0
        after, _ = denoise_oof_foldwise(oof, perturbed, folds, process_noise=0.1)
        assert np.isclose(base[row], after[row], atol=1e-12)


def test_foldwise_reports_one_r_per_fold():
    rng = np.random.default_rng(4)
    n = 60
    y = rng.normal(size=n)
    oof = y + rng.normal(scale=0.5, size=n)
    folds = _simple_folds(n, k=5)

    _, r_per_fold = denoise_oof_foldwise(oof, y, folds, process_noise=0.1)
    assert len(r_per_fold) == 5
    assert all(r > 0 for r in r_per_fold)


def test_foldwise_none_mode_is_identity():
    rng = np.random.default_rng(5)
    n = 40
    y = rng.normal(size=n)
    oof = y + rng.normal(scale=0.3, size=n)
    out, _ = denoise_oof_foldwise(oof, y, _simple_folds(n), process_noise=0.1, mode="none")
    np.testing.assert_allclose(out, oof, atol=1e-12)


def test_foldwise_rejects_incomplete_coverage():
    rng = np.random.default_rng(6)
    n = 40
    y = rng.normal(size=n)
    oof = y + rng.normal(scale=0.3, size=n)
    partial = [(np.arange(20, n), np.arange(0, 20))]  # rows 20.. never validated
    with pytest.raises(ValueError, match="unassigned"):
        denoise_oof_foldwise(oof, y, partial, process_noise=0.1)
