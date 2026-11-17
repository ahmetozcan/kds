"""Tests for cross-validation fold construction."""

import numpy as np
import pytest

from tbm_rop.cv import CVConfig, make_folds


def test_shuffled_folds_cover_all_rows():
    folds = make_folds(100, CVConfig(scheme="shuffled", n_splits=5))
    all_val = np.concatenate([val for _, val in folds])
    assert sorted(all_val.tolist()) == list(range(100))


def test_shuffled_folds_are_reproducible():
    a = make_folds(100, CVConfig(scheme="shuffled", random_state=42))
    b = make_folds(100, CVConfig(scheme="shuffled", random_state=42))
    for (_, va), (_, vb) in zip(a, b):
        np.testing.assert_array_equal(va, vb)


def test_blocked_folds_are_contiguous():
    folds = make_folds(100, CVConfig(scheme="blocked", n_splits=5))
    for _, val in folds:
        assert np.all(np.diff(val) == 1)


def test_blocked_gap_purges_neighbours():
    folds = make_folds(100, CVConfig(scheme="blocked", n_splits=5, gap=3))
    for train, val in folds:
        forbidden = set(range(val[0] - 3, val[-1] + 4))
        assert forbidden.isdisjoint(train.tolist())


def test_blocked_no_overlap_between_train_and_val():
    folds = make_folds(101, CVConfig(scheme="blocked", n_splits=5, gap=0))
    for train, val in folds:
        assert set(train.tolist()).isdisjoint(val.tolist())


def test_unknown_scheme_raises():
    with pytest.raises(ValueError, match="Unknown CV scheme"):
        make_folds(10, CVConfig(scheme="spiral"))
