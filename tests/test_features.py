"""Tests for feature builders and data loading helpers."""

import numpy as np
import pandas as pd
import pytest

from tbm_rop.data import clean_column_names
from tbm_rop.features import FEATURE_BUILDER_REGISTRY, build_features


@pytest.fixture()
def raw_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 60
    return pd.DataFrame(
        {
            "rop_mm_m": rng.uniform(10, 100, n),
            "rpm_rotate_per_minute": rng.uniform(1, 7, n),
            "chd_knm": rng.uniform(500, 3000, n),
            "thrust_kn": rng.uniform(5000, 30000, n),
            "alfa_degree": rng.uniform(30, 70, n),
            "watinflow_l_m": rng.uniform(0, 50, n),
            "weathering": rng.integers(1, 6, n).astype(float),
            "cai": rng.uniform(0.5, 4, n),
            "ucs_mpa": rng.uniform(10, 190, n),
            "ucs_weat": rng.uniform(2, 190, n),
        }
    )


def test_clean_column_names():
    df = pd.DataFrame(columns=["ROP (mm/m)", "CHD (kNm)", "UCS_Weat"])
    cleaned = clean_column_names(df)
    assert cleaned.columns.tolist() == ["rop_mm_m", "chd_knm", "ucs_weat"]


def test_all_builders_registered():
    assert {"physics", "temporal", "clustering", "polynomial"} <= set(FEATURE_BUILDER_REGISTRY)


def test_full_feature_pipeline_shapes(raw_frame):
    out = build_features(raw_frame)
    assert len(out) == len(raw_frame)
    assert "power_index_chd_rpm" in out.columns
    assert "rpm_rotate_per_minute_lag1" in out.columns
    assert "cluster_id" in out.columns
    assert "log_thrust^2" in out.columns
    assert not out.isna().any().any()


def test_temporal_features_are_causal(raw_frame):
    out = build_features(raw_frame, steps=["temporal"])
    modified = raw_frame.copy()
    modified.loc[len(modified) - 1, "thrust_kn"] += 1e6  # perturb only the last row
    out_mod = build_features(modified, steps=["temporal"])
    # All rows before the perturbed one must be unchanged.
    pd.testing.assert_frame_equal(out.iloc[:-5], out_mod.iloc[:-5])


def test_target_never_used_in_features(raw_frame):
    shifted = raw_frame.copy()
    shifted["rop_mm_m"] = 0.0  # zero out the target
    out_a = build_features(raw_frame)
    out_b = build_features(shifted)
    feature_cols = [c for c in out_a.columns if c != "rop_mm_m"]
    pd.testing.assert_frame_equal(out_a[feature_cols], out_b[feature_cols])


def test_unknown_step_raises(raw_frame):
    with pytest.raises(KeyError, match="Unknown feature builder"):
        build_features(raw_frame, steps=["alchemy"])
