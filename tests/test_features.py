"""Tests for feature construction, warm-up handling, and train-only binning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nvda_rl.config import REGIME_FEATURES, STATE_FEATURES
from nvda_rl.features import (
    add_features,
    apply_binner,
    drop_warmup,
    fit_binner,
    regime_matrix,
)


def prices(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "date": pd.bdate_range("2019-01-01", periods=n),
        "close": 100 * np.cumprod(1 + rng.normal(0.001, 0.02, n)),
        "volume": rng.integers(1e6, 5e6, n),
    })
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"]).diff()
    return df


def test_every_regime_feature_is_produced():
    out = add_features(prices())
    for col in REGIME_FEATURES:
        assert col in out.columns


def test_momentum_matches_a_hand_computed_window():
    df = prices(300)
    out = add_features(df)
    i = 250
    expected = df.loc[i, "close"] / df.loc[i - 21, "close"] - 1
    assert out.loc[i, "momentum_21"] == pytest.approx(expected)


def test_distance_from_high_is_never_positive():
    """The current close cannot exceed the max of a window that contains it."""
    out = add_features(prices(400))
    tail = out.dropna(subset=["dist_252d_high"])
    assert not tail.empty
    assert (tail["dist_252d_high"] <= 1e-12).all()


def test_distance_from_high_is_zero_on_a_series_that_only_rises():
    """A monotonically rising close is always at its own trailing high."""
    n = 300
    df = pd.DataFrame({
        "date": pd.bdate_range("2019-01-01", periods=n),
        "close": np.linspace(100, 400, n),
        "volume": [1e6] * n,
    })
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"]).diff()
    tail = add_features(df).dropna(subset=["dist_252d_high"])
    assert tail["dist_252d_high"].abs().max() == pytest.approx(0.0, abs=1e-12)


def test_drop_warmup_removes_only_the_incomplete_leading_rows():
    out = add_features(prices(400))
    trimmed = drop_warmup(out)
    assert trimmed[REGIME_FEATURES].notna().all().all()
    assert len(trimmed) < len(out)
    assert trimmed["date"].min() > out["date"].min()


def test_regime_matrix_keeps_the_column_order_fixed():
    out = drop_warmup(add_features(prices()))
    assert list(regime_matrix(out).columns) == REGIME_FEATURES


def test_binner_produces_balanced_terciles_on_the_training_data():
    train = drop_warmup(add_features(prices(600)))
    binner = fit_binner(train, STATE_FEATURES, n_bins=3)
    binned = apply_binner(train, binner)
    counts = binned[f"{STATE_FEATURES[0]}_bin"].value_counts()
    assert set(counts.index) == {0, 1, 2}
    assert counts.max() - counts.min() <= 2


def test_binner_is_reusable_and_returns_integer_bins():
    train = drop_warmup(add_features(prices(500, seed=1)))
    test = drop_warmup(add_features(prices(300, seed=2)))
    binner = fit_binner(train, STATE_FEATURES, n_bins=2)
    out = apply_binner(test, binner)
    for col in STATE_FEATURES:
        values = out[f"{col}_bin"]
        assert values.dtype.kind == "i"
        assert values.between(0, 1).all()


def test_binning_the_same_frame_twice_is_stable():
    train = drop_warmup(add_features(prices(400)))
    binner = fit_binner(train, STATE_FEATURES, n_bins=3)
    first = apply_binner(train, binner)
    second = apply_binner(train, binner)
    pd.testing.assert_frame_equal(first, second)
