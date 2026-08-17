"""
Leakage guards.

The dangerous failure in this project is not a crash, it is a backtest that
looks excellent because the state encoded the future. These tests are written
to prove that specifically, which the earlier versions did not: fitting the
binner twice on the same frame only demonstrated that the code is
deterministic. Here the fitting workflow meets two wildly different test sets
and the fitted artefacts must come out byte-identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nvda_rl.config import REGIME_FEATURES
from nvda_rl.features import apply_binner, fit_binner
from nvda_rl.modeling.regimes import assign_regimes, fit_kmeans_regimes


def make_features(n: int, seed: int, scale: float = 1.0, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({col: rng.normal(shift, scale, n) for col in REGIME_FEATURES})


def fitted_artefacts(train: pd.DataFrame, unseen: pd.DataFrame) -> dict:
    """
    Run the whole fit-then-apply workflow and return everything that was
    learned. If any of it depends on `unseen`, the numbers below will differ
    between two calls with different unseen data.
    """
    binner = fit_binner(train, REGIME_FEATURES, n_bins=3)
    returns = train[REGIME_FEATURES[0]].to_numpy()
    model = fit_kmeans_regimes(train, returns, n_regimes=3)

    apply_binner(unseen, binner)          # the test window passes through here
    assign_regimes(model, unseen)         # and here

    scaler = model["pipeline"].named_steps["scale"]
    return {
        "bin_edges": [np.asarray(e) for e in binner["encoder"].bin_edges_],
        "scaler_mean": scaler.mean_.copy(),
        "scaler_scale": scaler.scale_.copy(),
        "centroids": model["pipeline"].named_steps["kmeans"].cluster_centers_.copy(),
        "mapping": dict(model["mapping"]),
    }


def test_fitted_artefacts_ignore_the_test_window_entirely():
    """
    Same training data, two test sets that could hardly be more different: one
    drawn from the training distribution, one shifted and fifty times wider.
    Every learned parameter must be identical.
    """
    train = make_features(400, seed=0)
    mild = make_features(300, seed=1)
    wild = make_features(300, seed=2, scale=50.0, shift=25.0)

    a = fitted_artefacts(train, mild)
    b = fitted_artefacts(train, wild)

    for left, right in zip(a["bin_edges"], b["bin_edges"], strict=True):
        assert np.array_equal(left, right), "bin edges moved with the test data"
    assert np.array_equal(a["scaler_mean"], b["scaler_mean"]), "scaler mean moved"
    assert np.array_equal(a["scaler_scale"], b["scaler_scale"]), "scaler scale moved"
    assert np.array_equal(a["centroids"], b["centroids"]), "centroids moved"
    assert a["mapping"] == b["mapping"], "label ordering moved"


def test_bin_edges_come_from_training_quantiles_only():
    """The edges must sit inside the training range, not the combined range."""
    train = make_features(500, seed=3)
    binner = fit_binner(train, REGIME_FEATURES, n_bins=3)
    for col, edges in zip(REGIME_FEATURES, binner["encoder"].bin_edges_, strict=True):
        interior = np.asarray(edges)[1:-1]
        assert interior.min() > train[col].min()
        assert interior.max() < train[col].max()


def test_extreme_unseen_values_clip_into_the_outer_bins():
    train = make_features(300, seed=4)
    binner = fit_binner(train, REGIME_FEATURES, n_bins=3)
    extreme = pd.DataFrame({col: [-1e6, 1e6] for col in REGIME_FEATURES})
    binned = apply_binner(extreme, binner)
    for col in REGIME_FEATURES:
        assert binned[f"{col}_bin"].tolist() == [0, 2]


def test_predicting_on_new_data_does_not_move_the_centroids():
    train = make_features(400, seed=5)
    model = fit_kmeans_regimes(train, train[REGIME_FEATURES[0]].to_numpy(), n_regimes=3)
    before = model["pipeline"].named_steps["kmeans"].cluster_centers_.copy()

    for seed, scale in [(11, 1.0), (12, 30.0), (13, 0.01)]:
        assign_regimes(model, make_features(200, seed=seed, scale=scale))

    after = model["pipeline"].named_steps["kmeans"].cluster_centers_
    assert np.array_equal(before, after)


def test_regime_labels_are_ordered_by_mean_return():
    """
    Regime 0 must always be the highest-mean-return state. K-means numbers its
    clusters arbitrarily, so without the reordering every written claim about
    "regime 0" would silently rot between runs.
    """
    rng = np.random.default_rng(6)
    n = 300
    frame = pd.DataFrame({col: rng.normal(0, 0.1, 3 * n) for col in REGIME_FEATURES})
    frame.loc[: n - 1, "return"] = rng.normal(0.03, 0.001, n)
    frame.loc[n: 2 * n - 1, "return"] = rng.normal(0.00, 0.001, n)
    frame.loc[2 * n:, "return"] = rng.normal(-0.03, 0.001, n)

    model = fit_kmeans_regimes(frame, frame["return"].to_numpy(), n_regimes=3)
    labels = assign_regimes(model, frame)
    means = pd.Series(frame["return"].to_numpy()).groupby(labels).mean()
    assert means.loc[0] > means.loc[1] > means.loc[2]


def test_feature_timing_never_uses_the_future():
    """
    Rolling features must depend on the past only. Changing a future row must
    leave every earlier feature value untouched.
    """
    from nvda_rl.features import add_features

    rng = np.random.default_rng(7)
    n = 400
    base = pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=n),
        "close": 100 * np.cumprod(1 + rng.normal(0.001, 0.02, n)),
        "volume": rng.integers(1e6, 5e6, n),
    })
    base["return"] = base["close"].pct_change()
    base["log_return"] = np.log(base["close"]).diff()

    tampered = base.copy()
    tampered.loc[n - 1, "close"] *= 5          # a shock on the very last day

    a = add_features(base)
    b = add_features(tampered)
    cols = ["volatility_21", "momentum_21", "dist_252d_high", "volume_zscore"]
    pd.testing.assert_frame_equal(a.loc[: n - 30, cols], b.loc[: n - 30, cols])


def test_next_return_is_the_following_days_return():
    """The environment pays next_return, so its alignment is a leakage matter."""
    from nvda_rl.features import add_features

    df = pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=5),
        "close": [100.0, 110.0, 121.0, 121.0, 121.0],
        "volume": [1e6] * 5,
    })
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"]).diff()
    out = add_features(df)
    assert out.loc[0, "next_return"] == pytest.approx(out.loc[1, "return"])
    assert out.loc[1, "next_return"] == pytest.approx(0.10)
    assert pd.isna(out["next_return"].iloc[-1])
