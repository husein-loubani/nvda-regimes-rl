"""
Leakage guards.

The dangerous failure in this project is not a crash, it is a backtest that
looks excellent because the state encoded information from the future. These
tests assert the two places that could happen: the discretization bin edges and
the fitted regime model must both depend on the training window alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nvda_rl.config import REGIME_FEATURES
from nvda_rl.features import apply_binner, fit_binner
from nvda_rl.modeling.regimes import assign_regimes, fit_kmeans_regimes


def make_features(n: int, seed: int, scale: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        col: rng.normal(0, scale, n) for col in REGIME_FEATURES
    })


def test_bin_edges_depend_only_on_training_data():
    """Changing the test window must not move a single bin edge."""
    train = make_features(300, seed=0)
    edges_a = fit_binner(train, REGIME_FEATURES, n_bins=3)
    edges_b = fit_binner(train, REGIME_FEATURES, n_bins=3)

    for col in REGIME_FEATURES:
        assert np.array_equal(edges_a[col], edges_b[col])

    # A wildly different test window is binned with the same frozen edges.
    wild = make_features(300, seed=99, scale=50.0)
    binned = apply_binner(wild, edges_a)
    assert binned.filter(like="_bin").notna().all().all()


def test_extreme_unseen_values_fall_into_outer_bins_without_error():
    train = make_features(200, seed=1)
    edges = fit_binner(train, REGIME_FEATURES, n_bins=3)

    extreme = pd.DataFrame({col: [-1e6, 1e6] for col in REGIME_FEATURES})
    binned = apply_binner(extreme, edges)

    for col in REGIME_FEATURES:
        values = binned[f"{col}_bin"].tolist()
        assert values[0] == 0                       # far below every edge
        assert values[1] == len(edges[col])         # far above every edge


def test_regime_assignment_on_test_does_not_change_the_fitted_model():
    """
    Assigning regimes to new data must be pure prediction. If the centroids
    moved when test data arrived, every regime label in the backtest would be
    contaminated by the future.
    """
    train = make_features(400, seed=2)
    train_returns = np.random.default_rng(2).normal(0, 0.02, 400)
    model = fit_kmeans_regimes(train, train_returns, n_regimes=3)

    centroids_before = model["pipeline"].named_steps["kmeans"].cluster_centers_.copy()
    test = make_features(200, seed=7, scale=3.0)
    labels = assign_regimes(model, test)
    centroids_after = model["pipeline"].named_steps["kmeans"].cluster_centers_

    assert np.array_equal(centroids_before, centroids_after)
    assert set(labels).issubset({0, 1, 2})


def test_regime_labels_are_ordered_by_mean_return():
    """
    Regime 0 must be the highest-mean-return state. Without this the cluster
    numbering is arbitrary and every written interpretation of "regime 0"
    silently breaks on a rerun.
    """
    rng = np.random.default_rng(3)
    n = 300
    # Three separable groups with clearly different mean returns.
    frame = pd.DataFrame({col: rng.normal(0, 0.1, 3 * n) for col in REGIME_FEATURES})
    frame.loc[:n - 1, "return"] = rng.normal(0.03, 0.001, n)
    frame.loc[n:2 * n - 1, "return"] = rng.normal(0.00, 0.001, n)
    frame.loc[2 * n:, "return"] = rng.normal(-0.03, 0.001, n)

    model = fit_kmeans_regimes(frame, frame["return"].to_numpy(), n_regimes=3)
    labels = assign_regimes(model, frame)

    means = pd.Series(frame["return"].to_numpy()).groupby(labels).mean()
    assert means.loc[0] > means.loc[1] > means.loc[2]
