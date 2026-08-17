"""Tests for regime fitting, frozen prediction, label ordering, and stability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nvda_rl.config import REGIME_FEATURES
from nvda_rl.modeling.regimes import (
    assign_regimes,
    cluster_quality,
    fit_kmeans_regimes,
    regime_stability,
    select_n_regimes,
    transition_matrix,
)


def separable(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Three well-separated groups with clearly different mean returns."""
    rng = np.random.default_rng(seed)
    parts, rets = [], []
    for centre, ret in [(4.0, 0.03), (0.0, 0.0), (-4.0, -0.03)]:
        block = pd.DataFrame({c: rng.normal(centre, 0.3, n) for c in REGIME_FEATURES})
        block["return"] = rng.normal(ret, 0.001, n)
        parts.append(block)
        rets.append(block["return"].to_numpy())
    return pd.concat(parts, ignore_index=True), np.concatenate(rets)


def test_select_n_regimes_reports_every_candidate():
    x, _ = separable(80)
    out = select_n_regimes(x, k_range=range(2, 6))
    assert list(out.index) == [2, 3, 4, 5]
    assert {"inertia", "silhouette", "davies_bouldin"} <= set(out.columns)
    assert out["inertia"].is_monotonic_decreasing


def test_silhouette_finds_the_true_number_of_groups():
    x, _ = separable(150)
    out = select_n_regimes(x, k_range=range(2, 7))
    assert out["silhouette"].idxmax() == 3


def test_labels_are_ordered_by_mean_return():
    x, rets = separable()
    labels = assign_regimes(fit_kmeans_regimes(x, rets, 3), x)
    means = pd.Series(rets).groupby(labels).mean()
    assert means.loc[0] > means.loc[1] > means.loc[2]


def test_assignment_is_pure_prediction():
    x, rets = separable()
    model = fit_kmeans_regimes(x, rets, 3)
    before = model["pipeline"].named_steps["kmeans"].cluster_centers_.copy()
    assign_regimes(model, x.sample(frac=1.0, random_state=3))
    assert np.array_equal(before, model["pipeline"].named_steps["kmeans"].cluster_centers_)


def test_the_same_row_always_gets_the_same_label():
    x, rets = separable()
    model = fit_kmeans_regimes(x, rets, 3)
    assert np.array_equal(assign_regimes(model, x), assign_regimes(model, x))


def test_labels_cover_exactly_the_requested_regimes():
    x, rets = separable()
    labels = assign_regimes(fit_kmeans_regimes(x, rets, 3), x)
    assert set(labels) == {0, 1, 2}


def test_cluster_quality_ignores_dbscan_noise():
    x, rets = separable(60)
    labels = assign_regimes(fit_kmeans_regimes(x, rets, 3), x)
    noisy = labels.copy()
    noisy[:10] = -1
    assert cluster_quality(x, noisy)["n_clusters"] == 3


def test_cluster_quality_handles_a_degenerate_single_cluster():
    x, _ = separable(50)
    out = cluster_quality(x, np.zeros(len(x), dtype=int))
    assert out["n_clusters"] == 0
    assert np.isnan(out["silhouette"])


def test_transition_matrix_rows_are_probabilities():
    labels = np.array([0, 0, 1, 1, 1, 2, 2, 0])
    out = transition_matrix(labels)
    assert out.shape == (3, 3)
    assert np.allclose(out.sum(axis=1).dropna(), 1.0)


def test_transition_matrix_detects_perfect_persistence():
    labels = np.array([0] * 20 + [1] * 20)
    out = transition_matrix(labels)
    assert out.loc[0, 0] > 0.9
    assert out.loc[1, 1] == pytest.approx(1.0)


def test_well_separated_groups_are_stable_across_seeds():
    """If the groups are real, reseeding must not move them."""
    x, rets = separable(120)
    out = regime_stability(x, rets, n_regimes=3, n_seeds=5)
    assert out.loc[0, "mean_ARI"] > 0.95
    assert out.loc[0, "min_ARI"] > 0.90
