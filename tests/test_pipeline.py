"""
End-to-end tests over the real cached data.

The unit tests check each piece in isolation; these check that the pieces still
line up once they are joined, which is where a timing or split mistake actually
shows itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nvda_rl.config import N_REGIMES, STATE_BINS, STATE_FEATURES
from nvda_rl.dataset import clean_prices, load_prices, split_three_way
from nvda_rl.features import add_features, apply_binner, drop_warmup, fit_binner, regime_matrix
from nvda_rl.modeling.agents import buy_and_hold_policy
from nvda_rl.modeling.environment import TradingEnv
from nvda_rl.modeling.evaluate import performance
from nvda_rl.modeling.regimes import assign_regimes, fit_kmeans_regimes

DATA = "data/raw"


@pytest.fixture(scope="module")
def featured() -> pd.DataFrame:
    return drop_warmup(add_features(clean_prices(load_prices(DATA))))


def test_the_pipeline_produces_a_usable_panel(featured):
    assert len(featured) > 3000
    assert featured[STATE_FEATURES].notna().all().all()
    assert featured["date"].is_monotonic_increasing
    assert not featured["date"].duplicated().any()


def test_the_three_windows_are_ordered_and_disjoint(featured):
    train, validation, test = split_three_way(featured)
    assert len(train) and len(validation) and len(test)
    assert train["date"].max() < validation["date"].min()
    assert validation["date"].max() < test["date"].min()
    assert len(train) + len(validation) + len(test) == len(featured)


def test_buy_and_hold_reproduces_the_underlying_move(featured):
    """
    The clearest end-to-end timing check available: a permanently long,
    cost-free position must earn exactly what the stock did over the same bars.
    If the reward were shifted by a day, these would not match.
    """
    _, _, test = split_three_way(featured)
    env = TradingEnv(test, ["dist_252d_high"], cost=0.0, return_column="next_return")
    ledger = env.run_policy(buy_and_hold_policy)

    traded = test.iloc[: len(ledger)]
    expected = float(np.prod(1 + traded["next_return"].to_numpy()))
    assert ledger["equity"].iloc[-1] == pytest.approx(expected, rel=1e-9)


def test_costs_can_only_reduce_the_result(featured):
    _, _, test = split_three_way(featured)
    free = performance(TradingEnv(test, ["dist_252d_high"], cost=0.0).run_policy(buy_and_hold_policy))
    dear = performance(TradingEnv(test, ["dist_252d_high"], cost=0.01).run_policy(buy_and_hold_policy))
    assert dear["cumulative_pnl_%"] <= free["cumulative_pnl_%"]


def test_regimes_fitted_on_train_apply_cleanly_to_later_windows(featured):
    train, validation, test = split_three_way(featured)
    model = fit_kmeans_regimes(regime_matrix(train), train["return"].to_numpy(), N_REGIMES)
    for window in (train, validation, test):
        labels = assign_regimes(model, regime_matrix(window))
        assert len(labels) == len(window)
        assert set(labels) <= set(range(N_REGIMES))


def test_state_columns_are_finite_and_bounded_after_binning(featured):
    train, _, test = split_three_way(featured)
    binner = fit_binner(train, STATE_FEATURES, n_bins=STATE_BINS)
    binned = apply_binner(test, binner)
    for col in STATE_FEATURES:
        values = binned[f"{col}_bin"]
        assert values.between(0, STATE_BINS - 1).all()


def test_an_agent_can_train_and_trade_the_real_tape(featured):
    """A short run, purely to prove the wiring holds outside the toy fixtures."""
    from nvda_rl.modeling.agents import QLearningAgent

    train, _, test = split_three_way(featured)
    binner = fit_binner(train, STATE_FEATURES, n_bins=STATE_BINS)
    train, test = apply_binner(train, binner), apply_binner(test, binner)
    cols = [f"{c}_bin" for c in STATE_FEATURES]

    agent = QLearningAgent()
    agent.train(TradingEnv(train, cols), n_episodes=3)
    ledger = TradingEnv(test, cols).run_policy(agent.greedy_policy())

    assert len(ledger) == TradingEnv(test, cols).n_steps
    assert ledger["net_return"].notna().all()
    assert np.isfinite(performance(ledger)["cumulative_pnl_%"])
