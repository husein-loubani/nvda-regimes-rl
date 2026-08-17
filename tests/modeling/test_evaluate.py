"""
Tests for the reporting layer.

Every headline number in the notebook comes out of these functions, so a
mistake here changes the conclusions without changing anything that looks like
a model. Each metric is pinned against a hand-computed example.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nvda_rl.config import TRADING_DAYS
from nvda_rl.modeling.evaluate import (
    action_distribution,
    compare_policies,
    performance,
    sharpe_ratio,
    walk_forward,
)


def ledger_from(net: list[float], actions: list[int] | None = None,
                turnover: list[float] | None = None,
                start: str = "2021-01-04") -> pd.DataFrame:
    n = len(net)
    actions = actions if actions is not None else [1] * n
    turnover = turnover if turnover is not None else [0.0] * n
    df = pd.DataFrame({
        "date": pd.bdate_range(start, periods=n),
        "market_return": net,
        "position_before": [0] + actions[:-1],
        "action": actions,
        "gross_return": net,
        "cost": [0.0] * n,
        "turnover": turnover,
        "net_return": net,
    })
    df["equity"] = (1 + df["net_return"]).cumprod()
    return df


def test_cumulative_pnl_compounds_rather_than_sums():
    """Two +10% days compound to +21%, not +20%."""
    stats = performance(ledger_from([0.10, 0.10]))
    assert stats["total_growth_x"] == pytest.approx(1.21)
    assert stats["cumulative_pnl_%"] == pytest.approx(21.0)


def test_sharpe_matches_the_hand_calculation():
    net = np.array([0.01, -0.01, 0.02, 0.00, 0.01])
    expected = net.mean() / net.std(ddof=1) * np.sqrt(TRADING_DAYS)
    assert sharpe_ratio(net) == pytest.approx(expected)
    assert performance(ledger_from(list(net)))["sharpe"] == pytest.approx(round(expected, 2))


def test_sharpe_is_nan_when_returns_never_move():
    assert np.isnan(sharpe_ratio(np.zeros(10)))


def test_drawdown_is_measured_from_the_initial_capital():
    """
    A policy that loses on day one is already in drawdown. Anchoring the peak
    at the first bar instead of at 1.0 would report 0% for that first loss.
    """
    stats = performance(ledger_from([-0.10, 0.0]))
    assert stats["max_drawdown_%"] == pytest.approx(-10.0)


def test_drawdown_captures_the_worst_peak_to_trough_fall():
    # 1.0 -> 1.5 -> 0.75: the fall from the 1.5 peak is 50%.
    stats = performance(ledger_from([0.5, -0.5]))
    assert stats["max_drawdown_%"] == pytest.approx(-50.0)


def test_turnover_is_annualized_from_the_units_traded():
    n = 10
    stats = performance(ledger_from([0.0] * n, turnover=[1.0] * n))
    assert stats["turnover_per_year"] == pytest.approx(TRADING_DAYS)


def test_hit_ratio_counts_only_days_with_a_position():
    # Three days: win, loss, flat. The flat day must not count either way.
    stats = performance(ledger_from([0.01, -0.01, 0.0], actions=[1, 1, 0]))
    assert stats["hit_ratio"] == pytest.approx(0.5)
    assert stats["days_in_market_%"] == pytest.approx(66.7, abs=0.1)


def test_cost_drag_is_the_gap_between_gross_and_net():
    led = ledger_from([0.01, 0.01])
    led["gross_return"] = [0.02, 0.02]
    stats = performance(led)
    assert stats["cost_drag_%"] == pytest.approx(2.0)


def test_compare_policies_sorts_by_cumulative_pnl():
    out = compare_policies({
        "weak": ledger_from([0.01, 0.01]),
        "strong": ledger_from([0.05, 0.05]),
    })
    assert list(out.index) == ["strong", "weak"]


def test_action_distribution_sums_to_one_hundred():
    out = action_distribution({"p": ledger_from([0.0] * 4, actions=[1, -1, 0, 1])})
    assert out.loc["p"].sum() == pytest.approx(100.0)
    assert out.loc["p", "action_1_%"] == pytest.approx(50.0)


def test_walk_forward_splits_by_calendar_year():
    # 300 business days from early 2021 crosses into 2022.
    led = ledger_from([0.001] * 300, start="2021-01-04")
    out = walk_forward(led)
    assert set(out.index) == {2021, 2022}
    assert out["days"].sum() == 300
