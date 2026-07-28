"""
Tests for the trading environment.

The reward formula is the one piece of this project that cannot be wrong: every
PnL figure, every comparison against buy-and-hold, and the agent's entire
learned policy inherit whatever it does. These tests pin it to the
specification with numbers worked out by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nvda_rl.modeling.environment import TradingEnv

COST = 0.001


def make_frame(returns: list[float]) -> pd.DataFrame:
    """A minimal frame with one discrete state column and known next returns."""
    n = len(returns)
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="B"),
        "state_bin": [0] * n,
        "next_return": returns,
    })


def test_reward_pays_previous_position_not_current():
    """
    r_t = a_{t-1} * return_t, so the action chosen now cannot collect today's
    return. Going long on the first step must earn nothing, because the agent
    entered the episode flat.
    """
    env = TradingEnv(make_frame([0.10, 0.10, 0.10]), ["state_bin"], cost=0.0)
    _, reward, _, info = env.step(1)
    assert info["gross_return"] == 0.0
    assert reward == 0.0

    # Now the long position is on, so the next step collects the return.
    _, reward, _, info = env.step(1)
    assert info["gross_return"] == pytest.approx(0.10)
    assert reward == pytest.approx(0.10)


def test_short_position_earns_negative_of_return():
    env = TradingEnv(make_frame([-0.05, -0.05, -0.05]), ["state_bin"], cost=0.0)
    env.step(-1)                       # take the short, earns nothing yet
    _, reward, _, _ = env.step(-1)     # short through a -5% day
    assert reward == pytest.approx(0.05)


def test_cost_scales_with_size_of_position_change():
    """A short-to-long flip moves two units and must cost twice a flat-to-long."""
    env = TradingEnv(make_frame([0.0] * 4), ["state_bin"], cost=COST)

    _, _, _, info = env.step(1)        # 0 -> +1, one unit of turnover
    assert info["turnover"] == 1
    assert info["cost"] == pytest.approx(COST)

    _, _, _, info = env.step(-1)       # +1 -> -1, two units
    assert info["turnover"] == 2
    assert info["cost"] == pytest.approx(2 * COST)


def test_holding_a_position_incurs_no_cost():
    env = TradingEnv(make_frame([0.0] * 4), ["state_bin"], cost=COST)
    env.step(1)
    _, _, _, info = env.step(1)
    assert info["turnover"] == 0
    assert info["cost"] == 0.0


def test_reward_equals_formula_on_a_worked_example():
    """
    Hand-computed: entering long costs 0.001; the next day that long earns
    +2% and the flip to short costs 2 * 0.001.
    """
    env = TradingEnv(make_frame([0.02, 0.02, 0.02]), ["state_bin"], cost=COST)

    _, r0, _, _ = env.step(1)
    assert r0 == pytest.approx(0 * 0.02 - COST * 1)

    _, r1, _, _ = env.step(-1)
    assert r1 == pytest.approx(1 * 0.02 - COST * 2)


def test_position_is_part_of_the_state():
    """
    The position must appear in the observation, otherwise the agent cannot
    know what a trade would cost and the problem stops being an MDP.
    """
    env = TradingEnv(make_frame([0.0] * 3), ["state_bin"], cost=COST)
    state = env.reset()
    assert state[-1] == 0
    next_state, _, _, _ = env.step(1)
    assert next_state[-1] == 1


def test_episode_ends_before_the_last_row():
    """The final row has no next-day return to pay, so it is not tradeable."""
    env = TradingEnv(make_frame([0.01] * 5), ["state_bin"], cost=0.0)
    assert env.n_steps == 4
    steps = 0
    done = False
    while not done:
        _, _, done, _ = env.step(0)
        steps += 1
    assert steps == 4


def test_run_policy_ledger_matches_step_accounting():
    """The replayed ledger must reproduce the same net rewards as stepping."""
    frame = make_frame([0.03, -0.02, 0.01, 0.04])
    env = TradingEnv(frame, ["state_bin"], cost=COST)
    ledger = env.run_policy(lambda state: 1)

    expected_first = 0 * 0.03 - COST * 1        # enter long, earn nothing yet
    expected_second = 1 * (-0.02) - 0.0         # hold long through -2%, no cost
    assert ledger.loc[0, "net_return"] == pytest.approx(expected_first)
    assert ledger.loc[1, "net_return"] == pytest.approx(expected_second)
    assert ledger["equity"].iloc[-1] == pytest.approx(
        np.prod(1 + ledger["net_return"].to_numpy())
    )


def test_invalid_action_is_rejected():
    env = TradingEnv(make_frame([0.0] * 3), ["state_bin"], cost=COST)
    with pytest.raises(ValueError):
        env.step(2)


def test_missing_columns_are_rejected_early():
    frame = make_frame([0.0] * 3).drop(columns=["next_return"])
    with pytest.raises(KeyError):
        TradingEnv(frame, ["state_bin"])
