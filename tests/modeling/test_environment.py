"""
Tests for the trading environment.

The reward is the one thing here that cannot be wrong: every PnL figure, every
baseline comparison, and the agent's whole learned policy inherit it. These
tests pin it with named weekdays and hand-computed numbers, so a timing shift
fails loudly instead of quietly changing the results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nvda_rl.modeling.environment import TradingEnv

COST = 0.001


def calendar(returns_by_day: list[float]) -> pd.DataFrame:
    """
    A frame with real weekday dates where `returns_by_day[i]` is the return
    realized ON day i (close i-1 to close i). `next_return` therefore carries,
    on row i, the return that a position entered at close i will earn.
    """
    n = len(returns_by_day)
    close = [100.0]
    for r in returns_by_day[1:]:
        close.append(close[-1] * (1 + r))
    df = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=n),   # Mon, Tue, Wed, ...
        "close": close,
    })
    df["return"] = df["close"].pct_change()
    df["next_return"] = df["return"].shift(-1)
    df["state_bin"] = 0
    return df


def test_monday_action_earns_tuesday_return_exactly():
    """
    The reviewer's test, written out in full.

    Monday close: the agent sees Monday's state and goes long.
    Tuesday: the market rises 10%.
    That 10% belongs to the Monday decision, and the entry cost is charged on
    Monday, so the Monday step must pay exactly 1 * 0.10 - 0.001 = 0.099.

    An earlier version multiplied Tuesday's return by the position held *before*
    Monday's action, which was flat, so the Monday long earned nothing and the
    payoff landed a day late.
    """
    df = calendar([0.0, 0.10, 0.0, 0.0])          # Tuesday is +10%
    assert df.loc[0, "date"].day_name() == "Monday"
    assert df.loc[1, "date"].day_name() == "Tuesday"
    assert df.loc[0, "next_return"] == pytest.approx(0.10)

    env = TradingEnv(df, ["state_bin"], cost=COST)
    env.reset()

    _, monday_reward, _, _, info = env.step(1)     # long, decided at Monday close
    assert info["gross_return"] == pytest.approx(0.10), "Monday's long must earn Tuesday's move"
    assert info["cost"] == pytest.approx(COST), "entry cost is charged on Monday"
    assert monday_reward == pytest.approx(0.10 - COST)

    _, tuesday_reward, _, _, info = env.step(1)    # hold: no move, no cost
    assert info["gross_return"] == pytest.approx(0.0)
    assert tuesday_reward == pytest.approx(0.0)


def test_short_earns_the_negative_of_the_next_day_move():
    df = calendar([0.0, -0.05, 0.0, 0.0])
    env = TradingEnv(df, ["state_bin"], cost=0.0)
    env.reset()
    _, reward, _, _, _ = env.step(-1)
    assert reward == pytest.approx(0.05)


def test_flat_position_earns_nothing_whatever_the_market_does():
    df = calendar([0.0, 0.25, -0.25, 0.0])
    env = TradingEnv(df, ["state_bin"], cost=0.0)
    env.reset()
    for _ in range(2):
        _, reward, _, _, _ = env.step(0)
        assert reward == pytest.approx(0.0)


def test_cost_scales_with_the_size_of_the_position_change():
    df = calendar([0.0] * 5)
    env = TradingEnv(df, ["state_bin"], cost=COST)
    env.reset()

    _, _, _, _, info = env.step(1)      # 0 -> +1
    assert info["turnover"] == 1
    assert info["cost"] == pytest.approx(COST)

    _, _, _, _, info = env.step(-1)     # +1 -> -1 crosses two units
    assert info["turnover"] == 2
    assert info["cost"] == pytest.approx(2 * COST)

    _, _, _, _, info = env.step(-1)     # hold
    assert info["turnover"] == 0
    assert info["cost"] == 0.0


def test_full_reward_formula_on_a_worked_sequence():
    """Long Monday, flip short Tuesday, with every term computed by hand."""
    df = calendar([0.0, 0.02, -0.03, 0.0])
    env = TradingEnv(df, ["state_bin"], cost=COST)
    env.reset()

    _, r_mon, _, _, _ = env.step(1)      # earns Tue +2%, pays 1 unit of cost
    assert r_mon == pytest.approx(1 * 0.02 - COST * 1)

    _, r_tue, _, _, _ = env.step(-1)     # earns Wed -3% while short, pays 2 units
    assert r_tue == pytest.approx(-1 * -0.03 - COST * 2)


def test_position_is_part_of_the_observation():
    df = calendar([0.0] * 4)
    env = TradingEnv(df, ["state_bin"], cost=COST)
    state, _ = env.reset()
    assert state[-1] == 0
    next_state, _, _, _, _ = env.step(1)
    assert next_state[-1] == 1


def test_episode_ends_when_no_next_return_remains():
    df = calendar([0.0] * 5)             # the final row has next_return = NaN
    env = TradingEnv(df, ["state_bin"], cost=0.0)
    env.reset()
    assert env.n_steps == 4
    steps, terminated = 0, False
    while not terminated:
        _, _, terminated, _, _ = env.step(0)
        steps += 1
    assert steps == 4


def test_ledger_reproduces_the_step_accounting():
    df = calendar([0.0, 0.03, -0.02, 0.01])
    ledger = TradingEnv(df, ["state_bin"], cost=COST).run_policy(lambda s: 1)
    assert ledger.loc[0, "net_return"] == pytest.approx(0.03 - COST)
    assert ledger.loc[1, "net_return"] == pytest.approx(-0.02)
    assert ledger["equity"].iloc[-1] == pytest.approx(
        np.prod(1 + ledger["net_return"].to_numpy())
    )


def test_gymnasium_contract():
    """reset returns (obs, info) and step returns the five-tuple."""
    df = calendar([0.0] * 4)
    env = TradingEnv(df, ["state_bin"], cost=0.0)
    out = env.reset(seed=0)
    assert isinstance(out, tuple) and len(out) == 2
    assert env.action_space.n == 3
    assert len(env.step(0)) == 5


def test_invalid_action_is_rejected():
    env = TradingEnv(calendar([0.0] * 4), ["state_bin"], cost=COST)
    env.reset()
    with pytest.raises(ValueError):
        env.step(2)


def test_missing_columns_are_rejected_early():
    df = calendar([0.0] * 4).drop(columns=["next_return"])
    with pytest.raises(KeyError):
        TradingEnv(df, ["state_bin"])
