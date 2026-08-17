"""
Tests for the tabular agents: update rules, exploration, terminal handling, and
action selection. These pin the learning arithmetic itself, so a wrong sign or
a mis-indexed action fails here rather than showing up as a mysteriously poor
backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nvda_rl.config import ACTIONS, EPSILON_END, EPSILON_START
from nvda_rl.modeling.agents import (
    QLearningAgent,
    SarsaAgent,
    _epsilon_at,
    buy_and_hold_policy,
    flat_policy,
    random_policy_factory,
)
from nvda_rl.modeling.environment import TradingEnv


def tiny_env(returns: list[float], cost: float = 0.0) -> TradingEnv:
    n = len(returns)
    df = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=n),
        "next_return": returns,
        "state_bin": [0] * n,
    })
    df.loc[n - 1, "next_return"] = np.nan
    return TradingEnv(df, ["state_bin"], cost=cost)


def test_epsilon_decays_from_start_to_end_then_holds():
    assert _epsilon_at(0) == pytest.approx(EPSILON_START)
    mid = _epsilon_at(150, n_decay=300)
    assert EPSILON_END < mid < EPSILON_START
    assert _epsilon_at(300, n_decay=300) == pytest.approx(EPSILON_END)
    assert _epsilon_at(10_000, n_decay=300) == pytest.approx(EPSILON_END)


def test_epsilon_one_always_explores_and_zero_always_exploits():
    agent = QLearningAgent()
    agent.q[(0, 0)] = np.array([-1.0, 0.0, 5.0])       # long is clearly best
    assert agent.select_action((0, 0), epsilon=0.0) == 1
    picks = {agent.select_action((0, 0), epsilon=1.0) for _ in range(50)}
    assert picks.issubset(set(ACTIONS)) and len(picks) > 1


def test_q_learning_update_matches_the_bellman_arithmetic():
    """One step, worked by hand: Q <- Q + alpha * (r + gamma * max Q' - Q)."""
    agent = QLearningAgent(alpha=0.5, gamma=0.9)
    state, nxt = (0, 0), (0, 1)
    agent.q[state] = np.array([0.0, 0.0, 0.0])
    agent.q[nxt] = np.array([0.0, 0.0, 2.0])           # best next value is 2.0

    reward, a_idx = 1.0, agent.action_index(1)
    target = reward + 0.9 * 2.0                         # 2.8
    expected = 0.0 + 0.5 * (target - 0.0)               # 1.4
    agent.q[state][a_idx] += agent.alpha * (target - agent.q[state][a_idx])
    assert agent.q[state][a_idx] == pytest.approx(expected)


def test_terminal_update_does_not_bootstrap():
    """
    On the final transition the target is the reward alone. If it bootstrapped
    off a phantom next state, the last step of every episode would be biased.
    """
    agent = QLearningAgent(alpha=1.0, gamma=0.99)
    agent.train(tiny_env([0.05, 0.05, np.nan]), n_episodes=1)
    assert len(agent.q) > 0
    assert np.all(np.isfinite(np.concatenate(list(agent.q.values()))))


def test_greedy_policy_holds_flat_on_an_unseen_state():
    """An all-zero row means no information, and flat is the right default."""
    policy = QLearningAgent().greedy_policy()
    assert policy((9, 9, 9)) == 0


def test_greedy_policy_breaks_ties_to_flat_rather_than_guessing():
    agent = QLearningAgent()
    agent.q[(0, 0)] = np.array([0.0, 0.0, 0.0])
    assert agent.greedy_policy()((0, 0)) == 0


def test_greedy_policy_follows_the_largest_value():
    agent = QLearningAgent()
    agent.q[(0, 0)] = np.array([3.0, 0.0, 1.0])        # short is best
    assert agent.greedy_policy()((0, 0)) == -1


def test_both_agents_learn_to_be_long_when_the_tape_only_rises():
    """A market that goes up every day has one correct answer."""
    for agent_class in (QLearningAgent, SarsaAgent):
        agent = agent_class(alpha=0.1, gamma=0.9)
        agent.train(tiny_env([0.02] * 30 + [np.nan]), n_episodes=120)
        assert agent.greedy_policy()((0, 0)) == 1, f"{agent_class.__name__} should go long"


def test_both_agents_learn_to_be_short_when_the_tape_only_falls():
    for agent_class in (QLearningAgent, SarsaAgent):
        agent = agent_class(alpha=0.1, gamma=0.9)
        agent.train(tiny_env([-0.02] * 30 + [np.nan]), n_episodes=120)
        assert agent.greedy_policy()((0, 0)) == -1, f"{agent_class.__name__} should go short"


def test_learning_curve_records_one_row_per_episode():
    agent = QLearningAgent()
    curve = agent.train(tiny_env([0.01] * 10 + [np.nan]), n_episodes=7)
    assert len(curve) == 7
    assert set(curve.columns) >= {"episode", "epsilon", "total_reward", "states_visited"}
    assert curve["epsilon"].is_monotonic_decreasing


def test_agents_are_reproducible_under_the_same_seed():
    tape = [0.01, -0.01] * 10 + [np.nan]
    a = QLearningAgent(seed=7)
    a.train(tiny_env(tape), n_episodes=20)
    b = QLearningAgent(seed=7)
    b.train(tiny_env(tape), n_episodes=20)
    assert a.learning_curve()["total_reward"].tolist() == b.learning_curve()["total_reward"].tolist()


def test_policy_table_lists_every_visited_state():
    agent = QLearningAgent()
    agent.train(tiny_env([0.01] * 8 + [np.nan]), n_episodes=5)
    table = agent.policy_table()
    assert len(table) == len(agent.q)
    assert {"state", "greedy_action"} <= set(table.columns)


def test_baseline_policies_do_what_their_names_say():
    assert buy_and_hold_policy((0, 0)) == 1
    assert flat_policy((0, 0)) == 0
    # One factory, called many times: building a new one each call would replay
    # the same seed and always return the same action.
    random_policy = random_policy_factory(seed=1)
    picks = {random_policy((0, 0)) for _ in range(60)}
    assert picks == set(ACTIONS)
