"""
Tabular reinforcement learning agents: Q-learning and SARSA.

Both are deliberately tabular. The state space here is small and discrete by
construction, so a table is enough, and it has one decisive advantage over a
function approximator for this project: the learned policy can be read directly
and argued with. A neural policy that beats buy-and-hold on one bull run is not
evidence of anything; a table showing "short only in the high-volatility,
below-the-high regime" is a claim a desk can evaluate.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from nvda_rl.config import (
    ACTIONS,
    ALPHA,
    EPSILON_DECAY_EPISODES,
    EPSILON_END,
    EPSILON_START,
    GAMMA,
    N_EPISODES,
    RANDOM_SEED,
)


def _epsilon_at(episode: int, n_decay: int = EPSILON_DECAY_EPISODES) -> float:
    """
    Linear decay from EPSILON_START to EPSILON_END over n_decay episodes.

    Exploring hard early and barely at all late is what lets a tabular agent
    visit enough states to have opinions, then commit to them.
    """
    if episode >= n_decay:
        return EPSILON_END
    fraction = episode / n_decay
    return EPSILON_START + fraction * (EPSILON_END - EPSILON_START)


class TabularAgent:
    """
    Shared machinery for the two tabular learners.

    The Q-table is a defaultdict keyed by the environment's state tuple, so
    unseen states start neutral at zero instead of raising, which is exactly
    what happens on the test window when a market state appears that the
    training years never produced.
    """

    def __init__(self, alpha: float = ALPHA, gamma: float = GAMMA, seed: int = RANDOM_SEED):
        self.alpha = alpha
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.q: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(len(ACTIONS)))
        self.history: list[dict] = []

    def action_index(self, action: int) -> int:
        return ACTIONS.index(action)

    def select_action(self, state: tuple, epsilon: float) -> int:
        """Epsilon-greedy over the three positions."""
        if self.rng.random() < epsilon:
            return ACTIONS[self.rng.integers(len(ACTIONS))]
        return ACTIONS[int(np.argmax(self.q[state]))]

    def greedy_policy(self):
        """
        Freeze the learned table into a deterministic policy.

        Ties resolve to the first action, and a state never seen in training has
        an all-zero row, so the agent holds flat there rather than guessing. For
        a trading agent that default is the right one: do nothing when you have
        no information.
        """
        def policy(state: tuple) -> int:
            values = self.q[state]
            if not np.any(values):
                return 0
            return ACTIONS[int(np.argmax(values))]
        return policy

    def policy_table(self) -> pd.DataFrame:
        """The Q-table as a readable frame, one row per visited state."""
        rows = []
        for state, values in self.q.items():
            rows.append({
                "state": state,
                **{f"Q(a={a})": round(v, 6) for a, v in zip(ACTIONS, values, strict=True)},
                "greedy_action": ACTIONS[int(np.argmax(values))] if np.any(values) else 0,
                "visits_value_sum": round(float(np.abs(values).sum()), 6),
            })
        return pd.DataFrame(rows).sort_values("state").reset_index(drop=True)

    def learning_curve(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)


class QLearningAgent(TabularAgent):
    """
    Off-policy control: the update bootstraps from the best next action rather
    than the one actually taken, so the table converges toward the optimal
    policy even while the behavior policy is still exploring.
    """

    def train(self, env, n_episodes: int = N_EPISODES) -> pd.DataFrame:
        for episode in range(n_episodes):
            epsilon = _epsilon_at(episode)
            state = env.reset()
            total_reward, done = 0.0, False

            while not done:
                action = self.select_action(state, epsilon)
                next_state, reward, done, _ = env.step(action)
                a_idx = self.action_index(action)

                target = reward if done else reward + self.gamma * np.max(self.q[next_state])
                self.q[state][a_idx] += self.alpha * (target - self.q[state][a_idx])

                state = next_state
                total_reward += reward

            self.history.append({
                "episode": episode,
                "epsilon": round(epsilon, 4),
                "total_reward": total_reward,
                "states_visited": len(self.q),
            })
        return self.learning_curve()


class SarsaAgent(TabularAgent):
    """
    On-policy control: the update uses the action the behavior policy actually
    chose next, so the value learned is the value of the policy being followed,
    exploration included. In a market that difference is not academic; SARSA
    prices in the cost of its own mistakes and tends to settle on tamer
    positions than Q-learning.
    """

    def train(self, env, n_episodes: int = N_EPISODES) -> pd.DataFrame:
        for episode in range(n_episodes):
            epsilon = _epsilon_at(episode)
            state = env.reset()
            action = self.select_action(state, epsilon)
            total_reward, done = 0.0, False

            while not done:
                next_state, reward, done, _ = env.step(action)
                a_idx = self.action_index(action)

                if done:
                    target = reward
                    next_action = None
                else:
                    next_action = self.select_action(next_state, epsilon)
                    target = reward + self.gamma * self.q[next_state][self.action_index(next_action)]

                self.q[state][a_idx] += self.alpha * (target - self.q[state][a_idx])

                state, action = next_state, next_action
                total_reward += reward

            self.history.append({
                "episode": episode,
                "epsilon": round(epsilon, 4),
                "total_reward": total_reward,
                "states_visited": len(self.q),
            })
        return self.learning_curve()


def buy_and_hold_policy(state: tuple) -> int:
    """Always long. The benchmark that matters for a stock that went up 200x."""
    return 1


def flat_policy(state: tuple) -> int:
    """Never trade. Earns nothing, costs nothing, and bounds the cost drag."""
    return 0


def random_policy_factory(seed: int = RANDOM_SEED):
    """
    Uniformly random positions: the no-skill reference.

    It is the honest floor for an RL agent, because beating buy-and-hold can
    happen by accident in a drawdown while beating random requires the policy
    to actually depend on the state.
    """
    rng = np.random.default_rng(seed)

    def policy(state: tuple) -> int:
        return ACTIONS[rng.integers(len(ACTIONS))]
    return policy
