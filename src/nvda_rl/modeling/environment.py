"""
The NVDA trading environment.

A deterministic replay of a fixed historical tape, written against the
Gymnasium `Env` contract so the action space, observation space, seeding, and
step signature follow a standard interface rather than project-local
conventions.

Timing is the thing this file has to get right, so it is stated plainly:

    at the close of day t the agent observes state_t and chooses position a_t,
    and that position earns the return of day t+1.

`next_return[t]` is exactly that return, so the reward multiplies it by the
action taken now, not by the position carried in. An earlier version of this
environment multiplied `next_return[t]` by the *previous* position, which
shifted the payoff a second time and meant every position earned the return of
the day after the one it should have. The regression test in
`tests/modeling/test_environment.py` pins the correct behaviour with named
weekdays and a hand-computed reward.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from nvda_rl.config import ACTIONS, TRANSACTION_COST


class TradingEnv(gym.Env):
    """
    Single-asset daily trading environment over a fixed price history.

    Reward:

        r_t = a_t * next_return_t - cost * |a_t - a_{t-1}|

    which is the specification `a_{t-1} * return_t - cost * |a_t - a_{t-1}|`
    written from the decision's point of view rather than the settlement's: the
    position chosen at t is the one that earns t+1's return. Summed over an
    episode the two indexings give the same total, but only this one credits the
    action that actually caused the profit, which is what the Q-update needs.

    The cost scales with the size of the position change, so flipping from short
    to long costs twice what moving from flat to long does.

    Observation: the discretized market features followed by the position held
    coming into the step. Carrying the position makes this a genuine MDP, since
    the cost of the next action depends on it.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        frame: pd.DataFrame,
        state_columns: list[str],
        cost: float = TRANSACTION_COST,
        return_column: str = "next_return",
    ):
        super().__init__()
        missing = [c for c in [*state_columns, return_column] if c not in frame.columns]
        if missing:
            raise KeyError(f"frame is missing required columns: {missing}")

        self.frame = frame.reset_index(drop=True)
        self.state_columns = list(state_columns)
        self.cost = float(cost)
        self.return_column = return_column

        # The last row's next_return is undefined, so it cannot be traded.
        self.n_steps = int(self.frame[return_column].notna().sum())
        if self.n_steps < 1:
            raise ValueError("frame needs at least one row with a next-day return")

        self.returns = self.frame[return_column].to_numpy(dtype=float)
        self.states = self.frame[self.state_columns].to_numpy()

        self.action_space = spaces.Discrete(len(ACTIONS))
        highs = self.states.max(axis=0).astype(np.int64)
        self.observation_space = spaces.MultiDiscrete(
            np.append(highs + 1, len(ACTIONS))
        )
        self.reset()

    def _observe(self) -> tuple:
        """Market features plus the position held, as a hashable tuple."""
        return (*self.states[self.t].tolist(), self.position)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Start a new episode flat at the first tradeable day."""
        super().reset(seed=seed)
        self.t = 0
        self.position = 0
        return self._observe(), {}

    def step(self, action: int):
        """
        Apply an action and advance one trading day.

        Returns the Gymnasium five-tuple: observation, reward, terminated,
        truncated, info. `action` is the position to hold into the next day.
        """
        if action not in ACTIONS:
            raise ValueError(f"action {action!r} is not one of {ACTIONS}")

        previous_position = self.position
        turnover = abs(action - previous_position)

        # The position chosen now earns tomorrow's return, which next_return
        # already holds on this row.
        market_return = self.returns[self.t]
        gross = action * market_return
        transaction_cost = self.cost * turnover
        reward = gross - transaction_cost

        self.position = action
        self.t += 1
        terminated = self.t >= self.n_steps

        info = {
            "gross_return": gross,
            "cost": transaction_cost,
            "turnover": turnover,
            "market_return": market_return,
            "position_entered": action,
            "position_before": previous_position,
        }
        observation = None if terminated else self._observe()
        return observation, reward, terminated, False, info

    def run_policy(self, policy_fn) -> pd.DataFrame:
        """
        Replay the history under a deterministic policy and return a per-day
        ledger. Every downstream metric reads this frame, so the accounting
        lives in exactly one place.
        """
        state, _ = self.reset()
        rows = []
        terminated = False

        while not terminated:
            step_index = self.t
            action = policy_fn(state)
            next_state, reward, terminated, _, info = self.step(action)
            rows.append({
                "date": self.frame.loc[step_index, "date"],
                "market_return": info["market_return"],
                "position_before": info["position_before"],
                "action": action,
                "gross_return": info["gross_return"],
                "cost": info["cost"],
                "turnover": info["turnover"],
                "net_return": reward,
            })
            state = next_state

        ledger = pd.DataFrame(rows)
        ledger["equity"] = (1 + ledger["net_return"]).cumprod()
        return ledger
