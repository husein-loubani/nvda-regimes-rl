"""
The NVDA trading environment.

A minimal, fully deterministic episodic environment in the Gymnasium spirit,
written directly rather than wrapped around gymnasium because the dynamics are
a fixed historical tape: there is nothing to simulate, only to replay. Keeping
it explicit also makes the reward auditable, which matters more here than API
conformance since the reward is the one thing the whole project rests on.
"""

from __future__ import annotations

import pandas as pd

from nvda_rl.config import ACTIONS, TRANSACTION_COST


class TradingEnv:
    """
    Single-asset daily trading environment over a fixed price history.

    The reward implements the specification exactly:

        r_t = a_{t-1} * return_t  -  cost * |a_t - a_{t-1}|

    Two details in that formula do the real work. The position carried in from
    the previous step, not the one chosen now, earns today's return, which is
    what stops the agent from acting on a return it has already observed. And
    the cost scales with the *size* of the position change, so flipping from
    short to long pays twice what moving from flat to long does, exactly as a
    real book would.

    State: a tuple of discretized market features, optionally including the
    unsupervised regime label, plus the current position. Carrying the position
    in the state is what makes the problem a genuine MDP rather than a sequence
    of independent bets, because the cost of the next action depends on it.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        state_columns: list[str],
        cost: float = TRANSACTION_COST,
        return_column: str = "next_return",
    ):
        missing = [c for c in state_columns + [return_column] if c not in frame.columns]
        if missing:
            raise KeyError(f"frame is missing required columns: {missing}")

        self.frame = frame.reset_index(drop=True)
        self.state_columns = list(state_columns)
        self.cost = float(cost)
        self.return_column = return_column

        # The final row has no next-day return to pay out, so it cannot be traded.
        self.n_steps = len(self.frame) - 1
        if self.n_steps < 1:
            raise ValueError("frame needs at least two rows to form one transition")

        self.returns = self.frame[return_column].to_numpy(dtype=float)
        self.states = self.frame[self.state_columns].to_numpy()
        self.dates = self.frame["date"].to_numpy() if "date" in self.frame else None

        self.reset()

    def reset(self) -> tuple:
        """Start a new episode flat at the first tradeable day."""
        self.t = 0
        self.position = 0
        return self._observe()

    def _observe(self) -> tuple:
        """
        The discrete state: market features plus the position currently held.

        Features are already discretized upstream, so the tuple is hashable and
        can index a Q-table directly.
        """
        return (*self.states[self.t].tolist(), self.position)

    def step(self, action: int) -> tuple[tuple, float, bool, dict]:
        """
        Apply an action and advance one trading day.

        Returns (next_state, reward, done, info). `action` is the position to
        hold going into the next day, taken from {-1, 0, +1}.
        """
        if action not in ACTIONS:
            raise ValueError(f"action {action!r} is not one of {ACTIONS}")

        previous_position = self.position
        turnover = abs(action - previous_position)

        # The position held coming into this step earns today's realized return;
        # the trade into the new position is charged at the same instant.
        market_return = self.returns[self.t]
        gross = previous_position * market_return
        transaction_cost = self.cost * turnover
        reward = gross - transaction_cost

        self.position = action
        self.t += 1
        done = self.t >= self.n_steps

        info = {
            "gross_return": gross,
            "cost": transaction_cost,
            "turnover": turnover,
            "market_return": market_return,
            "position": action,
        }
        return (self._observe() if not done else None), reward, done, info

    def run_policy(self, policy_fn) -> pd.DataFrame:
        """
        Replay the whole history under a deterministic policy and return a
        per-day ledger of what happened.

        Everything downstream (PnL curves, drawdown, turnover, hit ratio) reads
        this frame, so the accounting exists in exactly one place.
        """
        self.reset()
        rows = []
        state = self._observe()
        done = False

        while not done:
            step_index = self.t
            action = policy_fn(state)
            next_state, reward, done, info = self.step(action)
            rows.append({
                "date": self.frame.loc[step_index, "date"],
                "market_return": info["market_return"],
                # The last element of the pre-step state is the position that
                # was carried in, and therefore the one that earned this return.
                "position_held": state[-1],
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
