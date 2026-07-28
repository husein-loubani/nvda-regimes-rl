"""
Performance accounting for a traded policy.

Everything here reads the per-day ledger the environment produces, so the PnL
quoted in the notebook and the reward the agent optimized are the same number
by construction rather than by coincidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nvda_rl.config import TRADING_DAYS


def performance(ledger: pd.DataFrame, name: str = "policy") -> dict:
    """
    The metric set the brief asks for, plus the risk numbers a desk would want.

    Cumulative PnL is the headline, but on its own it rewards whoever took the
    most risk in a bull market, so it is reported next to drawdown, volatility,
    and turnover. Turnover matters especially here: it is the quantity the
    transaction cost multiplies, and the gap between gross and net return is
    the whole argument for whether a strategy survives contact with a broker.
    """
    net = ledger["net_return"].to_numpy(dtype=float)
    gross = ledger["gross_return"].to_numpy(dtype=float)
    n_days = len(net)

    equity = np.cumprod(1 + net)
    total_growth = float(equity[-1])
    ann_return = total_growth ** (TRADING_DAYS / n_days) - 1
    ann_vol = net.std(ddof=1) * np.sqrt(TRADING_DAYS)
    drawdown = equity / np.maximum.accumulate(equity) - 1

    traded = ledger["turnover"].to_numpy(dtype=float)
    active = net[ledger["position_held"].to_numpy() != 0]

    return {
        "policy": name,
        "cumulative_pnl_%": round((total_growth - 1) * 100, 2),
        "total_growth_x": round(total_growth, 3),
        "ann_return_%": round(ann_return * 100, 2),
        "ann_vol_%": round(ann_vol * 100, 2),
        "sharpe": round(ann_return / ann_vol, 2) if ann_vol > 0 else np.nan,
        "max_drawdown_%": round(float(drawdown.min()) * 100, 2),
        # Hit ratio is computed over days with a position on; counting flat days
        # as neither win nor loss keeps it a statement about decisions taken.
        "hit_ratio": round(float((active > 0).mean()), 3) if len(active) else np.nan,
        "days_in_market_%": round(float((ledger["position_held"] != 0).mean()) * 100, 1),
        "turnover_per_year": round(float(traded.sum()) / n_days * TRADING_DAYS, 1),
        "cost_drag_%": round(float((gross.sum() - net.sum()) * 100), 2),
        "days": n_days,
    }


def compare_policies(ledgers: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per policy, sorted by cumulative PnL."""
    rows = [performance(ledger, name) for name, ledger in ledgers.items()]
    return (
        pd.DataFrame(rows)
        .set_index("policy")
        .sort_values("cumulative_pnl_%", ascending=False)
    )


def equity_curves(ledgers: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aligned equity curves for plotting, indexed by date."""
    frames = {name: led.set_index("date")["equity"] for name, led in ledgers.items()}
    return pd.DataFrame(frames)


def action_distribution(ledgers: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    How often each policy chose short, flat, and long.

    A policy whose distribution is nearly all one action is not reacting to the
    state, whatever its PnL says, and that is worth seeing before celebrating
    any result.
    """
    rows = {}
    for name, led in ledgers.items():
        counts = led["action"].value_counts(normalize=True).mul(100).round(1)
        rows[name] = {f"action_{a}_%": counts.get(a, 0.0) for a in (-1, 0, 1)}
    return pd.DataFrame(rows).T


def regime_action_map(ledger: pd.DataFrame, regime_labels: pd.Series) -> pd.DataFrame:
    """
    What the agent decided to do inside each discovered regime.

    This is the table that answers the project's central question directly: if
    the regime feature is doing any work, the action mix has to differ across
    regimes. If every regime produces the same position, the unsupervised layer
    is decoration.
    """
    joined = ledger.copy()
    joined["regime"] = regime_labels.to_numpy()[: len(joined)]
    out = (
        joined.groupby("regime")["action"]
        .value_counts(normalize=True)
        .mul(100).round(1)
        .unstack(fill_value=0.0)
        .rename(columns={-1: "short_%", 0: "flat_%", 1: "long_%"})
    )
    out["days"] = joined.groupby("regime").size()
    out["mean_net_return_bps"] = (
        joined.groupby("regime")["net_return"].mean().mul(1e4).round(2)
    )
    return out
