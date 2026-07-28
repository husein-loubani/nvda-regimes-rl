"""
Feature engineering for the regime models and the RL state.

Every feature at day t is computed from information available up to and
including day t's close, and the environment only ever pays day t+1's return
for a position taken at t. That ordering is what keeps the whole pipeline free
of look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nvda_rl.config import (
    HIGH_LOOKBACK,
    MOMENTUM_WINDOW,
    REGIME_FEATURES,
    VOL_WINDOW,
)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the market-state features used for regime detection and as RL state.

    The set is deliberately small and each column has a plain-language meaning,
    because a regime is only actionable if a trader can be told what it is:
    how much the market moved, how violently it has been moving, whether it has
    been trending, how far it sits below its yearly high, and whether volume is
    unusual.
    """
    out = df.copy()

    out["volatility_21"] = out["log_return"].rolling(VOL_WINDOW).std() * np.sqrt(252)
    out["momentum_21"] = out["close"].pct_change(MOMENTUM_WINDOW)
    out["dist_252d_high"] = out["close"] / out["close"].rolling(HIGH_LOOKBACK).max() - 1

    volume_mean = out["volume"].rolling(HIGH_LOOKBACK).mean()
    volume_std = out["volume"].rolling(HIGH_LOOKBACK).std()
    out["volume_zscore"] = (out["volume"] - volume_mean) / volume_std

    # The environment pays the next day's return for a position taken today, so
    # it is carried as an explicit column rather than recomputed by the agent.
    out["next_return"] = out["return"].shift(-1)

    return out


def drop_warmup(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Remove the leading rows where rolling windows have not filled yet.

    The 252-day high needs a year of history, so the first trading year cannot
    produce a complete feature row. Dropping those rows is honest; imputing
    them would invent a market state that never existed.
    """
    columns = columns or REGIME_FEATURES
    return df.dropna(subset=columns).reset_index(drop=True)


def regime_matrix(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """The feature block handed to the unsupervised models, in a fixed order."""
    columns = columns or REGIME_FEATURES
    return df[columns].copy()


def fit_binner(train: pd.DataFrame, columns: list[str], n_bins: int = 3) -> dict[str, np.ndarray]:
    """
    Learn quantile bin edges from the training window only.

    Tabular Q-learning needs a finite state space, so continuous features have
    to be discretized. Computing the quantiles over the full history would let
    the test period decide what counts as "high volatility", which is a quiet
    but real leak: the agent would be told where today sits in a distribution
    that includes days it has not traded yet. Edges are therefore frozen here
    and reused unchanged on the test window.
    """
    edges = {}
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    for col in columns:
        edges[col] = np.unique(train[col].quantile(quantiles).to_numpy())
    return edges


def apply_binner(df: pd.DataFrame, edges: dict[str, np.ndarray], suffix: str = "_bin") -> pd.DataFrame:
    """
    Map continuous columns onto the frozen bins.

    Values beyond the training range fall into the outer bins rather than
    raising, which is the honest behavior: an unprecedented day is still the
    most extreme bin the agent knows about.
    """
    out = df.copy()
    for col, cuts in edges.items():
        out[f"{col}{suffix}"] = np.digitize(out[col].to_numpy(), cuts).astype(int)
    return out
