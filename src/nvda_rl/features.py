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


def fit_binner(train: pd.DataFrame, columns: list[str], n_bins: int = 3):
    """
    Fit sklearn's KBinsDiscretizer on the training window only.

    Tabular Q-learning needs a finite state space, so the continuous features
    have to be discretized. This used to be hand-rolled with `quantile` plus
    `np.digitize`; KBinsDiscretizer provides the same fit-on-train,
    transform-on-test contract as one tested component, and it is consistent
    with the rest of the pipeline, which is sklearn throughout.

    Fitting on the full history would let the test period decide what counts as
    "high volatility", which is a quiet but real leak, so the encoder is frozen
    here and reused unchanged.
    """
    from sklearn.preprocessing import KBinsDiscretizer

    encoder = KBinsDiscretizer(
        n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None
    )
    encoder.fit(train[columns])
    return {"encoder": encoder, "columns": list(columns)}


def apply_binner(df: pd.DataFrame, binner: dict, suffix: str = "_bin") -> pd.DataFrame:
    """
    Map the continuous columns onto the frozen bins.

    Values beyond the training range clip into the outer bins rather than
    raising, which is the honest behaviour: an unprecedented day is still the
    most extreme bin the agent knows about.
    """
    out = df.copy()
    encoded = binner["encoder"].transform(df[binner["columns"]])
    for i, col in enumerate(binner["columns"]):
        out[f"{col}{suffix}"] = encoded[:, i].astype(int)
    return out
