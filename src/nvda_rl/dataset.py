"""
Download, cache, load, audit, and split the NVDA daily price series.

Data flows: yfinance -> data/raw/NVDA.csv -> one clean frame indexed by date.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nvda_rl.config import SPLIT_DATE, START_DATE, TICKER, VALIDATION_DATE


def download_data(raw_dir: str | Path, force: bool = False) -> Path:
    """
    Download daily OHLCV from START_DATE and cache it as one CSV. Skips the
    download when the cache exists unless force=True, so the notebook reruns
    offline and reproduces identical numbers.

    Prices come dividend- and split-adjusted (auto_adjust=True). NVDA split
    repeatedly over this period, so unadjusted closes would show enormous
    fictitious overnight losses that no return model should ever see.
    """
    import yfinance as yf

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / f"{TICKER}.csv"

    if out.exists() and not force:
        print(f"  {TICKER} cached ({out.name})")
        return out

    df = yf.download(TICKER, start=START_DATE, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.to_csv(out)
    print(f"  {TICKER} downloaded: {len(df):,} rows "
          f"({df.index.min().date()} to {df.index.max().date()})")
    return out


def load_prices(raw_dir: str | Path) -> pd.DataFrame:
    """
    Read the cached CSV into a date-sorted frame of raw prices.

    Returns are deliberately NOT computed here. They are derived in
    `clean_prices` after the bad rows are gone, because a return computed
    against a row that cleaning later deletes would silently survive in the
    following row.
    """
    path = Path(raw_dir) / f"{TICKER}.csv"
    df = pd.read_csv(path, parse_dates=["Date"]).rename(columns=str.lower)
    return df.sort_values("date").reset_index(drop=True)


def audit_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-row summary of the things that silently ruin a price series: gaps,
    duplicates, missing values, and non-positive prices.

    Runs on raw prices as well as cleaned ones, so the return statistics are
    derived here rather than assumed to be present. That keeps the audit usable
    before cleaning, which is exactly when it is most informative.
    """
    gaps = df["date"].diff().dt.days.dropna()
    returns = df["return"] if "return" in df else df["close"].pct_change()
    summary = {
        "rows": len(df),
        "first_date": df["date"].min().date(),
        "last_date": df["date"].max().date(),
        "duplicate_dates": int(df["date"].duplicated().sum()),
        "missing_close": int(df["close"].isna().sum()),
        "missing_volume": int(df["volume"].isna().sum()),
        "non_positive_close": int((df["close"] <= 0).sum()),
        "max_gap_days": int(gaps.max()),
        "mean_daily_return_%": round(returns.mean() * 100, 4),
        "daily_vol_%": round(returns.std() * 100, 3),
    }
    return pd.DataFrame([summary]).T.rename(columns={0: TICKER})


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the price sequence first, then derive returns from what survives.

    The order matters. Removing a duplicate date or a non-positive close after
    the returns are computed leaves the following row holding a return measured
    against a bar that no longer exists. Cleaning first and differencing second
    makes every return a difference between two rows that are both still in the
    frame.

    Extreme daily moves are deliberately kept. NVDA's largest single-day jumps
    are real earnings reactions, and they are precisely the events a regime
    model exists to identify, so trimming them would delete the signal.
    """
    out = df.drop_duplicates(subset="date", keep="first")
    out = out[out["close"].notna() & (out["close"] > 0)]
    out = out.sort_values("date").reset_index(drop=True)

    out["return"] = out["close"].pct_change()
    out["log_return"] = np.log(out["close"]).diff()

    # The opening row has no predecessor, so its return is undefined.
    return out.dropna(subset=["return"]).reset_index(drop=True)


def extreme_days(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """The n largest daily gains and losses, for the anomaly discussion."""
    cols = ["date", "close", "return", "volume"]
    top = df.nlargest(n, "return")[cols].assign(kind="largest gain")
    bottom = df.nsmallest(n, "return")[cols].assign(kind="largest loss")
    out = pd.concat([top, bottom], ignore_index=True)
    out["return_%"] = (out["return"] * 100).round(2)
    return out.drop(columns="return")


def split_prices(df: pd.DataFrame, split_date: str = SPLIT_DATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split: train strictly before split_date, test from it onward.

    Everything fitted downstream, the scaler, the regime models, and the agent's
    Q-table, sees the training window only. A random split would let the agent
    trade days it had already learned the outcome of.
    """
    cut = pd.Timestamp(split_date)
    train = df[df["date"] < cut].reset_index(drop=True)
    test = df[df["date"] >= cut].reset_index(drop=True)
    return train, test


def split_three_way(
    df: pd.DataFrame,
    validation_date: str = VALIDATION_DATE,
    split_date: str = SPLIT_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological train / validation / test split.

    Every hyperparameter is chosen against the validation window so the test
    window stays sealed until the single final evaluation. Without this, a
    choice like the learning rate is effectively fitted to the test set and the
    reported result stops being out of sample.
    """
    v, t = pd.Timestamp(validation_date), pd.Timestamp(split_date)
    train = df[df["date"] < v].reset_index(drop=True)
    validation = df[(df["date"] >= v) & (df["date"] < t)].reset_index(drop=True)
    test = df[df["date"] >= t].reset_index(drop=True)
    return train, validation, test
