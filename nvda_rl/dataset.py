"""
Download, cache, load, audit, and split the NVDA daily price series.

Data flows: yfinance -> data/raw/NVDA.csv -> one clean frame indexed by date.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nvda_rl.config import SPLIT_DATE, START_DATE, TICKER


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
    Read the cached CSV into a frame indexed by date, with simple and log
    returns attached. Log returns are used wherever returns get summed over
    time; simple returns are what the trading environment pays out.
    """
    path = Path(raw_dir) / f"{TICKER}.csv"
    df = pd.read_csv(path, parse_dates=["Date"]).rename(columns=str.lower)
    df = df.rename(columns={"date": "date"}).sort_values("date").reset_index(drop=True)

    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"]).diff()
    return df


def audit_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-row summary of the things that silently ruin a price series: gaps,
    duplicates, missing values, and non-positive prices.
    """
    gaps = df["date"].diff().dt.days.dropna()
    summary = {
        "rows": len(df),
        "first_date": df["date"].min().date(),
        "last_date": df["date"].max().date(),
        "duplicate_dates": int(df["date"].duplicated().sum()),
        "missing_close": int(df["close"].isna().sum()),
        "missing_volume": int(df["volume"].isna().sum()),
        "non_positive_close": int((df["close"] <= 0).sum()),
        "max_gap_days": int(gaps.max()),
        "mean_daily_return_%": round(df["return"].mean() * 100, 4),
        "daily_vol_%": round(df["return"].std() * 100, 3),
    }
    return pd.DataFrame([summary]).T.rename(columns={0: TICKER})


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicate dates, rows without a usable close, and the first row whose
    return is undefined by construction.

    Extreme daily moves are deliberately kept. NVDA's largest single-day jumps
    are real earnings reactions, and they are precisely the events a regime
    model exists to identify, so trimming them would delete the signal.
    """
    out = df.drop_duplicates(subset="date", keep="first")
    out = out[out["close"] > 0]
    out = out.dropna(subset=["close", "return"])
    return out.reset_index(drop=True)


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
