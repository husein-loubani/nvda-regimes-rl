"""Tests for loading, cleaning, and splitting the price series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nvda_rl.dataset import clean_prices, extreme_days, split_prices


def make_prices(n: int = 40, start: str = "2019-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range(start, periods=n)
    close = 100 * np.cumprod(1 + rng.normal(0.001, 0.02, n))
    df = pd.DataFrame({
        "date": dates,
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": rng.integers(1e6, 5e6, n),
    })
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"]).diff()
    return df


def test_first_row_without_a_return_is_dropped():
    """The opening row has no previous close, so its return is undefined."""
    cleaned = clean_prices(make_prices(20))
    assert cleaned["return"].notna().all()
    assert len(cleaned) == 19


def test_duplicate_dates_are_removed():
    df = make_prices(20)
    doubled = pd.concat([df, df.iloc[[5]]], ignore_index=True).sort_values("date")
    cleaned = clean_prices(doubled)
    assert cleaned["date"].duplicated().sum() == 0


def test_non_positive_close_is_removed():
    df = make_prices(20)
    df.loc[7, "close"] = 0.0
    cleaned = clean_prices(df)
    assert (cleaned["close"] > 0).all()


def test_split_is_chronological_and_disjoint():
    """
    Every training date must precede every test date. A random split would let
    the agent learn from days that come after the ones it is evaluated on.
    """
    df = clean_prices(make_prices(200, start="2019-01-01"))
    cut = "2019-07-01"
    train, test = split_prices(df, cut)

    assert len(train) + len(test) == len(df)
    assert train["date"].max() < pd.Timestamp(cut)
    assert test["date"].min() >= pd.Timestamp(cut)
    assert train["date"].max() < test["date"].min()
    assert set(train["date"]).isdisjoint(set(test["date"]))


def test_extreme_days_reports_both_tails():
    df = clean_prices(make_prices(60))
    out = extreme_days(df, n=3)
    assert len(out) == 6
    assert set(out["kind"]) == {"largest gain", "largest loss"}
    assert out.loc[out["kind"] == "largest gain", "return_%"].min() >= \
           out.loc[out["kind"] == "largest loss", "return_%"].max()
