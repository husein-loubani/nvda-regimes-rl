# Data dictionary

## Source

NVIDIA (NVDA) daily bars from Yahoo Finance via `yfinance`, from 2010-01-01 to the build date, cached to `data/raw/NVDA.csv`.

Prices arrive dividend- and split-adjusted (`auto_adjust=True`). This is not cosmetic: NVDA split several times over the period, and unadjusted closes would show enormous fictitious overnight losses on the split dates, which a regime model would dutifully classify as crashes.

## Raw columns

| Column | Type | Description |
|---|---|---|
| `date` | datetime | Trading day |
| `open` | float | Adjusted opening price (USD) |
| `high` | float | Adjusted session high (USD) |
| `low` | float | Adjusted session low (USD) |
| `close` | float | Adjusted closing price (USD) |
| `volume` | int | Shares traded |

## Derived columns

| Column | Definition | Why it exists |
|---|---|---|
| `return` | `close.pct_change()` | Simple daily return, the quantity the environment pays out |
| `log_return` | `log(close).diff()` | Used wherever returns are summed or a standard deviation is taken |
| `volatility_21` | 21-day rolling std of `log_return`, annualized | How violently the market is moving, the strongest regime separator |
| `momentum_21` | 21-day percentage price change | Whether the market has been trending, and in which direction |
| `dist_252d_high` | `close / close.rolling(252).max() - 1` | Distance below the yearly high; separates drawdowns from advances |
| `rsi_14` | Wilder's 14-day relative strength index | Bounded momentum oscillator, reported in EDA |
| `volume_zscore` | Volume standardized against a rolling 252-day mean and std | Unusual participation; standardized because volume grew structurally |
| `return_lag_{1,2,3,5}` | Lagged returns | Short-horizon history available to the state |
| `next_return` | `return.shift(-1)` | The return a position taken today will earn tomorrow |
| `regime` | K-means cluster label, ordered by mean return | The unsupervised state feature; 0 is the highest-return regime |
| `*_bin` | Tercile index from train-only quantile edges | Discretization for the tabular Q-table |
| `anomaly_iso` | Isolation Forest prediction, `-1` anomalous | Bounds how far the test window extrapolates from training conditions |

## Feature timing

Every feature at day *t* uses information available at *t*'s close and no later. The environment pays day *t+1*'s return for a position taken at *t*. That ordering is what keeps the pipeline free of look-ahead, and `next_return` is carried as an explicit column so the timing is visible rather than implied.

## Warm-up rows

`dist_252d_high` needs a full trading year of history, so the first 252 rows cannot produce a complete feature vector and are dropped rather than imputed. Filling them would invent market states that never existed.

## Regime labels

| Label | Name | Character |
|---|---|---|
| 0 | Melt-up | Highest mean return, high volatility, near the yearly high |
| 1 | Calm advance | Modest positive drift, lowest volatility, the majority of days |
| 2 | Drawdown | Negative drift, high volatility, far below the yearly high |

Labels are reordered after fitting so regime 0 always has the highest mean return. K-means numbers clusters arbitrarily, so without that step the same regime would change identity between runs and every written interpretation would silently rot.
