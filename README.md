# NVDA market regimes and a reinforcement learning trader

**Module 3 · Sprint 4: Unsupervised and reinforcement learning** | Turing College · Data Science Program

Clustering finds market regimes in NVIDIA's daily price history. A tabular RL agent then trades the stock with those regimes in its state. The question the project answers is whether the unsupervised layer actually makes the agent decide better.

---

## Headline result

The regime label is built from features the agent already had, yet adding it roughly **doubles** test PnL for both algorithms.

| Policy | Test PnL | Ann. return | Sharpe | Max drawdown | Turnover / yr |
|---|---|---|---|---|---|
| Buy and hold | +1381% | 62.7% | 1.23 | −66.3% | 0.2 |
| **SARSA [market + regime]** | **+1360%** | **62.3%** | **1.25** | −67.2% | 16.1 |
| Q-learning [market + regime] | +749% | 47.1% | 0.93 | −66.7% | 26.2 |
| SARSA [market only] | +706% | 45.8% | 0.90 | −60.0% | 13.2 |
| Q-learning [market only] | +408% | 34.1% | 0.67 | −61.3% | 19.0 |
| Random | −96% | −44.1% | −1.05 | −96.5% | 221.3 |

Sealed test window: 2021-01-04 to 2026-07-28, 1,397 trading days.

Two things to be clear about. Against buy-and-hold the best agent **ties** rather than wins: 1360% against 1381%, with a Sharpe gap that sits inside noise. And the test period is one huge bull run where every drawdown recovered, which flatters anything that stays long. The claim worth defending is the ablation, not the baseline race.

---

## What the project does

- **Unsupervised:** cluster daily returns and volatility into regimes I can name, then check they persist and hold up out of sample.
- **Reinforcement learning:** learn a long / flat / short policy under transaction costs, with the regime label in the state.

The reward, exactly as the brief specifies:

$$r_t = a_{t-1} \cdot \text{return}_t - \text{cost} \cdot |a_t - a_{t-1}|$$

Yesterday's position earns today's return, so the agent cannot act on a return it has already seen. The cost scales with how far the position moves. Unit tests pin this formula with hand-computed values.

---

## The regimes

Three regimes, fitted on the training window and frozen before touching test data:

| Regime | Name | Share of days | Ann. return | Ann. vol | Hit ratio | Below 52w high |
|---|---|---|---|---|---|---|
| 0 | Melt-up | 11% | +506% | 69% | 0.70 | −6% |
| 1 | Calm advance | 62% | +42% | 26% | 0.54 | −6% |
| 2 | Drawdown | 26% | −143% | 51% | 0.42 | −36% |

They persist, with diagonal transition probabilities of 0.69, 0.93, and 0.89. And they generalize: centroids fitted on 2010 to 2020 give the same ordering on 2021 to 2026 without being refitted.

---

## Repository structure

```
.
├── data/raw/NVDA.csv               <- yfinance cache, keeps reruns offline
├── notebooks/
│   └── nvda_regimes_rl.ipynb       <- the deliverable
├── nvda_rl/                        <- all logic; the notebook stays thin
│   ├── config.py                   <- constants, seeds, palette, hyperparameters
│   ├── dataset.py                  <- download, cache, load, audit, clean, split
│   ├── features.py                 <- market features, train-only binning
│   ├── plots.py                    <- every figure, returns a Figure
│   └── modeling/
│       ├── regimes.py              <- K-means, HMM, DBSCAN, Ward, PCA, anomalies
│       ├── environment.py          <- the trading MDP and its reward
│       ├── agents.py               <- Q-learning, SARSA, and baselines
│       └── evaluate.py             <- PnL, drawdown, hit ratio, turnover, cost drag
├── tests/                          <- 19 tests, including leakage guards
├── references/data_dictionary.md
├── reports/figures/
├── pyproject.toml
└── uv.lock                         <- exact pinned versions, committed
```

---

## Guarding against look-ahead

The dangerous failure here is not a crash. It is a backtest that looks great because the state encoded the future. Three guards, all covered by tests:

1. The split is chronological. Train before 2021, test after, never shuffled.
2. The scaler and cluster centroids are fitted on train only, then frozen, so a test-day label cannot reflect days the agent never traded.
3. The discretization bin edges come from training quantiles, so the test period never decides what counts as high volatility.

`tests/test_leakage.py` asserts that centroids do not move when test data arrives and that bin edges depend only on training data.

---

## How to run

The notebook ships with the `turing-college` kernel selected. Install the package into that environment once:

```bash
conda activate turing-college
cd "Module 3/Sprint 4"
pip install -e .             # installs nvda_rl and its dependencies
jupyter lab notebooks/nvda_regimes_rl.ipynb
```

Without that install the first cell raises `ModuleNotFoundError: No module named 'nvda_rl'`, because the notebook imports the project as a package rather than reaching for files by path.

There is also a uv path, which is what `uv.lock` pins:

```bash
uv sync --extra dev          # builds .venv from the lockfile
uv run pytest -q             # 19 tests
uv run ruff check .          # package, tests, and notebook
```

Both environments give identical results. The conda env runs Python 3.10 with scikit-learn 1.7.2 and the uv env runs 3.12 with 1.9.0, and every figure in this README came out the same on both.

The notebook runs top to bottom. The first run downloads NVDA from yfinance; later runs read the CSV cache, so no network is needed. Everything stochastic is seeded with `RANDOM_SEED = 42`.

---

## Sprint 4 coverage

- Clustering: K-means with elbow, silhouette, Davies-Bouldin, and Calinski-Harabasz; DBSCAN; agglomerative with Ward linkage
- Dimensionality reduction: PCA with named loadings, UMAP
- Sequence model: Gaussian HMM, for regimes that persist rather than flicker
- Anomaly detection: Isolation Forest and Local Outlier Factor, used to bound how far the test window extrapolates
- Cluster evaluation: internal metrics only, since market days have no ground-truth regime
- RL: MDP formulation, ε-greedy with decay, tabular Q-learning (off-policy) and SARSA (on-policy)
- Evaluation: cumulative PnL against buy-and-hold, random, and flat, plus Sharpe, max drawdown, hit ratio, turnover, and cost drag
