# NVDA market regimes and a reinforcement learning trader

**Module 3 · Sprint 4: Unsupervised and reinforcement learning** | Turing College · Data Science Program

Unsupervised learning discovers interpretable market regimes in NVIDIA's daily price history; a tabular reinforcement learning agent then trades the stock using those regimes as part of its state. The project's central question is whether the unsupervised layer measurably improves the agent's decisions.

---

## Headline result

The regime label is derived entirely from features the agent already had, yet adding it to the state roughly **doubles** cumulative test PnL for both algorithms.

| Policy | Test PnL | Ann. return | Sharpe | Max drawdown | Turnover / yr |
|---|---|---|---|---|---|
| Buy and hold | +1381% | 62.7% | 1.23 | −66.3% | 0.2 |
| **SARSA [market + regime]** | **+1360%** | **62.3%** | **1.25** | −67.2% | 16.1 |
| Q-learning [market + regime] | +749% | 47.1% | 0.93 | −66.7% | 26.2 |
| SARSA [market only] | +706% | 45.8% | 0.90 | −60.0% | 13.2 |
| Q-learning [market only] | +408% | 34.1% | 0.67 | −61.3% | 19.0 |
| Random | −96% | −44.1% | −1.05 | −96.5% | 221.3 |

Sealed test window: 2021-01-04 to 2026-07-28 (1,397 trading days).

Two honest qualifications. Against buy-and-hold the best agent **ties** rather than wins: 1360% against 1381%, with a marginally better Sharpe that sits well inside noise. And the test period is a single extraordinary bull run in which every drawdown recovered, which flatters anything that stays long. The defensible claim is the ablation, not the baseline race.

---

## Project goal

Financial markets are noisy and non-stationary, which makes them a natural setting for combining the two paradigms:

- **Unsupervised (exploration):** cluster daily returns and volatility into regimes that a trader can name, and verify they persist and generalize out of sample.
- **Reinforcement learning (decision-making):** learn a long / flat / short policy under transaction costs, with the regime label in the state space.

Reward, exactly as specified in the brief:

$$r_t = a_{t-1} \cdot \text{return}_t - \text{cost} \cdot |a_t - a_{t-1}|$$

The position carried in from yesterday earns today's return, so the agent cannot act on a return it has already seen, and the cost scales with the size of the position change. This formula is pinned by unit tests with hand-computed values.

---

## What the unsupervised layer found

Three regimes, selected on the training window and frozen before ever touching test data:

| Regime | Name | Share of days | Ann. return | Ann. vol | Hit ratio | Distance from 52w high |
|---|---|---|---|---|---|---|
| 0 | Melt-up | 11% | +506% | 69% | 0.70 | −6% |
| 1 | Calm advance | 62% | +42% | 26% | 0.54 | −6% |
| 2 | Drawdown | 26% | −143% | 51% | 0.42 | −36% |

They **persist** (diagonal transition probabilities 0.69, 0.93, 0.89) and they **generalize**: centroids fitted on 2010–2020 reproduce the same economic ordering on 2021–2026 without being refitted.

---

## Repository structure

```
.
├── data/raw/NVDA.csv               <- yfinance cache, keeps reruns offline and deterministic
├── notebooks/
│   └── nvda_regimes_rl.ipynb       <- the deliverable
├── nvda_rl/                        <- all logic lives here; the notebook stays thin
│   ├── config.py                   <- every constant, seed, palette, hyperparameter
│   ├── dataset.py                  <- download, cache, load, audit, clean, chronological split
│   ├── features.py                 <- market features, train-only quantile binning
│   ├── plots.py                    <- every figure; returns a Figure, never calls plt.show()
│   └── modeling/
│       ├── regimes.py              <- K-means, HMM, DBSCAN, Ward, PCA, anomaly detection
│       ├── environment.py          <- the trading MDP and its reward
│       ├── agents.py               <- tabular Q-learning and SARSA, plus baselines
│       └── evaluate.py             <- PnL, drawdown, hit ratio, turnover, cost drag
├── tests/                          <- 19 tests: reward formula, cost accounting, leakage guards
├── references/data_dictionary.md
├── reports/figures/                <- PNGs written by the notebook
├── pyproject.toml                  <- dependencies and ruff config, single source of truth
└── uv.lock                         <- exact pinned resolution, committed
```

---

## Guarding against look-ahead

The dangerous failure here is not a crash, it is a backtest that looks excellent because the state encoded the future. Three guards, all enforced by tests:

1. **Chronological split.** Train before 2021, test from 2021 onward, never shuffled.
2. **Unsupervised models fitted on train only.** The scaler and cluster centroids are frozen and applied to test unchanged, so a test-day label cannot reflect days the agent has not traded.
3. **Discretization edges from train only.** Quantile bin boundaries come from the training window, so the test period never decides what counts as "high volatility".

`tests/test_leakage.py` asserts that centroids do not move when test data arrives and that bin edges depend only on training data.

---

## How to run

```bash
cd "Module 3/Sprint 4"
uv sync --extra dev          # creates .venv from uv.lock, exact pinned versions
uv run pytest -q             # 19 tests
uv run ruff check .          # lints package, tests, and notebook
uv run jupyter lab notebooks/nvda_regimes_rl.ipynb
```

The notebook runs top to bottom. The first run downloads NVDA from yfinance; later runs read the CSV cache, so no network is needed. Every stochastic step is seeded with `RANDOM_SEED = 42`.

---

## Sprint 4 curriculum coverage

- **Clustering:** K-means with elbow, silhouette, Davies-Bouldin and Calinski-Harabasz selection; DBSCAN; agglomerative with Ward linkage
- **Dimensionality reduction:** PCA with named component loadings, UMAP
- **Mixture / sequence models:** Gaussian HMM for regimes that persist rather than flicker
- **Anomaly detection:** Isolation Forest and Local Outlier Factor, used to bound how far the test window extrapolates
- **Cluster evaluation:** internal metrics only, since market days carry no ground-truth regime
- **RL:** MDP formulation, ε-greedy with decay, tabular Q-learning (off-policy) and SARSA (on-policy)
- **Evaluation:** cumulative PnL against buy-and-hold, random, and flat baselines, plus Sharpe, max drawdown, hit ratio, turnover, and cost drag
