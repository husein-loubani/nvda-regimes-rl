# NVDA market regimes and a reinforcement learning trader

**Module 3 · Sprint 4: Unsupervised and reinforcement learning** | Turing College · Data Science Program

Clustering finds market regimes in NVIDIA's daily price history. A tabular RL agent then trades the stock with those regimes in its state. The question the project answers is whether the unsupervised layer actually makes the agent decide better.

---

## Headline result

Three things are true at once, and the project is more useful for holding all three.

**The unsupervised half worked.** Three regimes, stable under reseeding (ARI 1.00) and bootstrap resampling (ARI 0.92), persistent over weeks, and predictive of forward volatility and drawdown on quantities the clustering never saw.

**The RL half did not beat the benchmark.**

| Policy | Test PnL | Ann. return | Sharpe | Max drawdown | Turnover / yr |
|---|---|---|---|---|---|
| Buy and hold | +1414% | 63.3% | 1.22 | -66.3% | 0.2 |
| Q-learning [market only] | +904% | 51.7% | 1.09 | -64.1% | 123.3 |
| SARSA [market only] | +624% | 42.9% | 0.99 | -60.0% | 139.5 |
| Q-learning [market + regime] | +348% | 31.1% | 0.79 | -79.8% | 84.7 |
| Random | +96% | 12.9% | 0.49 | -49.2% | 221.3 |
| SARSA [market + regime] | +26% | 4.3% | 0.32 | -68.8% | 79.6 |

**And the regime label made the agent worse**, despite being informative. That is the finding I would defend: an informative feature is not automatically a useful state variable. Appending the label triples the reachable state space from 96 to 288 while the training window stays the same length, and the estimation error costs more than the information repays.

Sealed test window: 2021-01-04 to 2026-07-28 (1,397 trading days).

### Two corrections that reversed an earlier conclusion

An earlier version reported that the regime state nearly doubled PnL and tied buy-and-hold. Both claims were artefacts.

1. **A reward timing bug.** `next_return` already shifts the return forward a day, and the environment then multiplied it by the *previous* position, shifting it again. A position decided on day *t* earned the return of day *t+2*, so the agent traded on a one-day lag throughout. Every existing test shared that convention, so they all passed while the timeline was wrong. `tests/modeling/test_environment.py` now pins it with named weekdays and a hand-computed reward.
2. **An unfair ablation.** The baseline saw three binned features while the regime was built from five, so the label smuggled in two extra inputs. Both arms now use identical features.

Fix both and the advantage disappears.

### What survives contact with reality

The walk-forward is not flattering. The best agent is strongly positive in 2021, 2023, 2024 and 2025, then loses **55.9%** in 2022 with a Sharpe of -1.08, and its turnover more than doubles that year. Four good years carrying one very bad one is not a repeatable edge.

The cost curve is worse news. At 0 bps the strategy returns +1888%, at the headline 10 bps +904%, at 20 bps +407%, and by 50 bps it is **negative**. Since 50 bps is not unreasonable for a daily-rebalanced strategy that shorts, the edge lives inside the assumption that execution is cheap. Buy-and-hold, which trades twice, barely notices the same axis.

---

## What the project does

- **Unsupervised:** cluster daily returns and volatility into regimes I can name, then check they persist and hold up out of sample.
- **Reinforcement learning:** learn a long / flat / short policy under transaction costs, with the regime label in the state.

The reward, exactly as the brief specifies:

$$r_t = a_{t-1} \cdot \text{return}_t - \text{cost} \cdot |a_t - a_{t-1}|$$

The position chosen at the close of day *t* earns day *t+1*'s return, and the cost scales with how far the position moves. The environment implements this from the decision's point of view (`a_t * next_return_t`), which is the same sum but credits the action that actually caused the profit. Getting that indexing wrong is what produced the earlier incorrect results, so it is pinned by a test with named weekdays and a hand-computed number.

---

## The regimes

Fitted on the training window and frozen. Profiling them by *current* return is partly circular, since current return is one of the clustering inputs and the labels are then sorted by it, so the honest test is what happens **after** the label is observed:

| Regime | Days | Forward 1d | Forward 5d | Forward 1d hit | Forward 21d vol | Forward 21d drawdown |
|---|---|---|---|---|---|---|
| 0 Melt-up | 174 | +59.6 bps | +203.5 bps | 0.59 | 40.9% | -4.8% |
| 1 Calm advance | 1372 | +13.7 bps | +80.1 bps | 0.52 | 31.9% | -4.4% |
| 2 Drawdown | 466 | -1.4 bps | -18.2 bps | 0.48 | 45.2% | -8.5% |

None of those columns was shown to the clustering. The regimes carry real forward information, and what they predict best is **risk**: forward drawdown roughly doubles between the calm and the stressed state, while forward direction separates much more weakly.

They also persist, with diagonal transition probabilities of 0.69, 0.93 and 0.89, so yesterday's label says something about today.

---

## Repository structure

```
.
├── data/raw/NVDA.csv               <- yfinance cache, keeps reruns offline
├── notebooks/
│   └── nvda_regimes_rl.ipynb       <- the deliverable
├── src/nvda_rl/                    <- all logic; the notebook stays thin
│   ├── config.py                   <- constants, seeds, palette, hyperparameters
│   ├── dataset.py                  <- download, cache, load, audit, clean, three-way split
│   ├── features.py                 <- market features, KBinsDiscretizer binning
│   ├── plots.py                    <- every figure, returns a Figure
│   └── modeling/
│       ├── regimes.py              <- K-means, HMM, DBSCAN, Ward, PCA, forward validation, stability
│       ├── environment.py          <- the trading MDP, on the Gymnasium Env contract
│       ├── agents.py               <- Q-learning, SARSA, and baselines
│       └── evaluate.py             <- PnL, Sharpe, drawdown, turnover, cost sweep, walk-forward
├── tests/                          <- 83 tests, mirroring the package
│   ├── test_dataset.py  test_features.py  test_plots.py
│   ├── test_leakage.py  test_pipeline.py
│   └── modeling/  test_agents.py  test_environment.py  test_evaluate.py  test_regimes.py
├── references/data_dictionary.md
├── reports/figures/
├── pyproject.toml
└── uv.lock                         <- exact pinned versions, committed
```

---

## Guarding against look-ahead

The dangerous failure here is not a crash. It is a backtest that looks great because the state encoded the future. Four guards, all covered by tests:

1. The split is chronological and three-way. Train, then validation for every tuning decision, then a test window opened once.
2. The scaler and cluster centroids are fitted on train only, then frozen, so a test-day label cannot reflect days the agent never traded.
3. The discretization bin edges come from training quantiles, so the test period never decides what counts as high volatility.
4. Returns are derived *after* cleaning, so a return can never be measured against a bar that cleaning later removed.

`tests/test_leakage.py` proves this rather than asserting it. An earlier version fitted the binner twice on the same frame, which only showed the code was deterministic. The current test runs the whole fitting workflow against two very different test sets and requires the bin edges, scaler parameters, centroids, and label ordering to come out identical.

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
uv run pytest -q             # 83 tests
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
