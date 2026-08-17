"""
Global constants for the NVDA regimes and reinforcement learning project.

Every magic number and string lives here so the notebook carries no hardcoded
literals and a change propagates to every cell at once.
"""

RANDOM_SEED = 42

TICKER = "NVDA"
START_DATE = "2010-01-01"

# Chronological split. Shuffling would leak the future into training, and a
# regime label fitted on data the agent has not lived through yet is the same
# leak wearing a different hat, so the unsupervised models are fitted on the
# training window only.
# Three chronological windows. Hyperparameters (learning rate, episode count,
# bin count, number of regimes) are chosen on validation and the test window is
# opened once, at the end. Picking them on test would make the final number a
# training score wearing a disguise.
VALIDATION_DATE = "2019-01-01"   # train < this
SPLIT_DATE = "2021-01-01"        # validation is [VALIDATION_DATE, SPLIT_DATE), test >= this

# Feature windows, in trading days.
VOL_WINDOW = 21             # one trading month of realized volatility
MOMENTUM_WINDOW = 21
RETURN_LAGS = [1, 2, 3, 5]
HIGH_LOOKBACK = 252         # one trading year, for distance from the rolling high
RSI_WINDOW = 14

# Columns handed to the unsupervised regime models. Deliberately small and
# economically meaningful, because a regime label is only useful if it can be
# explained to a desk in one sentence.
REGIME_FEATURES = [
    "return",
    "volatility_21",
    "momentum_21",
    "dist_252d_high",
    "volume_zscore",
]

# The agent's market state is built from the SAME columns the regime model
# clusters on. An earlier version binned only three of them while the regime was
# derived from all five, so the regime label smuggled in two extra features and
# the ablation flattered it. Same inputs both sides; the only difference is
# whether the cluster label is appended.
STATE_FEATURES = REGIME_FEATURES

N_REGIMES = 3               # bullish / bearish / sideways, per the brief
STATE_BINS = 2              # per feature; chosen on validation, see the notebook
DBSCAN_EPS = 0.9
DBSCAN_MIN_SAMPLES = 20
PCA_COMPONENTS = 2
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
ANOMALY_CONTAMINATION = 0.02
REGIME_STABILITY_SEEDS = 20      # reseeded K-means runs for the agreement check
FORWARD_HORIZONS = (1, 5)        # trading days ahead used to validate regimes

# Trading environment.
ACTIONS = (-1, 0, 1)        # short, flat, long
TRANSACTION_COST = 0.001    # 10 bps of notional per unit of position change
TRADING_DAYS = 252
# Cost levels for the sensitivity sweep: frictionless, the headline 10 bps, and
# two levels that stand in for slippage and harder-to-borrow shorts.
COST_LEVELS = (0.0, 0.0005, 0.001, 0.002, 0.005)

# Tabular RL. The state space is discrete, so these are the classic settings:
# a decaying epsilon explores early and exploits late, and gamma below one
# keeps the value of a daily reward stream finite.
# Daily rewards are roughly 0.2% with a 3% standard deviation, so the target of
# every update is mostly noise. A large learning rate makes the table chase that
# noise instead of averaging it: at alpha = 0.1 the learned values sat around
# 0.001 with no stable ordering, and the greedy policy collapsed to permanently
# flat. A small rate is the fix, and gamma near one suits a daily horizon where
# a position is meant to be held for weeks rather than hours.
ALPHA = 0.005               # learning rate
GAMMA = 0.99                # discount factor
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_EPISODES = 300
N_EPISODES = 600

# Plotting.
PALETTE_PRIMARY = "#4C72B0"
PALETTE_ACCENT = "#DD8452"
PALETTE_LIST = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
                "#CCB974", "#64B5CD", "#8C8C8C", "#937860", "#DA8BC3"]
# Regime colors carry meaning: green for the calm advance, red for the stressed
# decline, grey for the directionless middle.
REGIME_COLORS = {0: "#55A868", 1: "#C44E52", 2: "#8C8C8C", -1: "#000000"}
ACTION_COLORS = {1: "#55A868", 0: "#8C8C8C", -1: "#C44E52"}
CMAP_DIV = "RdBu_r"
CMAP_SEQ = "Blues"
