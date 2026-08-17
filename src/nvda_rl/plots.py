"""
All Matplotlib / Seaborn visualization functions.

Design rules:
  - Every function returns a Figure without calling plt.show().
  - apply_global_style() sets project-wide aesthetics; call once at notebook start.
  - No hardcoded colors: palettes come from nvda_rl.config.
  - Axes always carry title, x-label, and y-label.
  - Regime and action colors are consistent everywhere, so green always means
    the constructive state and red always means the stressed one.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from nvda_rl.config import (
    ACTION_COLORS,
    CMAP_SEQ,
    PALETTE_ACCENT,
    PALETTE_LIST,
    PALETTE_PRIMARY,
    REGIME_COLORS,
    TICKER,
    TRADING_DAYS,
)


def apply_global_style() -> None:
    """Apply project-wide styling. Call once at notebook start."""
    sns.set_theme(style="whitegrid", palette=PALETTE_LIST, font_scale=1.05)
    plt.rcParams.update({
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E8E8E8",
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "font.size": 11,
    })


def save_figure(fig: Figure, name: str, figures_dir) -> None:
    """Save a figure as PNG at 150 dpi."""
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(figures_dir) / f"{name}.png", dpi=150, bbox_inches="tight")


def plot_price_and_volume(df: pd.DataFrame) -> Figure:
    """
    Adjusted close on a log axis with volume beneath.

    The log axis is not decoration: NVDA grew by orders of magnitude over this
    window, and on a linear axis the first decade would be an unreadable flat
    line against the last two years.
    """
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(df["date"], df["close"], color=PALETTE_PRIMARY, lw=1.1)
    axes[0].set_yscale("log")
    axes[0].set_title(f"{TICKER} adjusted close (log scale)", fontsize=12)
    axes[0].set_ylabel("Price (USD, log)")

    axes[1].fill_between(df["date"], df["volume"] / 1e6, color=PALETTE_ACCENT, alpha=0.6)
    axes[1].set_title("Daily volume", fontsize=11)
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Shares (millions)")
    fig.tight_layout()
    return fig


def plot_return_distribution(df: pd.DataFrame) -> Figure:
    """Daily return histogram against a matched normal, plus the Q-Q plot."""
    from scipy import stats as sp_stats

    returns = df["return"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    trimmed = returns[returns.abs() < 0.25]
    axes[0].hist(trimmed, bins=100, density=True, color=PALETTE_PRIMARY, alpha=0.7)
    x = np.linspace(trimmed.min(), trimmed.max(), 300)
    axes[0].plot(x, sp_stats.norm.pdf(x, returns.mean(), returns.std()), "k--", lw=1.2,
                 label="Normal fit")
    axes[0].set_title(f"Daily returns (excess kurtosis = {sp_stats.kurtosis(returns):.1f})",
                      fontsize=12)
    axes[0].set_xlabel("Daily return")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    sp_stats.probplot(returns, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q plot against the normal", fontsize=12)
    axes[1].set_xlabel("Theoretical quantiles")
    axes[1].set_ylabel("Observed quantiles")
    fig.tight_layout()
    return fig


def plot_volatility_and_drawdown(df: pd.DataFrame) -> Figure:
    """Rolling annualized volatility above the running drawdown from the peak."""
    equity = df["close"] / df["close"].iloc[0]
    drawdown = (equity / equity.cummax() - 1) * 100

    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
    axes[0].plot(df["date"], df["volatility_21"] * 100, color=PALETTE_ACCENT, lw=1.0)
    axes[0].set_title("Rolling 21-day volatility, annualized", fontsize=12)
    axes[0].set_ylabel("Volatility (% per year)")

    axes[1].fill_between(df["date"], drawdown, 0, color="#C44E52", alpha=0.5)
    axes[1].set_title("Drawdown from the running peak", fontsize=12)
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Drawdown (%)")
    fig.tight_layout()
    return fig


def plot_regime_selection(scores: pd.DataFrame) -> Figure:
    """
    Elbow, silhouette, and Davies-Bouldin side by side.

    Inertia can only ever suggest a bend, so it is shown next to the two
    metrics that actually trade cohesion against separation and can pick a
    winner rather than hint at one.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    panels = [
        ("inertia", "Inertia (elbow)", "lower is better, look for the bend"),
        ("silhouette", "Silhouette", "higher is better"),
        ("davies_bouldin", "Davies-Bouldin", "lower is better"),
    ]
    for ax, (col, title, note) in zip(axes, panels, strict=True):
        ax.plot(scores.index, scores[col], marker="o", color=PALETTE_PRIMARY)
        best = scores[col].idxmax() if col == "silhouette" else scores[col].idxmin()
        ax.axvline(best, color=PALETTE_ACCENT, ls="--", lw=1.2, label=f"best k = {best}")
        ax.set_title(f"{title}\n{note}", fontsize=11)
        ax.set_xlabel("Number of regimes (k)")
        ax.set_ylabel(title)
        ax.legend()
    fig.tight_layout()
    return fig


def plot_regime_timeline(df: pd.DataFrame, label_col: str, title: str) -> Figure:
    """
    The price series colored by regime.

    This is the plot that makes a regime model falsifiable: if the colors do
    not line up with the visible calm stretches and drawdowns, the clustering
    has found something other than what it claims to have found.
    """
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(df["date"], df["close"], color="#BBBBBB", lw=0.8, zorder=1)
    for regime in sorted(df[label_col].unique()):
        mask = df[label_col] == regime
        ax.scatter(df.loc[mask, "date"], df.loc[mask, "close"], s=5,
                   color=REGIME_COLORS.get(regime, "#333333"),
                   label=f"Regime {regime}", zorder=2)
    ax.set_yscale("log")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD, log)")
    ax.legend(markerscale=3, ncol=4)
    fig.tight_layout()
    return fig


def plot_regime_projection(embedding: np.ndarray, labels: np.ndarray,
                           method: str) -> Figure:
    """Regimes drawn in a two-dimensional projection of the feature space."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for regime in sorted(set(labels)):
        mask = labels == regime
        name = "Noise" if regime == -1 else f"Regime {regime}"
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=8, alpha=0.7,
                   color=REGIME_COLORS.get(regime, "#333333"), label=name)
    ax.set_title(f"Market states in {method} space", fontsize=12)
    ax.set_xlabel(f"{method} component 1")
    ax.set_ylabel(f"{method} component 2")
    ax.legend(markerscale=2)
    fig.tight_layout()
    return fig


def plot_transition_matrix(matrix: pd.DataFrame, title: str) -> Figure:
    """
    Regime-to-regime transition probabilities.

    The diagonal carries the argument: high values mean regimes persist, which
    is what makes yesterday's label informative about today and therefore worth
    putting in the agent's state at all.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap=CMAP_SEQ, vmin=0, vmax=1,
                linewidths=0.6, linecolor="white", ax=ax,
                cbar_kws={"label": "P(next regime | current regime)"})
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Regime tomorrow")
    ax.set_ylabel("Regime today")
    fig.tight_layout()
    return fig


def plot_anomalies(df: pd.DataFrame, flags: np.ndarray) -> Figure:
    """Price series with the days an unsupervised detector called anomalous."""
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df["date"], df["close"], color=PALETTE_PRIMARY, lw=0.9, label="Close")
    mask = flags == -1
    ax.scatter(df.loc[mask, "date"], df.loc[mask, "close"], s=22, color="#C44E52",
               zorder=3, label=f"Anomalous days ({int(mask.sum())})")
    ax.set_yscale("log")
    ax.set_title("Days flagged as anomalous by Isolation Forest", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD, log)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_learning_curves(curves: dict[str, pd.DataFrame], window: int = 20) -> Figure:
    """
    Total episode reward per training episode, smoothed.

    The raw series is noisy because the market tape is fixed while epsilon keeps
    injecting randomness, so the rolling mean is what shows whether the agent is
    actually improving rather than sampling a lucky path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for (name, curve), color in zip(curves.items(), PALETTE_LIST, strict=False):
        axes[0].plot(curve["episode"],
                     curve["total_reward"].rolling(window, min_periods=1).mean(),
                     label=name, color=color, lw=1.4)
        axes[1].plot(curve["episode"], curve["states_visited"], label=name,
                     color=color, lw=1.4)
    axes[0].set_title(f"Episode reward ({window}-episode rolling mean)", fontsize=12)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total net reward")
    axes[0].legend()

    axes[1].set_title("Distinct states visited", fontsize=12)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("States in the Q-table")
    axes[1].legend()
    fig.tight_layout()
    return fig


def plot_equity_curves(curves: pd.DataFrame, title: str) -> Figure:
    """
    Cumulative growth of one unit of capital under each policy.

    Drawn on a log axis so a policy that doubles and one that ten-times are
    both legible, and so equal vertical distances mean equal percentage moves.
    """
    fig, ax = plt.subplots(figsize=(13, 6))
    for column, color in zip(curves.columns, PALETTE_LIST, strict=False):
        ax.plot(curves.index, curves[column], label=column, lw=1.4, color=color)
    ax.axhline(1.0, color="#888888", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of 1 unit (log scale)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_action_timeline(ledger: pd.DataFrame, price: pd.Series, title: str) -> Figure:
    """The position the agent held over time, drawn against the price."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(ledger["date"], price.to_numpy()[: len(ledger)],
                 color=PALETTE_PRIMARY, lw=1.0)
    axes[0].set_yscale("log")
    axes[0].set_title(title, fontsize=12)
    axes[0].set_ylabel("Price (USD, log)")

    for action in (-1, 0, 1):
        mask = ledger["action"] == action
        axes[1].scatter(ledger.loc[mask, "date"], ledger.loc[mask, "action"], s=4,
                        color=ACTION_COLORS[action],
                        label={1: "long", 0: "flat", -1: "short"}[action])
    axes[1].set_yticks([-1, 0, 1])
    axes[1].set_yticklabels(["short", "flat", "long"])
    axes[1].set_title("Position held", fontsize=11)
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Action")
    axes[1].legend(ncol=3, markerscale=3)
    fig.tight_layout()
    return fig


def plot_policy_by_regime(regime_map: pd.DataFrame) -> Figure:
    """
    The action mix inside each regime, as stacked shares.

    This chart is the direct test of whether the unsupervised layer earned its
    place: if the bars are identical across regimes, the regime feature is not
    changing any decision and the state augmentation is decoration.
    """
    shares = regime_map[["short_%", "flat_%", "long_%"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(shares))
    for column, action in zip(shares.columns, (-1, 0, 1), strict=True):
        ax.bar(shares.index.astype(str), shares[column], bottom=bottom,
               color=ACTION_COLORS[action], label=column.replace("_%", ""))
        bottom += shares[column].to_numpy()
    ax.set_title("Position mix chosen inside each regime", fontsize=12)
    ax.set_xlabel("Regime")
    ax.set_ylabel("Share of days (%)")
    ax.set_ylim(0, 100)
    ax.legend(ncol=3)
    fig.tight_layout()
    return fig


def plot_metric_comparison(summary: pd.DataFrame, metric: str, note: str) -> Figure:
    """
    One metric across policies, with the best bar highlighted.

    Bars start at zero so their lengths stay proportional to the values, which
    matters most for the drawdown panel where a truncated axis would make a
    catastrophic loss look survivable.
    """
    values = summary[metric].sort_values()
    best = values.idxmax() if "drawdown" not in metric else values.idxmin()
    colors = [PALETTE_ACCENT if name == best else PALETTE_PRIMARY for name in values.index]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(values.index.astype(str), values.to_numpy(), color=colors)
    ax.axvline(0, color="#666666", lw=1)
    ax.set_title(f"{metric} by policy\n{note}", fontsize=12)
    ax.set_xlabel(metric)
    ax.set_ylabel("Policy")
    fig.tight_layout()
    return fig


def plot_rolling_sharpe(ledgers: dict[str, pd.DataFrame], window: int = TRADING_DAYS) -> Figure:
    """
    Rolling one-year Sharpe per policy.

    A single headline Sharpe hides when the edge existed. This shows whether a
    policy was consistently decent or simply carried by one extraordinary
    stretch, which is the difference between a strategy and a story.
    """
    fig, ax = plt.subplots(figsize=(13, 5))
    for (name, ledger), color in zip(ledgers.items(), PALETTE_LIST, strict=False):
        net = ledger.set_index("date")["net_return"]
        # Same definition as the headline figure, via evaluate.sharpe_ratio.
        rolling = (net.rolling(window).mean() / net.rolling(window).std()) * np.sqrt(TRADING_DAYS)
        ax.plot(rolling.index, rolling, label=name, lw=1.3, color=color)
    ax.axhline(0, color="#666666", ls=":", lw=1)
    ax.set_title(f"Rolling {window}-day Sharpe ratio", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Sharpe")
    ax.legend()
    fig.tight_layout()
    return fig
