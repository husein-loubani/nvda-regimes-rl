"""
Unsupervised market-regime discovery.

Every estimator here is fitted on the training window and then applied to the
test window unchanged. Fitting a scaler or a cluster model on the full history
would let the agent's state encode information from days it has not traded yet,
which is the subtlest and most damaging leak available in this project: the
resulting backtest would look excellent and mean nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nvda_rl.config import (
    ANOMALY_CONTAMINATION,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    FORWARD_HORIZONS,
    N_REGIMES,
    PCA_COMPONENTS,
    RANDOM_SEED,
    REGIME_STABILITY_SEEDS,
)


def select_n_regimes(x_train: pd.DataFrame, k_range: range = range(2, 9)) -> pd.DataFrame:
    """
    Elbow and silhouette diagnostics over candidate cluster counts.

    Inertia always falls as k rises, so it can only ever suggest a bend, not a
    winner; silhouette and Davies-Bouldin actually trade cohesion against
    separation and are the numbers to argue from.
    """
    scaler = StandardScaler().fit(x_train)
    x_scaled = scaler.transform(x_train)

    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10).fit(x_scaled)
        labels = km.labels_
        rows.append({
            "k": k,
            "inertia": round(km.inertia_, 1),
            "silhouette": round(silhouette_score(x_scaled, labels), 4),
            "davies_bouldin": round(davies_bouldin_score(x_scaled, labels), 4),
            "calinski_harabasz": round(calinski_harabasz_score(x_scaled, labels), 1),
        })
    return pd.DataFrame(rows).set_index("k")


def _order_by_mean_return(labels: np.ndarray, returns: np.ndarray) -> dict[int, int]:
    """
    Relabel clusters so 0 is the highest-mean-return regime and the last is the
    lowest. K-means numbers its clusters arbitrarily, so without this the same
    regime changes identity between runs and every interpretation written about
    "regime 1" silently rots.
    """
    order = (
        pd.DataFrame({"label": labels, "ret": returns})
        .groupby("label")["ret"].mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    return {old: new for new, old in enumerate(order)}


def fit_kmeans_regimes(
    x_train: pd.DataFrame, train_returns: np.ndarray, n_regimes: int = N_REGIMES
) -> dict:
    """
    Fit the scaler and K-means on the training window only, then freeze both.

    Returns a dict carrying the fitted pipeline and the label remapping, which
    together are everything needed to assign a regime to an unseen day.
    """
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("kmeans", KMeans(n_clusters=n_regimes, random_state=RANDOM_SEED, n_init=10)),
    ]).fit(x_train)

    raw_labels = pipe.named_steps["kmeans"].labels_
    mapping = _order_by_mean_return(raw_labels, train_returns)
    return {"pipeline": pipe, "mapping": mapping, "n_regimes": n_regimes}


def assign_regimes(model: dict, x: pd.DataFrame) -> np.ndarray:
    """
    Apply a frozen regime model to any window, training or test.

    This is the only path by which a test day receives a label, so the test
    window can never influence where the cluster centroids sit.
    """
    raw = model["pipeline"].predict(x)
    return np.array([model["mapping"][label] for label in raw])


def fit_hmm_regimes(
    x_train: pd.DataFrame, train_returns: np.ndarray, n_regimes: int = N_REGIMES
) -> dict:
    """
    Gaussian hidden Markov model over the same features.

    K-means treats every day as independent, which is plainly wrong for markets:
    a volatile day is overwhelmingly likely to be followed by another one. An
    HMM models that persistence explicitly through its transition matrix, so its
    regimes tend to form runs rather than flickering day to day, and a run is
    what a trading policy can actually act on.
    """
    from hmmlearn.hmm import GaussianHMM

    scaler = StandardScaler().fit(x_train)
    x_scaled = scaler.transform(x_train)

    hmm = GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=200,
        random_state=RANDOM_SEED,
    ).fit(x_scaled)

    mapping = _order_by_mean_return(hmm.predict(x_scaled), train_returns)
    return {"scaler": scaler, "hmm": hmm, "mapping": mapping, "n_regimes": n_regimes,
            "converged": bool(hmm.monitor_.converged)}


def assign_hmm_regimes(model: dict, x: pd.DataFrame) -> np.ndarray:
    """Decode the most likely state sequence for a window with a frozen HMM."""
    raw = model["hmm"].predict(model["scaler"].transform(x))
    return np.array([model["mapping"][label] for label in raw])


def fit_dbscan(x_train: pd.DataFrame, eps: float = DBSCAN_EPS,
               min_samples: int = DBSCAN_MIN_SAMPLES) -> tuple[np.ndarray, dict]:
    """
    DBSCAN on the scaled training features, which labels low-density days -1.

    DBSCAN cannot label unseen data, so it is used here as a diagnostic rather
    than as a state feature: it answers whether the regime structure is density
    based or merely partitional, and how many days sit in no cluster at all.
    """
    x_scaled = StandardScaler().fit_transform(x_train)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(x_scaled)
    info = {
        "n_clusters": int(len(set(labels) - {-1})),
        "n_noise": int((labels == -1).sum()),
        "noise_share_%": round((labels == -1).mean() * 100, 2),
    }
    return labels, info


def fit_agglomerative(x_train: pd.DataFrame, n_regimes: int = N_REGIMES) -> np.ndarray:
    """Ward-linkage agglomerative clustering, as a shape check on K-means."""
    x_scaled = StandardScaler().fit_transform(x_train)
    return AgglomerativeClustering(n_clusters=n_regimes, metric="euclidean",
                                   linkage="ward").fit_predict(x_scaled)


def cluster_quality(x: pd.DataFrame, labels: np.ndarray) -> dict:
    """
    Internal validation only, since market days carry no ground-truth regime.

    Noise points are excluded before scoring, because a metric that treats
    DBSCAN's -1 bucket as a cluster is measuring an artifact.
    """
    x_scaled = StandardScaler().fit_transform(x)
    mask = labels != -1
    if len(set(labels[mask])) < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "n_clusters": 0}
    return {
        "silhouette": round(silhouette_score(x_scaled[mask], labels[mask]), 4),
        "davies_bouldin": round(davies_bouldin_score(x_scaled[mask], labels[mask]), 4),
        "n_clusters": int(len(set(labels[mask]))),
    }


def regime_profile(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Describe what each regime actually is, in the units a trader thinks in.

    A cluster number means nothing on its own; this table is what turns it into
    "the calm advance" or "the stressed drawdown", and it is the evidence for
    every interpretation claimed about the regimes.
    """
    out = (
        df.groupby(label_col)
        .agg(
            days=("return", "size"),
            mean_return_bps=("return", lambda r: round(r.mean() * 1e4, 1)),
            ann_return_pct=("return", lambda r: round(r.mean() * 252 * 100, 1)),
            ann_vol_pct=("return", lambda r: round(r.std() * np.sqrt(252) * 100, 1)),
            hit_ratio=("return", lambda r: round((r > 0).mean(), 3)),
            mean_momentum=("momentum_21", "mean"),
            mean_dist_high=("dist_252d_high", "mean"),
        )
    )
    out["share_%"] = (out["days"] / out["days"].sum() * 100).round(1)
    out["mean_momentum"] = out["mean_momentum"].round(4)
    out["mean_dist_high"] = out["mean_dist_high"].round(4)
    return out


def transition_matrix(labels: np.ndarray) -> pd.DataFrame:
    """
    Day-to-day regime transition probabilities.

    The diagonal is the interesting part: a high value means regimes persist,
    which is what makes them worth conditioning a policy on. If the matrix were
    close to uniform, yesterday's regime would say nothing about today's and the
    whole state augmentation would be noise.
    """
    states = sorted(set(labels))
    counts = pd.DataFrame(0, index=states, columns=states, dtype=float)
    for a, b in zip(labels[:-1], labels[1:], strict=True):
        counts.loc[a, b] += 1
    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    probs.index.name = "from_regime"
    probs.columns.name = "to_regime"
    return probs.round(3)


def fit_pca(x_train: pd.DataFrame, n_components: int = PCA_COMPONENTS) -> Pipeline:
    """Scale-then-PCA, fitted on training data, for latent state features."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=RANDOM_SEED)),
    ]).fit(x_train)


def pca_loadings(pipe: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """
    Component loadings, so the latent features can be named rather than trusted.

    An unnamed principal component in an RL state is an unfalsifiable claim;
    reading the loadings is what lets the notebook say which market property
    each component actually encodes.
    """
    pca = pipe.named_steps["pca"]
    out = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"PC{i + 1}" for i in range(pca.n_components_)],
    ).round(3)
    out.loc["explained_variance_%"] = (pca.explained_variance_ratio_ * 100).round(1)
    return out


def detect_anomalies(x_train: pd.DataFrame, x_all: pd.DataFrame,
                     contamination: float = ANOMALY_CONTAMINATION) -> dict:
    """
    Isolation Forest (fitted on train, applied everywhere) and Local Outlier
    Factor (fitted with novelty detection so it can score unseen days too).

    Anomalies matter here for a specific reason: an RL agent that meets a market
    state unlike anything in its training window is extrapolating, and knowing
    how often that happens in the test period bounds how much to trust the
    backtest.
    """
    scaler = StandardScaler().fit(x_train)
    iso = IsolationForest(
        contamination=contamination, random_state=RANDOM_SEED, n_estimators=200
    ).fit(scaler.transform(x_train))
    lof = LocalOutlierFactor(
        n_neighbors=20, contamination=contamination, novelty=True
    ).fit(scaler.transform(x_train))

    x_scaled = scaler.transform(x_all)
    return {
        "isolation_forest": iso.predict(x_scaled),
        "local_outlier_factor": lof.predict(x_scaled),
        "iso_score": iso.score_samples(x_scaled),
    }


def forward_regime_profile(
    df: pd.DataFrame, label_col: str, horizons: tuple = FORWARD_HORIZONS
) -> pd.DataFrame:
    """
    Describe each regime by what happens *after* it is observed.

    The ordinary profile is partly circular: current return is one of the
    clustering inputs and the labels are then sorted by mean return, so finding
    a high-return and a low-return regime afterwards is close to guaranteed.
    This table instead reports forward returns, forward volatility and forward
    drawdown measured over the days following the label. Those quantities were
    never shown to the clustering, so they are the honest test of whether a
    regime carries information a trading decision could use.
    """
    out = df[[label_col]].copy()
    for h in horizons:
        fwd = df["close"].shift(-h) / df["close"] - 1
        out[f"fwd_{h}d_return_bps"] = fwd * 1e4
        out[f"fwd_{h}d_hit_ratio"] = (fwd > 0).astype(float)
    out["fwd_21d_vol_%"] = (
        df["return"].shift(-1).rolling(21).std().shift(-20) * np.sqrt(252) * 100
    )
    forward_min = df["close"].shift(-1).rolling(21).min().shift(-20)
    out["fwd_21d_drawdown_%"] = (forward_min / df["close"] - 1) * 100

    agg = out.groupby(label_col).mean().round(2)
    agg.insert(0, "days", out.groupby(label_col).size())
    return agg


def regime_stability(
    x_train: pd.DataFrame,
    train_returns: np.ndarray,
    n_regimes: int = N_REGIMES,
    n_seeds: int = REGIME_STABILITY_SEEDS,
) -> pd.DataFrame:
    """
    Re-fit K-means under many seeds and measure how often the same days end up
    together, using the adjusted Rand index against the reference labelling.

    A silhouette score says the clusters are geometrically tidy; it says nothing
    about whether they would survive a slightly different sample. If the labels
    move around under reseeding, a regime is a poor state variable no matter how
    clean it looks, because the agent would be conditioning on an accident of
    initialisation.
    """
    from sklearn.metrics import adjusted_rand_score

    reference = assign_regimes(fit_kmeans_regimes(x_train, train_returns, n_regimes), x_train)
    scores = []
    for seed in range(1, n_seeds + 1):
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("kmeans", KMeans(n_clusters=n_regimes, random_state=seed, n_init=10)),
        ]).fit(x_train)
        labels = pipe.named_steps["kmeans"].labels_
        mapping = _order_by_mean_return(labels, train_returns)
        relabelled = np.array([mapping[x] for x in labels])
        scores.append(adjusted_rand_score(reference, relabelled))

    scores = np.array(scores)
    return pd.DataFrame([{
        "seeds": n_seeds,
        "mean_ARI": round(float(scores.mean()), 4),
        "min_ARI": round(float(scores.min()), 4),
        "std_ARI": round(float(scores.std()), 4),
        "share_above_0.9": round(float((scores > 0.9).mean()), 3),
    }])


def bootstrap_stability(
    x_train: pd.DataFrame,
    train_returns: np.ndarray,
    n_regimes: int = N_REGIMES,
    n_boot: int = 20,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    The harder version of the same question: refit on bootstrap resamples and
    score agreement on the overlapping days. Reseeding only perturbs the
    initialisation, while resampling perturbs the data itself, which is closer
    to what happens when new months of history arrive.
    """
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(seed)
    reference_model = fit_kmeans_regimes(x_train, train_returns, n_regimes)
    reference = assign_regimes(reference_model, x_train)

    scores = []
    n = len(x_train)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        sample = x_train.iloc[idx]
        model = fit_kmeans_regimes(sample, train_returns[idx], n_regimes)
        scores.append(adjusted_rand_score(reference, assign_regimes(model, x_train)))

    scores = np.array(scores)
    return pd.DataFrame([{
        "bootstraps": n_boot,
        "mean_ARI": round(float(scores.mean()), 4),
        "min_ARI": round(float(scores.min()), 4),
        "std_ARI": round(float(scores.std()), 4),
    }])
