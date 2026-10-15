"""Metrics with honest uncertainty: day-block bootstrap intervals, calibration, net
benefit, and class-conditional conformal bands."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from accidents import config


def wilson_interval(
    successes: np.ndarray | float, n: np.ndarray | float, z: float = 1.96
) -> tuple[np.ndarray, np.ndarray]:
    """95% interval for a rate; unlike the normal approximation it stays inside [0, 1]
    and behaves at rates near zero, which is where Severity 4 lives."""
    successes = np.asarray(successes, dtype=float)
    n = np.asarray(n, dtype=float)
    rate = successes / n
    centre = (rate + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    # Clip float rounding at the edges (a zero count gives -1e-17, not 0).
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def rate_table(y: pd.Series, by: pd.Series | list[pd.Series], min_count: int = 1) -> pd.DataFrame:
    """Count, Severity-4 rate and its Wilson interval for each group."""
    grouped = y.groupby(by, observed=True).agg(n="size", positives="sum")
    grouped = grouped[grouped["n"] >= min_count]
    grouped["rate"] = grouped["positives"] / grouped["n"]
    grouped["low"], grouped["high"] = wilson_interval(grouped["positives"], grouped["n"])
    return grouped


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """Mean gap between predicted and observed rate over equal-count bins.

    Equal-count bins, because with ~3% positives almost every prediction sits near zero
    and equal-width bins would leave most bins empty.
    """
    order = np.argsort(p, kind="stable")
    gaps = []
    weights = []
    for chunk in np.array_split(order, bins):
        if len(chunk) == 0:
            continue
        gaps.append(abs(p[chunk].mean() - y[chunk].mean()))
        weights.append(len(chunk))
    return float(np.average(gaps, weights=weights))


def scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Ranking (AP, ROC-AUC) and probability quality (log loss, Brier, ECE)."""
    return {
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
    }


def rows_by_group(groups: pd.Series | np.ndarray) -> list[np.ndarray]:
    codes, _ = pd.factorize(np.asarray(groups))
    order = np.argsort(codes, kind="stable")
    boundaries = np.flatnonzero(np.diff(codes[order])) + 1
    return np.split(order, boundaries)


def bootstrap(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    groups: pd.Series | np.ndarray,
    metric=average_precision_score,
    n_boot: int = 1000,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Metric for each model on `n_boot` resamples of whole groups (days).

    Accidents on the same day share weather and traffic, so days, not rows, are the
    independent units. Every model is scored on the same resamples, so differences
    between columns are paired.
    """
    blocks = rows_by_group(groups)
    rng = np.random.default_rng(seed)
    results = {name: np.empty(n_boot) for name in predictions}
    for b in range(n_boot):
        picked = rng.integers(0, len(blocks), len(blocks))
        idx = np.concatenate([blocks[i] for i in picked])
        for name, p in predictions.items():
            results[name][b] = metric(y[idx], p[idx])
    return pd.DataFrame(results)


def interval(samples: np.ndarray | pd.Series, level: float = 0.95) -> tuple[float, float]:
    tail = (1 - level) / 2
    low, high = np.quantile(samples, [tail, 1 - tail])
    return float(low), float(high)


def net_benefit(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    """Decision-curve analysis: benefit of acting on the model at each threshold,
    against acting on every case or on none."""
    n = len(y)
    prevalence = y.mean()
    rows = []
    for t in thresholds:
        act = p >= t
        tp = np.sum(act & (y == 1)) / n
        fp = np.sum(act & (y == 0)) / n
        odds = t / (1 - t)
        rows.append(
            {
                "threshold": t,
                "model": tp - fp * odds,
                "act_on_all": prevalence - (1 - prevalence) * odds,
                "act_on_none": 0.0,
            }
        )
    return pd.DataFrame(rows)


def conformal_thresholds(y_cal: np.ndarray, p_cal: np.ndarray, alpha: float = 0.1) -> dict:
    """Class-conditional split conformal: each class keeps coverage 1 - alpha.

    Nonconformity of class c is 1 - P(c). For the positive class that is 1 - p, for the
    negative class it is p, which turns the two quantiles into two thresholds on p.
    """

    def quantile(values: np.ndarray) -> float:
        n = len(values)
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        return float(np.quantile(values, level, method="higher"))

    return {
        "alpha": alpha,
        "minor_max": quantile(p_cal[y_cal == 0]),  # "minor" is in the set if p <= this
        "major_min": 1 - quantile(1 - p_cal[y_cal == 1]),  # "major" is in the set if p >= this
    }


def conformal_band(p: np.ndarray, thresholds: dict) -> np.ndarray:
    """The prediction set as a word: minor, major, uncertain (both) or none (neither)."""
    minor = p <= thresholds["minor_max"]
    major = p >= thresholds["major_min"]
    return np.select([minor & major, minor, major], ["uncertain", "minor", "major"], default="none")


def population_stability(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population stability index of `actual` against `expected` (e.g. test vs train).

    Numeric features use the expected data's equal-count bins; categories are their own
    bins; missing values always get a bin of their own, because missingness itself drifts.
    Rule of thumb: below 0.1 stable, 0.1-0.25 moderate shift, above 0.25 large shift.
    """
    if isinstance(expected.dtype, pd.CategoricalDtype) or expected.dtype == object:
        e_bins = expected.astype(str)
        a_bins = actual.astype(str)
    else:
        edges = np.unique(np.nanquantile(expected, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e_bins = pd.cut(expected, edges.tolist()).astype(str)
        a_bins = pd.cut(actual, edges.tolist()).astype(str)
    e_share = e_bins.value_counts(normalize=True)
    a_share = a_bins.value_counts(normalize=True)
    shares = pd.concat([e_share, a_share], axis=1, keys=["e", "a"]).fillna(0).clip(lower=1e-4)
    return float(((shares["a"] - shares["e"]) * np.log(shares["a"] / shares["e"])).sum())
