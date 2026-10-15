"""Explanation helpers that work for any fitted model with `predict_proba`.

Attributions here are associational: they describe what the model uses, not what causes
major traffic impact (notebook 07 states the limits).
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
import xgboost
from catboost import EFstrType, Pool
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from accidents import config, features, models

# Feature families, so that e.g. lat, lng, state and the grid history read as one "place".
FAMILIES = {
    "place": ["lat", "lng", "state", "cell_count", "cell_rate"],
    "time": ["hour", "weekday", "month", "is_weekend", "is_holiday"],
    "weather": [*features.WEATHER_COLUMNS.values(), "weather", "wind_direction"],
    "road": ["road_class", *features.POI_COLUMNS.values()],
}
FAMILY_OF = {name: family for family, names in FAMILIES.items() for name in names}


def tree_contributions(model, x: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Exact TreeSHAP values in log-odds, from each library's own implementation, and the
    base value they start from. Row sums plus the base value equal the model's log-odds."""
    if isinstance(model, LGBMClassifier):
        raw = np.asarray(model.predict(x, pred_contrib=True))
    elif isinstance(model, XGBClassifier):
        matrix = xgboost.DMatrix(x, enable_categorical=True)
        raw = model.get_booster().predict(matrix, pred_contribs=True)
    elif isinstance(model, models.CatBoostOnStrings):
        pool = Pool(models.as_strings(x), cat_features=features.CATEGORICAL_FEATURES)
        raw = np.asarray(model.model.get_feature_importance(pool, type=EFstrType.ShapValues))
    else:
        raise TypeError(f"no native TreeSHAP for {type(model).__name__}")
    contributions = pd.DataFrame(raw[:, :-1], columns=x.columns, index=x.index)
    return contributions, float(raw[0, -1])


def family_contributions(contributions: pd.DataFrame) -> pd.DataFrame:
    """Per-row contribution of each family. SHAP values add up, so a family's contribution
    is the sum of its members' contributions."""
    return pd.DataFrame(
        {
            family: contributions[[n for n in names if n in contributions]].sum(axis=1)
            for family, names in FAMILIES.items()
        }
    )


def mean_absolute(contributions: pd.DataFrame) -> pd.Series:
    """Global importance: average size of a feature's push, in either direction."""
    return contributions.abs().mean().sort_values(ascending=False)


def permutation_importance(
    predict: Callable[[pd.DataFrame], np.ndarray],
    x: pd.DataFrame,
    y: np.ndarray,
    repeats: int = 3,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Drop in average precision when one feature's values are shuffled across rows,
    which breaks its link to the label while keeping its distribution."""
    rng = np.random.default_rng(seed)
    baseline = average_precision_score(y, predict(x))
    rows = []
    for name in x.columns:
        drops = []
        for _ in range(repeats):
            shuffled = x.copy()
            # iloc keeps the column's dtype; a NumPy round trip turns categories into objects.
            shuffled[name] = x[name].iloc[rng.permutation(len(x))].set_axis(x.index)
            drops.append(baseline - average_precision_score(y, predict(shuffled)))
        rows.append({"feature": name, "ap_drop": np.mean(drops), "sd": np.std(drops)})
    return pd.DataFrame(rows).set_index("feature").sort_values("ap_drop", ascending=False)


def rank_agreement(importances: dict[str, pd.Series]) -> pd.DataFrame:
    """Kendall's tau between every pair of importance rankings (1 = same order)."""
    names = list(importances)
    common = sorted(set.intersection(*(set(s.index) for s in importances.values())))
    table = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            tau = importances[a][common].corr(importances[b][common], method="kendall")
            table.loc[a, b] = tau
    return table


def deletion_test(
    predict: Callable[[pd.DataFrame], np.ndarray],
    x: pd.DataFrame,
    contributions: pd.DataFrame,
    reference: pd.Series,
    steps: int = 8,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Faithfulness check. For each row, replace its k most positive-contributing features
    with a reference value (median or most common) and record the mean prediction; repeat
    with k random features. If the attributions are faithful, removing the top features
    lowers the prediction much faster than removing random ones."""
    rng = np.random.default_rng(seed)
    orders = {
        "top attributions": np.argsort(-contributions[x.columns].to_numpy(), axis=1),
        "random features": np.argsort(rng.random((len(x), len(x.columns))), axis=1),
    }
    rows = []
    for k in range(steps + 1):
        row: dict[str, float] = {"features removed": k}
        for label, order in orders.items():
            masked = x.copy()
            for position, column in enumerate(x.columns):
                chosen = (order[:, :k] == position).any(axis=1)
                masked.loc[chosen, column] = reference[column]
            row[label] = float(predict(masked).mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("features removed")


def reference_values(x: pd.DataFrame) -> pd.Series:
    """Median of each numeric feature and most common level of each category. The median
    is an observed value ("nearest"), so it fits integer columns."""
    values = {}
    for name in x.columns:
        if isinstance(x[name].dtype, pd.CategoricalDtype):
            values[name] = x[name].mode().iloc[0]
        else:
            values[name] = x[name].quantile(0.5, interpolation="nearest")
    return pd.Series(values, dtype=object)


def partial_dependence_at_rows(
    margin: Callable[[pd.DataFrame], np.ndarray], x: pd.DataFrame, columns: list[str]
) -> np.ndarray:
    """Centred partial dependence of `columns`, held jointly, at each row's own values: for
    row i, the mean model output over all rows after setting `columns` to row i's values."""
    n = len(x)
    base = x.reset_index(drop=True)
    # Block i of the grid is every row of `base` with `columns` set to row i's values.
    grid = base.iloc[np.tile(np.arange(n), n)].reset_index(drop=True)
    for column in columns:
        grid[column] = base[column].iloc[np.repeat(np.arange(n), n)].reset_index(drop=True)
    values = margin(grid).reshape(n, n).mean(axis=1)
    return values - values.mean()


def interaction_strength(
    margin: Callable[[pd.DataFrame], np.ndarray], x: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Friedman and Popescu's H statistic for every pair of `columns`, unnormalised: the
    root mean square of the pair's joint partial dependence left after their two separate
    ones, in the units of `margin` (log-odds). It needs only predictions, so categorical
    splits are handled exactly. Cost: len(x) squared predictions per feature and per pair."""
    single = {c: partial_dependence_at_rows(margin, x, [c]) for c in columns}
    table = pd.DataFrame(0.0, index=columns, columns=columns)
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            joint = partial_dependence_at_rows(margin, x, [a, b])
            value = float(np.sqrt(np.mean((joint - single[a] - single[b]) ** 2)))
            table.loc[a, b] = value
            table.loc[b, a] = value
    return table


def numeric_matrix(x: pd.DataFrame) -> np.ndarray:
    """The table as numbers for tools that only take arrays: categories become codes."""
    columns = []
    for name in x.columns:
        values = x[name]
        if isinstance(values.dtype, pd.CategoricalDtype):
            columns.append(values.cat.codes.to_numpy(dtype=float))
        else:
            columns.append(values.to_numpy(dtype=float))
    return np.column_stack(columns)


def table_from_matrix(matrix: np.ndarray, like: pd.DataFrame) -> pd.DataFrame:
    """Inverse of `numeric_matrix`, using `like` for column names and category levels."""
    out = {}
    for position, name in enumerate(like.columns):
        values = matrix[:, position]
        dtype = like[name].dtype
        if isinstance(dtype, pd.CategoricalDtype):
            codes = np.clip(np.rint(values).astype(int), 0, len(dtype.categories) - 1)
            out[name] = pd.Categorical.from_codes(codes, dtype=dtype)
        else:
            out[name] = values
    return pd.DataFrame(out)


def table_from_strings(frame: pd.DataFrame, like: pd.DataFrame) -> pd.DataFrame:
    """Restore the model's column types for tools (such as DiCE) that pass text."""
    out = {}
    for name in like.columns:
        dtype = like[name].dtype
        if isinstance(dtype, pd.CategoricalDtype):
            out[name] = pd.Categorical(frame[name].astype(str), dtype=dtype)
        else:
            out[name] = frame[name].astype(float).to_numpy()
    return pd.DataFrame(out, index=frame.index)
