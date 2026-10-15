"""Everything the browser dashboard needs to score an accident exactly as Python does,
as plain JSON: the trees, the category codes, the location-history lookup, holidays,
calibration, conformal thresholds, and parity fixtures for the TypeScript tests."""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from pandas.tseries.holiday import USFederalHolidayCalendar

from accidents import calibrate, config, features

NODE_KEYS = [
    "feature",
    "threshold",
    "decision",
    "default_left",
    "missing",
    "left",
    "right",
    "count",
]


def flatten_tree(structure: dict) -> dict:
    """One LightGBM tree (nested dicts) as flat arrays.

    Children: an index >= 0 is an internal node, a negative index -k-1 is leaf k. A split
    is numeric ("<=": go left when value <= threshold) or categorical ("==": go left when
    the category code is in `threshold`). Missing values follow LightGBM: with
    missing_type "NaN" they take the default side; with "None" they count as 0.0.
    `count` (rows reaching a node) is what TreeSHAP weighs paths by.
    """
    tree: dict[str, list] = {key: [] for key in [*NODE_KEYS, "leaf_value", "leaf_count"]}

    def visit(node: dict) -> int:
        if "split_index" not in node:
            tree["leaf_value"].append(float(node["leaf_value"]))
            tree["leaf_count"].append(int(node.get("leaf_count", 0)))
            return -len(tree["leaf_value"])  # leaf k is stored as -k-1
        index = len(tree["feature"])
        for key in NODE_KEYS:
            tree[key].append(None)
        tree["feature"][index] = int(node["split_feature"])
        if node["decision_type"] == "==":
            tree["threshold"][index] = [int(c) for c in str(node["threshold"]).split("||")]
        else:
            tree["threshold"][index] = float(node["threshold"])
        tree["decision"][index] = node["decision_type"]
        tree["default_left"][index] = bool(node["default_left"])
        tree["missing"][index] = node["missing_type"]
        tree["count"][index] = int(node["internal_count"])
        tree["left"][index] = visit(node["left_child"])
        tree["right"][index] = visit(node["right_child"])
        return index

    visit(structure)
    return tree


def lightgbm_trees(model: LGBMClassifier) -> dict:
    """The fitted model as plain JSON: feature order, category codes, and every tree."""
    dump = model.booster_.dump_model()
    trees = [flatten_tree(info["tree_structure"]) for info in dump["tree_info"]]
    categories = model.booster_.pandas_categorical or []
    categorical = [f for f in dump["feature_names"] if f in features.CATEGORICAL_FEATURES]
    return {
        "kind": "lightgbm",
        "features": dump["feature_names"],
        "categories": {
            name: list(levels) for name, levels in zip(categorical, categories, strict=True)
        },
        "trees": trees,
    }


def cell_history(clean: pd.DataFrame, prior_rate: float, as_of: str) -> dict:
    """Each grid cell's accident count and smoothed Severity-4 rate over the 365 days
    before `as_of`: the lookup the dashboard uses for a location's history features.
    Cells with no accidents in that window are left out; the dashboard then uses count 0
    and the prior rate, exactly what `features.history_at` gives an empty cell."""
    cells = features.grid_cell(clean["Start_Lat"], clean["Start_Lng"])
    days = clean["Start_Time"].dt.normalize()
    daily = features.daily_cell_totals(cells, days, clean["y"])
    unique_cells = pd.Series(sorted(cells.unique()))
    query_days = pd.Series(pd.Timestamp(as_of), index=unique_cells.index)
    history = features.history_at(daily, unique_cells, query_days, prior_rate)
    table = {}
    for cell, count, rate in zip(
        unique_cells, history["cell_count"], history["cell_rate"], strict=True
    ):
        if count > 0:
            table[cell] = [int(count), round(float(rate), 6)]
    return {"prior_rate": prior_rate, "as_of": as_of, "degrees": config.GRID_DEGREES,
            "window_days": config.HISTORY_DAYS, "cells": table}  # fmt: skip


def holidays(first_year: int = 2020, last_year: int = 2035) -> list[str]:
    """US federal holidays as ISO dates, the same calendar `features.time_features` uses."""
    dates = USFederalHolidayCalendar().holidays(
        start=pd.Timestamp(first_year, 1, 1), end=pd.Timestamp(last_year, 12, 31)
    )
    return [d.strftime("%Y-%m-%d") for d in dates]


def json_value(value: object) -> object:
    """Missing numbers become null in JSON, as the dashboard represents them."""
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def parity_fixtures(model: LGBMClassifier, x: pd.DataFrame, calibrator: dict) -> list[dict]:
    """Rows with their features, the model's raw log-odds, calibrated probability and SHAP
    contributions: the TypeScript scorer must reproduce these to within 1e-6."""
    margin = np.asarray(model.predict(x, raw_score=True))
    p = calibrate.apply(calibrator, 1 / (1 + np.exp(-margin)))
    contributions = np.asarray(model.predict(x, pred_contrib=True))
    rows = []
    for i in range(len(x)):
        record = {name: json_value(x.iloc[i][name]) for name in x.columns}
        for name in features.CATEGORICAL_FEATURES:
            record[name] = str(x.iloc[i][name])
        rows.append({
            "features": record,
            "margin": float(margin[i]),
            "probability": float(p[i]),
            "contributions": [float(c) for c in contributions[i, :-1]],
            "base_value": float(contributions[i, -1]),
        })  # fmt: skip
    return rows


def records(frame: pd.DataFrame | pd.Series) -> list[dict]:
    """A table as JSON-ready rows: the index becomes columns, labels become strings,
    numbers are rounded to 6 decimals, missing values become null."""
    table = frame.to_frame() if isinstance(frame, pd.Series) else frame
    out = table.reset_index()
    out.columns = [str(c) for c in out.columns]
    rows = []
    for row in out.to_dict("records"):
        clean = {}
        for key, value in row.items():
            if isinstance(value, (bool, np.bool_)):
                clean[key] = bool(value)
            elif isinstance(value, (int, float, np.integer, np.floating)):
                number = float(value)
                clean[key] = None if np.isnan(number) else round(number, 6)
            else:
                clean[key] = str(value)
        rows.append(clean)
    return rows
