import math

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

from accidents import calibrate, export, features


def decide(tree: dict, node: int, value: float) -> int:
    """LightGBM's split rules (NumericalDecision / CategoricalDecision)."""
    if tree["decision"][node] == "==":
        if math.isnan(value) or value < 0:
            return tree["right"][node]
        return tree["left"][node] if int(value) in tree["threshold"][node] else tree["right"][node]
    missing = tree["missing"][node]
    if math.isnan(value) and missing != "NaN":
        value = 0.0
    if (missing == "Zero" and value == 0.0) or (missing == "NaN" and math.isnan(value)):
        return tree["left"][node] if tree["default_left"][node] else tree["right"][node]
    return tree["left"][node] if value <= tree["threshold"][node] else tree["right"][node]


def reference_margin(model_json: dict, row: dict) -> float:
    """Score one accident from the exported JSON alone, the way the dashboard will."""
    values = []
    for name in model_json["features"]:
        if name in model_json["categories"]:
            levels = model_json["categories"][name]
            values.append(float(levels.index(row[name])) if row[name] in levels else -1.0)
        else:
            values.append(float("nan") if row[name] is None else float(row[name]))
    total = 0.0
    for tree in model_json["trees"]:
        node = 0
        while node >= 0:
            node = decide(tree, node, values[tree["feature"][node]])
        total += tree["leaf_value"][-node - 1]
    return total


def toy_table(n=4000, seed=3):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({name: rng.normal(size=n) for name in features.NUMERIC_FEATURES})
    x.loc[::5, "temperature_f"] = np.nan
    x["state"] = pd.Categorical(
        rng.choice(["CA", "TX", "VA", "PA"], n), categories=["CA", "PA", "TX", "VA"]
    )
    x["road_class"] = pd.Categorical(
        rng.choice(features.ROAD_CLASSES, n), categories=features.ROAD_CLASSES
    )
    x["weather"] = pd.Categorical(
        rng.choice(features.WEATHER_CATEGORIES, n), categories=features.WEATHER_CATEGORIES
    )
    x["wind_direction"] = pd.Categorical(
        rng.choice(features.WIND_DIRECTIONS, n), categories=features.WIND_DIRECTIONS
    )
    logit = -3 + x["lat"] + 1.5 * (x["state"] == "VA") - 0.8 * x["temperature_f"].isna()
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return x[features.FEATURES], y.to_numpy()


def test_exported_trees_reproduce_lightgbm_exactly():
    x, y = toy_table()
    model = LGBMClassifier(n_estimators=40, num_leaves=15, verbose=-1).fit(x, y)
    model_json = export.lightgbm_trees(model)
    rows = export.parity_fixtures(model, x.head(300), calibrate.IDENTITY)
    for row in rows:
        assert reference_margin(model_json, row["features"]) == pytest.approx(
            row["margin"], abs=1e-9
        )
    assert model_json["categories"]["state"] == ["CA", "PA", "TX", "VA"]


def test_fixture_contributions_add_up_and_missing_becomes_null():
    x, y = toy_table(n=1500)
    model = LGBMClassifier(n_estimators=20, verbose=-1).fit(x, y)
    rows = export.parity_fixtures(model, x.head(20), calibrate.IDENTITY)
    for row in rows:
        assert sum(row["contributions"]) + row["base_value"] == pytest.approx(
            row["margin"], abs=1e-6
        )
        assert row["probability"] == pytest.approx(1 / (1 + math.exp(-row["margin"])), abs=1e-9)
    assert rows[0]["features"]["temperature_f"] is None  # row 0 had a missing temperature


def test_unseen_category_scores_like_lightgbm():
    x, y = toy_table(n=1500)
    model = LGBMClassifier(n_estimators=20, verbose=-1).fit(x, y)
    model_json = export.lightgbm_trees(model)
    row = export.parity_fixtures(model, x.head(1), calibrate.IDENTITY)[0]["features"]
    row["state"] = "ZZ"  # not a training level: code -1, LightGBM sends it right
    probe = x.head(1).copy()
    probe["state"] = pd.Categorical([np.nan], categories=x["state"].cat.categories)
    expected = float(np.asarray(model.predict(probe, raw_score=True))[0])
    assert reference_margin(model_json, row) == pytest.approx(expected, abs=1e-9)


def test_holidays_cover_independence_day_and_cell_history_skips_empty_cells():
    assert "2023-07-04" in export.holidays()
    clean = pd.DataFrame(
        {
            "Start_Lat": [35.1, 35.1, 40.0],
            "Start_Lng": [-80.0, -80.0, -75.0],
            "Start_Time": pd.to_datetime(
                ["2023-03-01 10:00", "2023-03-02 10:00", "2021-01-01 10:00"]
            ),
            "y": [1, 0, 1],
        }
    )
    history = export.cell_history(clean, prior_rate=0.03, as_of="2023-04-01")
    cell = features.grid_cell(pd.Series([35.1]), pd.Series([-80.0])).iloc[0]
    assert history["cells"][cell][0] == 2
    assert len(history["cells"]) == 1  # the 2021 cell is outside the 365-day window


def test_records_makes_tables_json_ready():
    frame = pd.DataFrame(
        {"rate": [0.0123456789, np.nan], "flag": [True, False]},
        index=pd.period_range("2022-01", periods=2, freq="M"),
    )
    rows = export.records(frame)
    assert rows[0] == {"index": "2022-01", "rate": 0.012346, "flag": True}
    assert rows[1]["rate"] is None
