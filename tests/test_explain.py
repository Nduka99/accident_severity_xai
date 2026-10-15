import numpy as np
import pandas as pd
import pytest

from accidents import explain, features


def test_every_feature_belongs_to_exactly_one_family():
    members = [name for names in explain.FAMILIES.values() for name in names]
    assert sorted(members) == sorted(features.FEATURES)


def test_family_contributions_add_members():
    contributions = pd.DataFrame({"lat": [0.1, -0.2], "lng": [0.3, 0.0], "hour": [0.5, 0.5]})
    families = explain.family_contributions(contributions)
    assert families["place"].tolist() == pytest.approx([0.4, -0.2])
    assert families["time"].tolist() == pytest.approx([0.5, 0.5])


def toy():
    """Label depends on `a` only; `b` is noise."""
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"a": rng.normal(size=4000), "b": rng.normal(size=4000)})
    y = (rng.random(4000) < 1 / (1 + np.exp(-3 * x["a"]))).astype(int).to_numpy()

    def predict(frame: pd.DataFrame) -> np.ndarray:
        return 1 / (1 + np.exp(-3 * frame["a"].to_numpy(dtype=float)))

    return x, y, predict


def test_permutation_importance_finds_the_used_feature():
    x, y, predict = toy()
    table = explain.permutation_importance(predict, x, y, repeats=2)
    assert table.index[0] == "a"
    assert table.loc["b", "ap_drop"] == pytest.approx(0.0, abs=1e-12)


def test_permutation_importance_keeps_categorical_dtype():
    x, y, _ = toy()
    x["c"] = pd.Categorical(np.where(x["b"] > 0, "up", "down"))

    def predict(frame: pd.DataFrame) -> np.ndarray:
        assert isinstance(frame["c"].dtype, pd.CategoricalDtype)
        return 1 / (1 + np.exp(-3 * frame["a"].to_numpy(dtype=float)))

    table = explain.permutation_importance(predict, x, y, repeats=1)
    assert table.loc["c", "ap_drop"] == pytest.approx(0.0, abs=1e-12)


def test_reference_values_fit_integer_columns():
    x = pd.DataFrame({"count": pd.array([1, 2, 3, 4], dtype="int32")})
    reference = explain.reference_values(x)
    assert reference["count"] in (2, 3)
    x.loc[[0], "count"] = reference["count"]  # an even-length median of 2.5 would fail here


def test_rank_agreement_is_one_for_identical_rankings():
    s = pd.Series({"a": 3.0, "b": 2.0, "c": 1.0})
    agreement = explain.rank_agreement({"x": s, "y": s * 10, "z": -s})
    assert agreement.loc["x", "y"] == pytest.approx(1.0)
    assert agreement.loc["x", "z"] == pytest.approx(-1.0)


def test_deletion_test_drops_faster_for_true_attributions():
    x, _, predict = toy()
    rows = x[x["a"] > 1].head(200).reset_index(drop=True)
    contributions = pd.DataFrame({"a": 3 * rows["a"], "b": np.zeros(len(rows))})
    curve = explain.deletion_test(
        predict, rows, contributions, explain.reference_values(x), steps=1
    )
    top = curve["top attributions"].to_numpy()
    random = curve["random features"].to_numpy()
    assert top[1] < random[1]
    assert top[0] == random[0]  # nothing removed yet


def small_table(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({name: rng.normal(size=n) for name in features.NUMERIC_FEATURES})
    x["state"] = pd.Categorical(rng.choice(["CA", "VA", "TX"], n))
    x["road_class"] = pd.Categorical(rng.choice(features.ROAD_CLASSES, n))
    x["weather"] = pd.Categorical(rng.choice(features.WEATHER_CATEGORIES, n))
    x["wind_direction"] = pd.Categorical(rng.choice(features.WIND_DIRECTIONS, n))
    y = (rng.random(n) < 1 / (1 + np.exp(2 - x["lat"] - (x["state"] == "VA")))).astype(int)
    return x[features.FEATURES], y.to_numpy()


def log_odds(p):
    return np.log(p / (1 - p))


def test_tree_contributions_add_up_to_each_models_log_odds():
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    from accidents import models

    x, y = small_table()
    fitted = [
        LGBMClassifier(n_estimators=30, verbose=-1).fit(x, y),
        XGBClassifier(n_estimators=30, enable_categorical=True, tree_method="hist").fit(x, y),
        models.CatBoostOnStrings(
            CatBoostClassifier(iterations=30, verbose=0, allow_writing_files=False,
                               cat_features=features.CATEGORICAL_FEATURES)
            .fit(models.as_strings(x), y)
        ),
    ]  # fmt: skip
    for model in fitted:
        contributions, base = explain.tree_contributions(model, x.head(200))
        p = np.asarray(model.predict_proba(x.head(200)))[:, 1]
        assert np.allclose(contributions.sum(axis=1) + base, log_odds(p), atol=1e-4)


def test_interaction_strength_is_zero_for_additive_models_only():
    rng = np.random.default_rng(1)
    x = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60)})
    x["s"] = pd.Categorical(np.where(rng.random(60) > 0.5, "up", "down"))

    def additive(frame: pd.DataFrame) -> np.ndarray:
        assert isinstance(frame["s"].dtype, pd.CategoricalDtype)
        return (frame["a"] + frame["b"] ** 2 + (frame["s"] == "up")).to_numpy(dtype=float)

    def product(frame: pd.DataFrame) -> np.ndarray:
        return (frame["a"] * frame["b"] + frame["a"] * (frame["s"] == "up")).to_numpy(dtype=float)

    columns = ["a", "b", "s"]
    flat = explain.interaction_strength(additive, x, columns)
    assert np.allclose(flat.to_numpy(), 0.0, atol=1e-12)
    joint = explain.interaction_strength(product, x, columns).to_numpy()  # order: a, b, s
    assert joint[0, 1] > 0.3  # a × b
    assert joint[0, 2] > 0.2  # a × s
    assert joint[1, 2] == pytest.approx(0.0, abs=1e-12)  # b × s
    assert np.allclose(joint, joint.T)


def test_matrix_and_string_round_trips_restore_the_table():
    x, _ = small_table(n=50)
    back = explain.table_from_matrix(explain.numeric_matrix(x), x)
    pd.testing.assert_frame_equal(back, x.reset_index(drop=True), check_dtype=False)
    as_text = x.astype(str)
    restored = explain.table_from_strings(as_text, x)
    assert (restored["state"] == x["state"]).all()
    assert np.allclose(restored["lat"].to_numpy(), x["lat"].to_numpy())
