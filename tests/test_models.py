import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from accidents import features, models


def test_impute_with_flags_keeps_missingness_as_a_feature():
    x = pd.DataFrame({name: [1.0, np.nan, 3.0] for name in models.WEATHER})
    medians = models.training_medians(x)
    out = models.impute_with_flags(x, medians)
    assert out[models.WEATHER].notna().all().all()
    assert out["temperature_f"].tolist() == [1.0, 2.0, 3.0]
    assert out["temperature_f_missing"].tolist() == [0, 1, 0]
    assert x["temperature_f"].isna().sum() == 1  # input untouched


def test_fourier_terms_make_time_circular():
    hours = pd.DataFrame({"hour": [23, 0, 12], "weekday": [0, 0, 0], "month": [1, 1, 1]})
    terms = models.fourier_terms(hours)
    assert terms.shape == (3, 18)  # 3 features × 3 harmonics × (sin, cos)
    near = np.linalg.norm(terms[0] - terms[1])
    far = np.linalg.norm(terms[0] - terms[2])
    assert near < far


def test_every_model_spec_has_a_unique_name_and_a_fit():
    assert set(models.MODELS) == {
        "dummy", "logistic", "ebm", "lightgbm", "xgboost", "catboost", "realmlp", "tabm",
    }  # fmt: skip
    for spec in models.MODELS.values():
        assert callable(spec.fit) and callable(spec.suggest)


def test_lightgbm_fits_the_feature_table_with_missing_values_and_categories():
    rng = np.random.default_rng(0)
    n = 2000
    x = pd.DataFrame({name: rng.normal(size=n) for name in features.NUMERIC_FEATURES})
    x.loc[::7, "temperature_f"] = np.nan
    x["state"] = pd.Categorical(rng.choice(["CA", "VA"], n))
    x["road_class"] = pd.Categorical(rng.choice(features.ROAD_CLASSES, n))
    x["weather"] = pd.Categorical(rng.choice(features.WEATHER_CATEGORIES, n))
    x["wind_direction"] = pd.Categorical(rng.choice(features.WIND_DIRECTIONS, n))
    y = (rng.random(n) < 0.1).astype(int)
    fit_rows, stop_rows = slice(0, 1500), slice(1500, n)
    model = models.fit_lightgbm(
        {"num_leaves": 7}, x.iloc[fit_rows], y[fit_rows], x.iloc[stop_rows], y[stop_rows]
    )
    p = np.asarray(model.predict_proba(x[features.FEATURES]))[:, 1]
    assert p.shape == (n,) and np.all((p > 0) & (p < 1))


def test_refit_all_rounds_keeps_settings_and_round_count():
    rng = np.random.default_rng(1)
    n = 3000
    x = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = (rng.random(n) < 1 / (1 + np.exp(-2 * x["a"]))).astype(int).to_numpy()
    fitted = models.fit_lightgbm({"num_leaves": 7, "learning_rate": 0.1}, x[:2000], y[:2000],
                                 x[2000:], y[2000:])  # fmt: skip
    rounds = models.boosting_rounds(fitted)
    refit = models.refit_all_rounds(fitted, x, y)
    assert 0 < rounds < 5000
    assert isinstance(refit, LGBMClassifier)
    assert refit.n_estimators == rounds
    assert refit.get_params()["num_leaves"] == 7
