"""The benchmark's models, each as a small spec: a search space, a fit function, and
whether the fitted model can be exported to the in-browser dashboard.

Trees (LightGBM, XGBoost, CatBoost, EBM) take the feature table as it is, with missing
values and categories handled natively. Logistic regression and the neural models cannot
take missing values, so they get training-set medians plus one flag per weather reading:
the flag keeps the signal that a reading was missing (notebook 02).
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, SplineTransformer
from xgboost import XGBClassifier

from accidents import config, features

WEATHER = list(features.WEATHER_COLUMNS.values())
CONTINUOUS = ["lat", "lng", *WEATHER, "cell_count", "cell_rate"]
CYCLIC_PERIODS = {"hour": 24, "weekday": 7, "month": 12}
BINARY = ["is_weekend", "is_holiday", *features.POI_COLUMNS.values()]
MISSING_FLAGS = [f"{name}_missing" for name in WEATHER]
CATBOOST_DIR = config.ROOT / "data" / "processed" / "catboost_info"
THREADS = os.cpu_count() or 4


def training_medians(x: pd.DataFrame) -> dict[str, float]:
    """Fill values for the weather readings, learned on training rows only."""
    return {name: float(x[name].median()) for name in WEATHER}


def impute_with_flags(x: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    out = x.copy()
    for name, median in medians.items():
        out[f"{name}_missing"] = out[name].isna().astype("int8")
        out[name] = out[name].fillna(median)
    return out


@dataclass(frozen=True)
class ModelSpec:
    name: str
    exportable: bool  # can the fitted model run in the browser dashboard (JSON trees/tables)
    needs_imputation: bool
    suggest: Callable[[optuna.Trial], dict]
    defaults: dict
    # (params, x_train, y_train, x_stop, y_stop) -> fitted model; x_stop/y_stop only
    # decide when iterative training stops, never which model wins.
    fit: Callable


# ---- Reference points ---------------------------------------------------------------


def fit_dummy(params, x_train, y_train, x_valid, y_valid):
    return DummyClassifier(strategy="prior").fit(x_train, y_train)


# ---- Logistic regression on spline features -------------------------------------------


def suggest_logistic(trial: optuna.Trial) -> dict:
    return {
        "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        "n_knots": trial.suggest_int("n_knots", 4, 10),
    }


def fourier_terms(x: pd.DataFrame) -> np.ndarray:
    """Sine and cosine of the first three harmonics of hour, weekday and month: 23:00
    sits next to 00:00, and a daily rhythm can have more than one peak."""
    columns = []
    for name, period in CYCLIC_PERIODS.items():
        angle = 2 * np.pi * x[name].to_numpy(dtype=float) / period
        for k in (1, 2, 3):
            columns += [np.sin(k * angle), np.cos(k * angle)]
    return np.column_stack(columns)


def fit_logistic(params, x_train, y_train, x_valid, y_valid):
    """Splines with quantile knots let a linear model bend along skewed readings;
    Fourier terms do the same for cyclic time."""
    encode = ColumnTransformer(
        [
            ("splines", SplineTransformer(n_knots=params["n_knots"], knots="quantile"), CONTINUOUS),
            ("fourier", FunctionTransformer(fourier_terms), list(CYCLIC_PERIODS)),
            ("binary", "passthrough", BINARY + MISSING_FLAGS),
            ("onehot", OneHotEncoder(handle_unknown="ignore"), features.CATEGORICAL_FEATURES),
        ]
    )
    model = LogisticRegression(C=params["C"], max_iter=2000)
    return make_pipeline(encode, model).fit(x_train, y_train)


# ---- Explainable Boosting Machine (glass-box GAM with pairwise interactions) --------


def suggest_ebm(trial: optuna.Trial) -> dict:
    return {
        "interactions": trial.suggest_int("interactions", 0, 20),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_leaves": trial.suggest_int("max_leaves", 2, 4),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 200, log=True),
        "max_bins": trial.suggest_categorical("max_bins", [128, 256, 512]),
    }


def fit_ebm(params, x_train, y_train, x_valid, y_valid):
    model = ExplainableBoostingClassifier(
        **params, outer_bags=8, n_jobs=THREADS, random_state=config.SEED
    )
    return model.fit(x_train, y_train)


# ---- Gradient-boosted trees ------------------------------------------------------------


def suggest_lightgbm(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 511, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 2000, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
        "cat_smooth": trial.suggest_float("cat_smooth", 1.0, 100.0, log=True),
    }


def fit_lightgbm(params, x_train, y_train, x_valid, y_valid):
    model = LGBMClassifier(
        **params,
        n_estimators=5000,
        subsample_freq=1,
        metric="binary_logloss",
        n_jobs=THREADS,
        random_state=config.SEED,
        verbose=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_X=x_valid,
        eval_y=y_valid,
        callbacks=[early_stopping(200, verbose=False), log_evaluation(0)],
    )
    return model


# Search ranges follow published tabular benchmarks (tree depth 3-10). Learning rates
# start at 0.03 so early stopping, not the 5000-round cap, ends training: measured here,
# CatBoost depth 10 costs about 2.5x depth 6 per round, and depth 10 at a 0.02 learning
# rate made one fit take over ten minutes on 20k rows.
def suggest_xgboost(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 200.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
    }


def fit_xgboost(params, x_train, y_train, x_valid, y_valid):
    model = XGBClassifier(
        **params,
        n_estimators=5000,
        tree_method="hist",
        device="cuda",
        enable_categorical=True,
        max_cat_to_onehot=1,
        eval_metric="logloss",
        early_stopping_rounds=200,
        random_state=config.SEED,
    )
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    # Predict on CPU: the inputs live there, and the saved model then loads anywhere.
    return model.set_params(device="cpu")


def as_strings(x: pd.DataFrame) -> pd.DataFrame:
    """CatBoost wants categories as plain strings."""
    out = x.copy()
    for name in features.CATEGORICAL_FEATURES:
        out[name] = out[name].astype(str)
    return out


def suggest_catboost(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
    }


class CatBoostOnStrings:
    """CatBoost with the string conversion built in, so it scores the plain table."""

    def __init__(self, model: CatBoostClassifier):
        self.model = model

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(as_strings(x))


def fit_catboost(params, x_train, y_train, x_valid, y_valid):
    Path(CATBOOST_DIR).mkdir(parents=True, exist_ok=True)
    model = CatBoostClassifier(
        **params,
        iterations=5000,
        task_type="GPU",
        border_count=128,  # the GPU default; 254 roughly doubles the cost per round
        cat_features=features.CATEGORICAL_FEATURES,
        od_type="Iter",
        od_wait=200,
        random_seed=config.SEED,
        train_dir=str(CATBOOST_DIR),
        verbose=0,
    )
    model.fit(as_strings(x_train), y_train, eval_set=(as_strings(x_valid), y_valid))
    return CatBoostOnStrings(model)


# ---- Deep tabular models (pytabkit, GPU) ----------------------------------------------


def suggest_realmlp(trial: optuna.Trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 0.01, 0.15, log=True),
        "p_drop": trial.suggest_categorical("p_drop", [0.0, 0.15, 0.3]),
        "wd": trial.suggest_categorical("wd", [0.0, 0.02]),
        "hidden_width": trial.suggest_categorical("hidden_width", [256, 512]),
        "use_plr_embeddings": trial.suggest_categorical("use_plr_embeddings", [True, False]),
    }


def fit_realmlp(params, x_train, y_train, x_valid, y_valid):
    from pytabkit import RealMLP_TD_Classifier

    hidden = params.get("hidden_width", 256)
    model = RealMLP_TD_Classifier(
        **{k: v for k, v in params.items() if k != "hidden_width"},
        hidden_sizes=[hidden, hidden, hidden],
        batch_size=2048,
        n_epochs=64,
        use_early_stopping=True,
        # pytabkit picks the best epoch by classification error unless told otherwise; at
        # a 2.8% positive rate that error is flat (always "minor"), so use log loss, and drop
        # label smoothing as pytabkit advises for probability metrics.
        val_metric_name="cross_entropy",
        use_ls=False,
        device="cuda",
        random_state=config.SEED,
        verbosity=0,
    )
    return model.fit(x_train, y_train, X_val=x_valid, y_val=y_valid)


def suggest_tabm(trial: optuna.Trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 3e-4, 5e-3, log=True),  # pytabkit default 0.002
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-4, 1e-3]),
        "dropout": trial.suggest_categorical("dropout", [0.0, 0.1, 0.25]),
        "d_block": trial.suggest_categorical("d_block", [256, 512]),
        "n_blocks": trial.suggest_int("n_blocks", 1, 4),
    }


def fit_tabm(params, x_train, y_train, x_valid, y_valid):
    from pytabkit import TabM_D_Classifier

    model = TabM_D_Classifier(
        **params,
        batch_size=1024,  # pytabkit default 256; larger for 400k+ rows, within the paper's range
        n_epochs=40,  # cap for time; early stopping usually ends sooner
        patience=16,  # pytabkit default
        val_metric_name="cross_entropy",  # same reason as RealMLP above
        device="cuda",
        random_state=config.SEED,
        verbosity=0,
    )
    return model.fit(x_train, y_train, X_val=x_valid, y_val=y_valid)


MODELS = {
    spec.name: spec
    for spec in [
        ModelSpec("dummy", True, False, lambda trial: {}, {}, fit_dummy),
        ModelSpec(
            "logistic", False, True, suggest_logistic, {"C": 1.0, "n_knots": 6}, fit_logistic
        ),
        ModelSpec("ebm", True, False, suggest_ebm, {}, fit_ebm),
        ModelSpec("lightgbm", True, False, suggest_lightgbm, {}, fit_lightgbm),
        ModelSpec("xgboost", True, False, suggest_xgboost, {}, fit_xgboost),
        # Not exportable: CatBoost encodes categories with hashed target statistics over
        # feature combinations (244 tables in a test model), and its ONNX export refuses
        # categorical features, so exact in-browser scoring is not practical.
        ModelSpec("catboost", False, False, suggest_catboost, {}, fit_catboost),
        ModelSpec("realmlp", False, True, suggest_realmlp, {}, fit_realmlp),
        ModelSpec("tabm", False, True, suggest_tabm, {}, fit_tabm),
    ]
}


def boosting_rounds(fitted) -> int:
    """Number of boosting rounds early stopping kept."""
    if isinstance(fitted, LGBMClassifier):
        return int(fitted.best_iteration_)  # LightGBM counts from 1
    if isinstance(fitted, XGBClassifier):
        return int(fitted.best_iteration) + 1  # XGBoost and CatBoost count from 0
    if isinstance(fitted, CatBoostOnStrings):
        return int(fitted.model.get_best_iteration()) + 1
    raise TypeError(f"{type(fitted).__name__} is not a boosting model")


def refit_all_rounds(fitted, x: pd.DataFrame, y: np.ndarray):
    """Refit a boosting model with the same settings on more rows (for example training
    plus early-stopping months) for the number of rounds early stopping chose, so the most
    recent months also shape the model. No early stopping: the round count is fixed."""
    rounds = boosting_rounds(fitted)
    if isinstance(fitted, LGBMClassifier):
        return LGBMClassifier(**{**fitted.get_params(), "n_estimators": rounds}).fit(x, y)
    if isinstance(fitted, XGBClassifier):
        settings = {**fitted.get_params(), "n_estimators": rounds, "early_stopping_rounds": None}
        model = XGBClassifier(**{**settings, "device": "cuda"}).fit(x, y, verbose=False)
        return model.set_params(device="cpu")
    if isinstance(fitted, CatBoostOnStrings):
        settings = {**fitted.model.get_params(), "iterations": rounds}
        for name in ("od_type", "od_wait", "use_best_model"):
            settings.pop(name, None)
        return CatBoostOnStrings(CatBoostClassifier(**settings).fit(as_strings(x), y))
    raise TypeError(f"{type(fitted).__name__} is not a boosting model")
