"""Tune and fit every benchmark model; save validation predictions, fitted models and a
results file per model. Notebook 05 only reads what this writes.

    uv run python -m accidents.benchmark                 # all models (several hours)
    uv run python -m accidents.benchmark lightgbm ebm    # some models
    uv run python -m accidents.benchmark --smoke         # tiny end-to-end check

Resumable: each Optuna study lives in benchmark/optuna.db and continues where it
stopped, and a model whose results file exists is skipped.

Protocol (equal for every model): the library-default configuration and N_TRIALS
Optuna (TPE) trials are fitted on the same 400k training rows; early stopping uses the
last two training months; each configuration is scored by average precision on the
same 200k validation rows. The best configuration is refitted on all training rows and
scored on all validation rows. Test months are never touched here.
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score

from accidents import config, features, models

OUT = config.ROOT / "data" / "processed" / "benchmark"
N_TRIALS = 25
TUNE_ROWS = {"fit": 400_000, "stop": 100_000, "valid": 200_000}
LEARNING_CURVE_ROWS = [50_000, 200_000, 800_000]


def load_splits() -> dict[str, pd.DataFrame]:
    table = pd.read_parquet(config.MODEL_TABLE)
    is_train = table["split"] == "train"
    stop_months = table["start_time"] >= pd.Timestamp(config.STOP_START)
    return {
        "fit": table[is_train & ~stop_months],
        "stop": table[is_train & stop_months],
        "valid": table[table["split"] == "valid"],
    }


def sample(frame: pd.DataFrame, rows: int) -> pd.DataFrame:
    return frame.sample(min(rows, len(frame)), random_state=config.SEED)


def prepare(spec: models.ModelSpec, frame: pd.DataFrame, medians: dict) -> tuple:
    x = frame[features.FEATURES]
    if spec.needs_imputation:
        x = models.impute_with_flags(x, medians)
    return x, frame["y"].to_numpy()


def score(model, x: pd.DataFrame, y: np.ndarray) -> float:
    return float(average_precision_score(y, model.predict_proba(x)[:, 1]))


def tune(spec: models.ModelSpec, data: dict, medians: dict, trials: int, out: Path) -> dict:
    """Default configuration plus `trials` TPE trials on the tuning subsample."""
    fit_x, fit_y = prepare(spec, data["fit"], medians)
    stop_x, stop_y = prepare(spec, data["stop"], medians)
    valid_x, valid_y = prepare(spec, data["valid"], medians)
    default = spec.fit(spec.defaults, fit_x, fit_y, stop_x, stop_y)
    result = {"default_params": spec.defaults, "default_ap": score(default, valid_x, valid_y)}
    if spec.name == "dummy":
        return result | {"best_params": {}, "best_ap": result["default_ap"], "trials": 0}

    def objective(trial: optuna.Trial) -> float:
        model = spec.fit(spec.suggest(trial), fit_x, fit_y, stop_x, stop_y)
        return score(model, valid_x, valid_y)

    study = optuna.create_study(
        study_name=spec.name,
        storage=f"sqlite:///{(out / 'optuna.db').as_posix()}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.SEED),
        load_if_exists=True,
    )
    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    study.optimize(objective, n_trials=max(0, trials - done), catch=(Exception,))
    best_is_default = result["default_ap"] >= study.best_value
    return result | {
        "best_params": spec.defaults if best_is_default else study.best_params,
        "best_ap": max(result["default_ap"], study.best_value),
        "trials": trials,
    }


def run(name: str, data: dict, trials: int, out: Path, curve_rows: list[int]) -> None:
    spec = models.MODELS[name]
    results_file = out / f"{name}.json"
    if results_file.exists():
        print(f"{name}: done already, skipping")
        return
    started = time.perf_counter()
    medians = models.training_medians(data["fit"][features.FEATURES])
    tuning = {key: sample(frame, TUNE_ROWS[key]) for key, frame in data.items()}
    result = tune(spec, tuning, medians, trials, out)
    print(f"{name}: tuned (default AP {result['default_ap']:.4f}, best {result['best_ap']:.4f})")

    fit_x, fit_y = prepare(spec, data["fit"], medians)
    stop_x, stop_y = prepare(spec, data["stop"], medians)
    valid_x, valid_y = prepare(spec, data["valid"], medians)
    fit_started = time.perf_counter()
    model = spec.fit(result["best_params"], fit_x, fit_y, stop_x, stop_y)
    fit_seconds = time.perf_counter() - fit_started
    predict_started = time.perf_counter()
    p_valid = model.predict_proba(valid_x)[:, 1]
    predict_seconds = time.perf_counter() - predict_started

    curve = {}
    for rows in curve_rows:
        subset_x, subset_y = prepare(spec, sample(data["fit"], rows), medians)
        small = spec.fit(result["best_params"], subset_x, subset_y, stop_x, stop_y)
        curve[rows] = score(small, valid_x, valid_y)
    curve[len(fit_x)] = float(average_precision_score(valid_y, p_valid))

    model_file = out / f"{name}.joblib"
    joblib.dump({"model": model, "medians": medians}, model_file)
    pd.DataFrame({"id": data["valid"]["id"].to_numpy(), "y": valid_y, "p": p_valid}).to_parquet(
        out / f"{name}_valid.parquet", index=False
    )
    result |= {
        "model": name,
        "exportable": spec.exportable,
        "valid_ap": curve[len(fit_x)],
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "valid_rows": len(valid_y),
        "model_megabytes": model_file.stat().st_size / 1e6,
        "learning_curve": {str(k): v for k, v in curve.items()},
        "total_minutes": (time.perf_counter() - started) / 60,
    }
    results_file.write_text(json.dumps(result, indent=2, default=str))
    print(f"{name}: valid AP {result['valid_ap']:.4f} in {result['total_minutes']:.1f} min")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune and fit every benchmark model.")
    parser.add_argument("models", nargs="*", default=list(models.MODELS))
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--smoke", action="store_true", help="tiny data, 2 trials, own folder")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    data = load_splits()
    out, trials, curve_rows = OUT, args.trials, LEARNING_CURVE_ROWS
    if args.smoke:
        data = {key: sample(frame, 20_000) for key, frame in data.items()}
        out, trials, curve_rows = OUT.parent / "benchmark_smoke", 2, [5_000]
    out.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        run(name, data, trials, out, curve_rows)


if __name__ == "__main__":
    main()
