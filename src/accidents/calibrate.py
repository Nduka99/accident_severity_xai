"""Probability calibration as plain JSON-compatible data, so the dashboard applies exactly
the same mapping as Python: Platt scaling is two numbers, isotonic regression a table."""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPSILON = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPSILON, 1 - EPSILON)
    return np.log(p / (1 - p))


def fit_platt(p: np.ndarray, y: np.ndarray) -> dict:
    """Logistic regression on the model's log-odds: rescales without reordering."""
    model = LogisticRegression(C=1e6).fit(logit(p).reshape(-1, 1), y)
    return {"kind": "platt", "a": float(model.coef_[0, 0]), "b": float(model.intercept_[0])}


def fit_isotonic(p: np.ndarray, y: np.ndarray) -> dict:
    """Monotone step mapping; stored as the breakpoints sklearn interpolates between."""
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
    return {
        "kind": "isotonic",
        "x": model.X_thresholds_.tolist(),
        "y": model.y_thresholds_.tolist(),
    }


def apply(calibrator: dict, p: np.ndarray) -> np.ndarray:
    if calibrator["kind"] == "identity":
        return np.asarray(p, dtype=float)
    if calibrator["kind"] == "platt":
        z = calibrator["a"] * logit(p) + calibrator["b"]
        return 1 / (1 + np.exp(-z))
    if calibrator["kind"] == "isotonic":
        return np.interp(p, calibrator["x"], calibrator["y"])
    raise ValueError(f"unknown calibrator {calibrator['kind']}")


IDENTITY = {"kind": "identity"}


def cost_threshold(cost_ratio: float) -> float:
    """With calibrated probabilities, flagging is worth it when p exceeds this: missing a
    major-impact accident costs `cost_ratio` times a false alarm."""
    return 1 / (1 + cost_ratio)
