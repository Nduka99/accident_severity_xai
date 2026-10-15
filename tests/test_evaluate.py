import numpy as np
import pandas as pd
import pytest

from accidents import evaluate


def synthetic(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.beta(0.5, 12, n)  # low prevalence, like the real target
    y = (rng.random(n) < p).astype(int)
    return y, p


def test_ece_is_small_for_calibrated_and_large_for_shifted():
    y, p = synthetic()
    assert evaluate.expected_calibration_error(y, p) < 0.01
    assert evaluate.expected_calibration_error(y, np.clip(p + 0.2, 0, 1)) > 0.15


def test_bootstrap_is_paired_and_brackets_the_estimate():
    y, p = synthetic()
    days = np.repeat(np.arange(200), len(y) // 200)
    noisy = np.clip(p + np.random.default_rng(1).normal(0, 0.05, len(p)), 0, 1)
    samples = evaluate.bootstrap(y, {"good": p, "noisy": noisy}, days, n_boot=200)
    low, high = evaluate.interval(samples["good"])
    point = evaluate.scores(y, p)["average_precision"]
    assert low <= point <= high
    assert (samples["good"] - samples["noisy"]).mean() > 0


def test_net_benefit_of_perfect_model_equals_prevalence_at_low_threshold():
    y = np.array([0, 0, 0, 1])
    curve = evaluate.net_benefit(y, y.astype(float), np.array([0.01]))
    assert curve["model"].iloc[0] == pytest.approx(0.25)
    assert curve["act_on_none"].iloc[0] == 0.0


def test_conformal_bands_reach_class_conditional_coverage():
    y, p = synthetic(n=40000)
    cal, new = slice(0, 20000), slice(20000, 40000)
    thresholds = evaluate.conformal_thresholds(y[cal], p[cal], alpha=0.1)
    band = evaluate.conformal_band(p[new], thresholds)
    y_new = y[new]
    covered_minor = np.isin(band[y_new == 0], ["minor", "uncertain"]).mean()
    covered_major = np.isin(band[y_new == 1], ["major", "uncertain"]).mean()
    assert covered_minor >= 0.88
    assert covered_major >= 0.88


def test_wilson_interval_brackets_rate_and_stays_in_unit_range():
    low, high = evaluate.wilson_interval(np.array([0, 3, 50]), np.array([10, 100, 50]))
    assert low[0] == 0.0 and high[0] > 0
    assert low[1] < 0.03 < high[1]
    assert high[2] <= 1.0


def test_rate_table_respects_min_count():
    y = pd.Series([1, 0, 0, 1, 0])
    group = pd.Series(["a", "a", "a", "b", "c"])
    table = evaluate.rate_table(y, group, min_count=2)
    assert table.index.tolist() == ["a"]
    assert table.loc["a", "rate"] == pytest.approx(1 / 3)


def test_rows_by_group_covers_every_row_once():
    groups = pd.Series(["b", "a", "b", "c", "a"])
    blocks = evaluate.rows_by_group(groups)
    assert sorted(np.concatenate(blocks).tolist()) == [0, 1, 2, 3, 4]
    assert len(blocks) == 3


def test_population_stability_is_zero_for_same_data_and_grows_with_shift():
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(0, 1, 50000))
    same = pd.Series(rng.normal(0, 1, 50000))
    shifted = pd.Series(rng.normal(1, 1, 50000))
    assert evaluate.population_stability(base, same) < 0.01
    assert evaluate.population_stability(base, shifted) > 0.25


def test_population_stability_counts_missing_and_categories():
    expected = pd.Series(pd.Categorical(["a"] * 90 + ["b"] * 10))
    actual = pd.Series(pd.Categorical(["a"] * 50 + ["b"] * 50))
    assert evaluate.population_stability(expected, actual) > 0.25
    numeric = pd.Series([1.0, 2.0, 3.0, 4.0] * 100)
    with_missing = pd.Series([1.0, 2.0, np.nan, np.nan] * 100)
    assert evaluate.population_stability(numeric, with_missing) > 0.25
