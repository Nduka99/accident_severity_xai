import numpy as np
import pytest

from accidents import calibrate, evaluate


def overconfident(n=40000, seed=0):
    """True rates are half of what the model says."""
    rng = np.random.default_rng(seed)
    p = rng.beta(0.6, 10, n)
    y = (rng.random(n) < p / 2).astype(int)
    return p, y


@pytest.mark.parametrize("fit", [calibrate.fit_platt, calibrate.fit_isotonic])
def test_calibrators_reduce_calibration_error_and_keep_order(fit):
    p, y = overconfident()
    calibrator = fit(p[:20000], y[:20000])
    fixed = calibrate.apply(calibrator, p[20000:])
    before = evaluate.expected_calibration_error(y[20000:], p[20000:])
    after = evaluate.expected_calibration_error(y[20000:], fixed)
    assert after < before / 2
    order = np.argsort(p[20000:])
    assert np.all(np.diff(fixed[order]) >= -1e-12)  # monotone: ranking unchanged


def test_calibrators_are_plain_json_data():
    p, y = overconfident(2000)
    for calibrator in (calibrate.fit_platt(p, y), calibrate.fit_isotonic(p, y)):
        values = [v for v in calibrator.values() if not isinstance(v, str)]
        assert all(isinstance(v, (float, list)) for v in values)


def test_cost_threshold():
    assert calibrate.cost_threshold(19) == pytest.approx(0.05)
    assert calibrate.apply(calibrate.IDENTITY, np.array([0.2]))[0] == 0.2
