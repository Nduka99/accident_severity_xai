import numpy as np
import pandas as pd
import pytest

from accidents import features

# Fields known only after the event, or that encode the label or its source (D4, D6).
BANNED = {
    "Description", "Source", "Severity", "End_Time", "End_Lat", "End_Lng", "Distance(mi)",
    "Duration", "Wind_Chill(F)", "Weather_Timestamp", "Turning_Loop", "y", "id", "cell",
    "start_time",
}  # fmt: skip


def test_feature_whitelist_has_no_banned_or_raw_names():
    assert not BANNED & set(features.FEATURES)
    assert len(features.FEATURES) == len(set(features.FEATURES))


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        ("Light Snow / Windy", "snow_ice"),
        ("Thunder / Wintry Mix", "snow_ice"),
        ("Light Rain with Thunder", "storm"),
        ("Funnel Cloud", "storm"),
        ("Heavy Rain", "rain"),
        ("Duststorm", "fog_obscured"),
        ("Partly Cloudy", "cloudy"),
        ("Fair / Windy", "clear"),
        ("N/A Precipitation", "unknown"),
        (None, "unknown"),
    ],
)
def test_weather_category(condition, expected):
    assert features.weather_category(condition) == expected


@pytest.mark.parametrize(
    ("street", "expected"),
    [
        ("I-95 N", "interstate"),
        ("Interstate 5 S", "interstate"),
        ("US-1", "us_route"),
        ("US Highway 41", "us_route"),
        ("CA-63", "state_route"),
        ("State Route 9", "state_route"),
        ("Pacific Coast Hwy", "other_highway"),
        ("Garden State Pkwy", "other_highway"),
        ("Main St", "local"),
        ("NE 5th Ave", "local"),
        ("West Street", "local"),
        (None, "local"),
    ],
)
def test_road_class(street, expected):
    assert features.road_class(street) == expected


def test_time_features_weekend_and_holiday():
    times = pd.Series(pd.to_datetime(["2022-07-04 08:15:00", "2022-07-09 23:59:00"]))
    out = features.time_features(times)
    assert out["is_holiday"].tolist() == [1, 0]  # Independence Day; a Saturday
    assert out["is_weekend"].tolist() == [0, 1]
    assert out["hour"].tolist() == [8, 23]


def test_grid_cell_floors_negative_longitudes():
    cells = features.grid_cell(pd.Series([35.1, 35.24]), pd.Series([-80.01, -80.24]))
    assert cells.tolist() == ["140_-321", "140_-321"]


def history_for(days, ys, prior=0.5, window=365, smoothing=0.0):
    days = pd.Series(pd.to_datetime(days))
    cells = pd.Series(["a"] * len(days))
    y = pd.Series(ys)
    daily = features.daily_cell_totals(cells, days, y)
    return features.history_at(daily, cells, days, prior, window, smoothing)


def test_history_uses_only_earlier_days():
    out = history_for(["2022-01-01", "2022-01-01", "2022-01-02", "2022-01-05"], [1, 0, 1, 0])
    # Same-day rows never see each other; later rows see every earlier day.
    assert out["cell_count"].tolist() == [0, 0, 2, 3]
    assert out["cell_rate"].iloc[2] == pytest.approx(0.5)
    assert out["cell_rate"].iloc[3] == pytest.approx(2 / 3)


def test_history_window_drops_old_days():
    out = history_for(["2022-01-01", "2022-01-10", "2022-01-12"], [1, 1, 0], window=5)
    assert out["cell_count"].tolist() == [0, 0, 1]


def test_history_rate_falls_back_to_prior_when_empty():
    out = history_for(["2022-01-01"], [1], prior=0.03, smoothing=20.0)
    assert out["cell_rate"].iloc[0] == pytest.approx(0.03)


def test_history_does_not_depend_on_row_order():
    days = ["2022-01-03", "2022-01-01", "2022-01-02", "2022-01-01"]
    ys = [0, 1, 1, 0]
    shuffled = history_for(days, ys, smoothing=20.0)
    expected = {"2022-01-01": 0, "2022-01-02": 2, "2022-01-03": 3}
    assert shuffled["cell_count"].tolist() == [expected[d] for d in days]
    assert np.isfinite(shuffled["cell_rate"]).all()


def test_history_accepts_query_days_at_another_timestamp_resolution():
    # Parquet gives microsecond timestamps; a date parsed from a string gives nanoseconds.
    history_days = pd.Series(pd.to_datetime(["2022-01-01", "2022-01-02"])).astype("datetime64[us]")
    cells = pd.Series(["a", "a"])
    daily = features.daily_cell_totals(cells, history_days, pd.Series([1, 0]))
    query = pd.Series([pd.Timestamp("2022-01-05")]).astype("datetime64[ns]")
    out = features.history_at(daily, pd.Series(["a"]), query, prior_rate=0.5, smoothing=0.0)
    assert out["cell_count"].tolist() == [2]
