"""Features known when an accident is first reported (decision D6: no description text).

The dashboard re-implements these rules, so each one is a plain, deterministic function
of fields a person can enter: place, time, weather and road.
"""

import re

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from accidents import config

POI_COLUMNS = {
    "Amenity": "amenity", "Bump": "bump", "Crossing": "crossing", "Give_Way": "give_way",
    "Junction": "junction", "No_Exit": "no_exit", "Railway": "railway",
    "Roundabout": "roundabout", "Station": "station", "Stop": "stop",
    "Traffic_Calming": "traffic_calming", "Traffic_Signal": "traffic_signal",
}  # fmt: skip

WEATHER_COLUMNS = {
    "Temperature(F)": "temperature_f",
    "Humidity(%)": "humidity_pct",
    "Pressure(in)": "pressure_in",
    "Visibility(mi)": "visibility_mi",
    "Wind_Speed(mph)": "wind_speed_mph",
    "Precipitation(in)": "precipitation_in",
}

# Ordered: the first rule whose word appears wins, so frozen surfaces outrank storms,
# storms outrank rain, and so on. Covers all 107 conditions in the target window.
WEATHER_RULES = [
    ("snow_ice", ("snow", "sleet", "ice", "freezing", "wintry", "hail")),
    ("storm", ("thunder", "t-storm", "tornado", "squall", "funnel")),
    ("rain", ("rain", "drizzle", "shower")),
    ("fog_obscured", ("fog", "mist", "haze", "smoke", "dust", "sand")),
    ("cloudy", ("cloudy", "overcast")),
    ("clear", ("fair", "clear")),
]
WEATHER_CATEGORIES = [name for name, _ in WEATHER_RULES] + ["unknown"]

# Ordered like WEATHER_RULES. Patterns match the upper-cased street name; a two-letter
# state prefix needs a hyphen ("CA-63") so that "NE 5th Ave" stays local.
ROAD_RULES = [
    ("interstate", r"\b(?:I|IH|INTERSTATE)[- ]?\d+"),
    ("us_route", r"\b(?:US|U\.S\.)(?:[- ]?(?:HWY|HIGHWAY|ROUTE|RTE))?[- ]?\d+"),
    ("state_route", r"\b(?:SR|SH|STATE (?:ROUTE|RTE|HWY|HIGHWAY|ROAD|RD))[- ]?\d+|\b[A-Z]{2}-\d+"),
    (
        "other_highway",
        r"\b(?:HWY|HIGHWAY|FWY|FREEWAY|EXPY|EXPRESSWAY|TPKE|TURNPIKE|PKWY|PARKWAY"
        r"|TOLLWAY|BELTWAY|THRUWAY|BYPASS|ROUTE|RTE)\b",
    ),
]
ROAD_CLASSES = [name for name, _ in ROAD_RULES] + ["local"]

WIND_DIRECTIONS = [
    "CALM", "VAR", "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "unknown",
]  # fmt: skip

TIME_FEATURES = ["hour", "weekday", "month", "is_weekend", "is_holiday"]
HISTORY_FEATURES = ["cell_count", "cell_rate"]
NUMERIC_FEATURES = [
    "lat", "lng", *WEATHER_COLUMNS.values(), *TIME_FEATURES, *HISTORY_FEATURES,
    *POI_COLUMNS.values(),
]  # fmt: skip
CATEGORICAL_FEATURES = ["state", "road_class", "weather", "wind_direction"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def weather_category(condition: object) -> str:
    if not isinstance(condition, str):
        return "unknown"
    text = condition.lower()
    for category, words in WEATHER_RULES:
        if any(word in text for word in words):
            return category
    return "unknown"


def road_class(street: object) -> str:
    if not isinstance(street, str):
        return "local"
    text = street.upper()
    for category, pattern in ROAD_RULES:
        if re.search(pattern, text):
            return category
    return "local"


def map_unique(values: pd.Series, rule) -> pd.Series:
    """Apply a per-value rule once per distinct value; the columns repeat heavily."""
    lookup = {value: rule(value) for value in values.dropna().unique()}
    return values.map(lookup).fillna(rule(None))


def time_features(start_time: pd.Series) -> pd.DataFrame:
    """Calendar features in the accident's local time (Start_Time is local)."""
    # Normalise the start: a holiday is dated midnight, before the first accident's time.
    holidays = USFederalHolidayCalendar().holidays(
        start=start_time.min().normalize(), end=start_time.max()
    )
    weekday = start_time.dt.weekday
    return pd.DataFrame(
        {
            "hour": start_time.dt.hour.astype("int8"),
            "weekday": weekday.astype("int8"),
            "month": start_time.dt.month.astype("int8"),
            "is_weekend": (weekday >= 5).astype("int8"),
            "is_holiday": start_time.dt.normalize().isin(holidays).astype("int8"),
        },
        index=start_time.index,
    )


def grid_cell(lat: pd.Series, lng: pd.Series, degrees: float = config.GRID_DEGREES) -> pd.Series:
    """Square grid cell id such as "143_-353" (row and column of `degrees`-sized cells)."""
    # Floor division rounds towards minus infinity, so -80.01 lands in column -321.
    row = (lat // degrees).astype(int).astype(str)
    col = (lng // degrees).astype(int).astype(str)
    return row + "_" + col


def daily_cell_totals(cells: pd.Series, days: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Running totals of accidents and Severity-4 accidents per cell, one row per cell-day."""
    daily = (
        pd.DataFrame({"cell": cells.values, "day": days.values, "y": y.values})
        .groupby(["cell", "day"], sort=True)
        .agg(n=("y", "size"), n4=("y", "sum"))
        .reset_index()
    )
    daily["cum_n"] = daily.groupby("cell")["n"].cumsum()
    daily["cum_n4"] = daily.groupby("cell")["n4"].cumsum()
    return daily[["cell", "day", "cum_n", "cum_n4"]]


def totals_before(daily: pd.DataFrame, query: pd.DataFrame) -> pd.DataFrame:
    """Running totals over all days strictly before each query (cell, day)."""
    order = query.sort_values("day", kind="stable")
    merged = pd.merge_asof(
        order,
        daily.sort_values("day", kind="stable"),
        on="day",
        by="cell",
        allow_exact_matches=False,  # the same day is not yet history
        direction="backward",
    )
    merged.index = order.index
    return merged[["cum_n", "cum_n4"]].fillna(0).reindex(query.index)


def history_at(
    daily: pd.DataFrame,
    cells: pd.Series,
    days: pd.Series,
    prior_rate: float,
    window_days: int = config.HISTORY_DAYS,
    smoothing: float = config.HISTORY_SMOOTHING,
) -> pd.DataFrame:
    """Accident count and smoothed Severity-4 rate in each cell over the `window_days`
    days before each day. Uses past rows only, so it is known at report time."""
    # merge_asof needs both sides at one timestamp resolution; Parquet reads give
    # microseconds while dates built from strings give nanoseconds.
    query_days = days.astype(daily["day"].dtype)
    query = pd.DataFrame({"cell": cells.values, "day": query_days.values}, index=cells.index)
    before_day = totals_before(daily, query)
    window_start = query.assign(day=query["day"] - pd.Timedelta(days=window_days))
    before_window = totals_before(daily, window_start)
    count = before_day["cum_n"] - before_window["cum_n"]
    severe = before_day["cum_n4"] - before_window["cum_n4"]
    rate = (severe + smoothing * prior_rate) / (count + smoothing)
    return pd.DataFrame({"cell_count": count.astype("int32"), "cell_rate": rate}, index=cells.index)


def as_categories(values: pd.Series, categories: list[str]) -> pd.Series:
    """Fixed category lists keep the encoding identical between training and the dashboard."""
    return pd.Series(pd.Categorical(values, categories=categories), index=values.index)


def build_model_table(clean: pd.DataFrame, prior_rate: float) -> pd.DataFrame:
    """One row per accident: identifiers, label, and every model feature.

    `clean` must come from `clean.target_window` (sorted by time, with y). `prior_rate`
    must be computed from training rows only, so no later label shapes the smoothing.
    """
    start = clean["Start_Time"]
    days = start.dt.normalize()
    cells = grid_cell(clean["Start_Lat"], clean["Start_Lng"])
    daily = daily_cell_totals(cells, days, clean["y"])
    states = sorted(clean["State"].dropna().unique())
    table = pd.DataFrame(
        {
            "id": clean["ID"],
            "start_time": start,
            "y": clean["y"],
            "cell": cells,
            "lat": clean["Start_Lat"],
            "lng": clean["Start_Lng"],
            "state": as_categories(clean["State"], states),
            "road_class": as_categories(map_unique(clean["Street"], road_class), ROAD_CLASSES),
            "weather": as_categories(
                map_unique(clean["Weather_Condition"], weather_category), WEATHER_CATEGORIES
            ),
            "wind_direction": as_categories(
                clean["Wind_Direction"].fillna("unknown"), WIND_DIRECTIONS
            ),
        }
    )
    for raw, name in WEATHER_COLUMNS.items():
        table[name] = clean[raw]
    for raw, name in POI_COLUMNS.items():
        table[name] = clean[raw].astype("int8")
    table = table.join(time_features(start))
    table = table.join(history_at(daily, cells, days, prior_rate))
    return table
