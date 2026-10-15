"""Read the raw CSV once, check it against a schema contract, and store it as Parquet."""

from pathlib import Path

import pandas as pd

from accidents import config

# Columns the pipeline relies on. Others in the file are kept but not checked.
REQUIRED_COLUMNS = [
    "ID", "Source", "Severity", "Start_Time", "End_Time", "Start_Lat", "Start_Lng",
    "Distance(mi)", "Description", "Street", "City", "County", "State", "Timezone",
    "Temperature(F)", "Wind_Chill(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)",
    "Wind_Direction", "Wind_Speed(mph)", "Precipitation(in)", "Weather_Condition",
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway",
    "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal", "Turning_Loop",
    "Sunrise_Sunset",
]  # fmt: skip

# Bounding box of the contiguous United States, with a small margin.
LAT_RANGE = (24.0, 50.0)
LNG_RANGE = (-125.0, -66.0)


def read_raw_csv(path: Path = config.RAW_CSV) -> pd.DataFrame:
    """The raw file as published. pyarrow parses the timestamps and booleans itself."""
    return pd.read_csv(path, engine="pyarrow")


def schema_problems(df: pd.DataFrame) -> list[str]:
    """Every way the table breaks the contract the rest of the pipeline assumes.

    An empty list means the table can be used. Out-of-range weather readings are not
    contract breaks; `clean.mask_impossible_readings` handles them.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]

    problems = []
    if not pd.api.types.is_datetime64_any_dtype(df["Start_Time"]):
        problems.append("Start_Time is not parsed as a timestamp")
    elif df["Start_Time"].isna().any():
        problems.append(f"{df['Start_Time'].isna().sum()} rows have no Start_Time")
    if df["ID"].duplicated().any():
        problems.append(f"{df['ID'].duplicated().sum()} duplicate IDs")
    if not df["Severity"].isin([1, 2, 3, 4]).all():
        problems.append("Severity has values outside 1-4")
    if not df["Start_Lat"].between(*LAT_RANGE).all():
        problems.append("Start_Lat outside the contiguous US")
    if not df["Start_Lng"].between(*LNG_RANGE).all():
        problems.append("Start_Lng outside the contiguous US")
    return problems


def write_parquet(df: pd.DataFrame, path: Path = config.RAW_PARQUET) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


# Row filter for the target population's source and period (decision D4), applied while
# reading so the other ~4.5 million records never enter memory.
POPULATION_ROWS = [
    ("Source", "==", config.SOURCE),
    ("Start_Time", ">=", pd.Timestamp(config.REGIME_START)),
    ("Start_Time", "<", pd.Timestamp(config.DATA_END)),
]

# Raw columns the population needs for features, labels and the leakage audit.
POPULATION_COLUMNS = [
    "ID", "Source", "Severity", "Start_Time", "End_Time", "Start_Lat", "Start_Lng",
    "Distance(mi)", "Description", "Street", "State", "Weather_Condition", "Wind_Direction",
    "Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)",
    "Precipitation(in)", "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit",
    "Railway", "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal",
]  # fmt: skip


def read_parquet(
    columns: list[str] | None = None,
    rows: list[tuple] | None = None,
    path: Path = config.RAW_PARQUET,
) -> pd.DataFrame:
    """Only the columns and rows a step needs; the full table takes several GB of memory."""
    return pd.read_parquet(path, columns=columns, filters=rows)
