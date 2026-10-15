import numpy as np
import pandas as pd

from accidents import clean, ingest, split


def raw_rows():
    return pd.DataFrame(
        {
            "Source": ["Source1", "Source1", "Source2", "Source1", "Source1"],
            "Severity": [2, 4, 4, 3, 4],
            "Start_Time": pd.to_datetime(
                ["2021-05-01", "2020-12-01", "2021-05-01", "2021-05-01", "2019-01-01"]
            ),
            "Temperature(F)": [207.0, 50.0, 60.0, 70.0, 80.0],
            "Humidity(%)": [50.0] * 5,
            "Pressure(in)": [0.0, 29.9, 29.9, 29.9, 29.9],
            "Visibility(mi)": [10.0] * 5,
            "Wind_Speed(mph)": [1087.0, 5.0, 5.0, 5.0, 5.0],
            "Precipitation(in)": [np.nan, 0.0, 0.0, 0.0, 0.0],
        }
    )


def test_target_window_keeps_source1_regime_severity_2_and_4():
    out = clean.target_window(raw_rows())
    # Source2, Severity 3 and the pre-regime 2019 row are gone; sorted by time.
    assert out["Start_Time"].tolist() == list(pd.to_datetime(["2020-12-01", "2021-05-01"]))
    assert out["y"].tolist() == [1, 0]


def test_impossible_readings_become_missing():
    out = clean.mask_impossible_readings(raw_rows())
    assert pd.isna(out.loc[0, "Temperature(F)"])
    assert pd.isna(out.loc[0, "Pressure(in)"])
    assert pd.isna(out.loc[0, "Wind_Speed(mph)"])
    assert out.loc[1, "Temperature(F)"] == 50.0
    assert pd.isna(out.loc[0, "Precipitation(in)"])  # missing stays missing


def test_split_boundaries_are_inclusive_starts_without_overlap():
    times = pd.Series(
        pd.to_datetime(
            ["2020-12-31 23:59", "2021-01-01 00:00", "2022-03-31 23:59", "2022-04-01 00:00",
             "2022-09-30 23:59", "2022-10-01 00:00", "2023-03-31 00:00"]
        )
    )  # fmt: skip
    labels = split.assign_split(times).tolist()
    assert labels == ["burn_in", "train", "train", "valid", "valid", "test", "test"]


def test_schema_problems_flags_duplicates_and_bad_severity():
    df = pd.DataFrame({column: [0, 0] for column in ingest.REQUIRED_COLUMNS})
    df["ID"] = ["A-1", "A-1"]
    df["Severity"] = [2, 5]
    df["Start_Time"] = pd.to_datetime(["2021-01-01", "2021-01-02"])
    df["Start_Lat"] = [35.0, 35.0]
    df["Start_Lng"] = [-80.0, -80.0]
    problems = ingest.schema_problems(df)
    assert any("duplicate IDs" in p for p in problems)
    assert any("Severity" in p for p in problems)
    assert ingest.schema_problems(df.drop(columns=["Street"])) == ["missing columns: ['Street']"]
