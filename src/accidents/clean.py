"""Turn raw records into the modelling population: the target window, a binary label,
and physically impossible sensor readings set to missing."""

import pandas as pd

from accidents import config

# Readings outside these bounds are sensor or unit errors (the raw data holds 207 F,
# 1087 mph and 0 inHg) and become missing. Pressure is station pressure, so the lower
# bound allows for high-altitude stations.
PHYSICAL_BOUNDS = {
    "Temperature(F)": (-60.0, 130.0),
    "Humidity(%)": (0.0, 100.0),
    "Pressure(in)": (15.0, 32.0),
    "Visibility(mi)": (0.0, 50.0),
    "Wind_Speed(mph)": (0.0, 150.0),
    "Precipitation(in)": (0.0, 12.0),
}


def target_window(df: pd.DataFrame) -> pd.DataFrame:
    """Source1 rows from the stable labelling regime, with label y = 1 for Severity 4.

    In this regime Source1 labels records 2 or 4; the handful of 1s and 3s are dropped.
    """
    in_window = (
        (df["Source"] == config.SOURCE)
        & (df["Start_Time"] >= config.REGIME_START)
        & (df["Start_Time"] < config.DATA_END)
    )
    labelled = df["Severity"].isin([2, 4])
    out = df.loc[in_window & labelled].copy()
    out["y"] = (out["Severity"] == 4).astype("int8")
    return out.sort_values("Start_Time", kind="stable").reset_index(drop=True)


def mask_impossible_readings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column, (low, high) in PHYSICAL_BOUNDS.items():
        out[column] = out[column].where(out[column].between(low, high))
    return out
