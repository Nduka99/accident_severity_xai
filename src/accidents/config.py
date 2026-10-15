"""Every fixed choice in one place: paths, the target definition and split dates.

The dates come from the forensic analysis in notebooks 02 and 03: the Source1 feed's
labelling is stable from October 2020, and later months test the model as it would be used.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "raw" / "US_Accidents_March23.csv"
RAW_PARQUET = ROOT / "data" / "processed" / "accidents_raw.parquet"
MODEL_TABLE = ROOT / "data" / "processed" / "model_table.parquet"

# D4: only Source1 is still reporting in 2023, and from 2020-Q4 on it labels every
# record Severity 2 or 4. Earlier Source1 data and the other sources use other scales.
SOURCE = "Source1"
REGIME_START = "2020-10-01"
DATA_END = "2023-04-01"  # exclusive; the dataset ends on 2023-03-31

# The first quarter of the regime only feeds the past-only location history, so that
# no training row has an empty history window.
TRAIN_START = "2021-01-01"

# D5: temporal split; each start date is inclusive.
VALID_START = "2022-04-01"
TEST_START = "2022-10-01"

# The last two training months decide when boosting and neural training stop, so the
# validation months are used only to compare and choose models.
STOP_START = "2022-02-01"

# Location history: square grid cells of this many degrees, over this many past days.
GRID_DEGREES = 0.25
HISTORY_DAYS = 365
# Pseudo-count that pulls a cell's past Severity-4 rate towards the overall rate.
HISTORY_SMOOTHING = 20.0

SEED = 42
