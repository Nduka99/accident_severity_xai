"""Temporal split (decision D5): the model is judged on months it has never seen."""

import numpy as np
import pandas as pd

from accidents import config

SPLITS = ["burn_in", "train", "valid", "test"]


def assign_split(start_time: pd.Series) -> pd.Series:
    """Label each row burn_in, train, valid or test from its start time.

    burn_in rows only feed the location history; they are never fitted on.
    """
    conditions = [
        start_time < pd.Timestamp(config.TRAIN_START),
        start_time < pd.Timestamp(config.VALID_START),
        start_time < pd.Timestamp(config.TEST_START),
    ]
    labels = np.select(conditions, SPLITS[:3], default="test")
    return pd.Series(labels, index=start_time.index, name="split")
