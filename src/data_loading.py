import warnings

import pandas as pd

from .config import COLUMN_NAMES, DATA_PATH

UCI_DATASET_ID = 144
UCI_DATASET_URL = "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data"


def _fetch_from_uci():
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    X = dataset.data.features.reset_index(drop=True).copy()
    y = dataset.data.targets.reset_index(drop=True)

    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.factorize(X[col])[0]

    df = X
    df.columns = COLUMN_NAMES[1:]
    df.insert(0, "target", (y["class"] == 1).astype(int).values)
    return df


def load_credit_data(path=DATA_PATH, prefer_dynamic=True):
    if prefer_dynamic:
        try:
            return _fetch_from_uci()
        except Exception as exc:
            warnings.warn(
                f"Could not fetch dataset live from UCI ({UCI_DATASET_URL}): {exc}. "
                f"Falling back to the local copy at '{path}'."
            )
    return pd.read_csv(path, header=None, names=COLUMN_NAMES)
