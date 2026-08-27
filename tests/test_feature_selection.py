import numpy as np
import pandas as pd

from src.feature_selection import rank_feature_importance, select_top_features


def test_rank_and_select_top_features():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.rand(100, 5), columns=[f"f{i}" for i in range(5)])
    y = (X["f0"] + rng.rand(100) * 0.1 > 0.5).astype(int).values

    importances = rank_feature_importance(X, y, random_state=0, n_estimators=50)
    assert set(importances.index) == set(X.columns)
    assert np.isclose(importances.sum(), 1.0, atol=1e-6)

    top = select_top_features(importances, n_qubits=3)
    assert len(top) == 3
    assert top[0] == importances.index[0]
