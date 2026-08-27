import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def rank_feature_importance(X_full, y, random_state=42, n_estimators=300):
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_full, y)
    return pd.Series(rf.feature_importances_, index=X_full.columns).sort_values(ascending=False)


def select_top_features(importances, n_qubits):
    return importances.index[:n_qubits].tolist()
