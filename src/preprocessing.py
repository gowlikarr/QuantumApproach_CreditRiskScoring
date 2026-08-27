import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def scale_features(X_selected, encoding_range):
    scaler = MinMaxScaler(feature_range=encoding_range)
    return scaler.fit_transform(X_selected), scaler


def stratified_subsample(X_scaled, y, n_per_class, random_state=42):
    rng = np.random.RandomState(random_state)
    idx_good = np.where(y == 1)[0]
    idx_bad = np.where(y == 0)[0]
    sel_idx = np.concatenate([
        rng.choice(idx_good, n_per_class, replace=False),
        rng.choice(idx_bad, n_per_class, replace=False),
    ])
    return X_scaled[sel_idx], y[sel_idx]


def split_data(X_sub, y_sub, test_size=0.30, random_state=42):
    return train_test_split(
        X_sub, y_sub, test_size=test_size, random_state=random_state, stratify=y_sub
    )
