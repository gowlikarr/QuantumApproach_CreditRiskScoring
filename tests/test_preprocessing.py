import numpy as np

from src.preprocessing import scale_features, split_data, stratified_subsample


def test_scale_features_respects_range():
    X = np.random.RandomState(0).rand(50, 3) * 100
    X_scaled, scaler = scale_features(X, (0, np.pi / 4))
    assert X_scaled.min() >= -1e-9
    assert X_scaled.max() <= np.pi / 4 + 1e-9


def test_stratified_subsample_balances_classes():
    y = np.array([1] * 300 + [0] * 700)
    X = np.arange(1000).reshape(-1, 1).astype(float)
    X_sub, y_sub = stratified_subsample(X, y, n_per_class=20, random_state=0)
    assert (y_sub == 1).sum() == 20
    assert (y_sub == 0).sum() == 20


def test_split_data_is_stratified():
    y = np.array([1] * 40 + [0] * 40)
    X = np.arange(80).reshape(-1, 1).astype(float)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25, random_state=0)
    assert len(X_test) == 20
    assert (y_test == 1).sum() == 10
