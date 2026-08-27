from src import config


def test_column_names_shape():
    assert config.COLUMN_NAMES[0] == "target"
    assert len(config.COLUMN_NAMES) == 21


def test_n_qubits_within_recommended_budget():
    assert 1 <= config.N_QUBITS <= 6
