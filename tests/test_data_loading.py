from src.data_loading import load_credit_data


def test_load_local_fallback_shape():
    df = load_credit_data(prefer_dynamic=False)
    assert df.shape == (1000, 21)
    assert "target" in df.columns


def test_load_local_fallback_class_balance():
    df = load_credit_data(prefer_dynamic=False)
    counts = df["target"].value_counts().to_dict()
    assert counts[1] == 700
    assert counts[0] == 300
