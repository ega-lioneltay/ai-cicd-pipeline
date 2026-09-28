import pandas as pd
from src.data_pipeline import validate

def test_validate_accepts_clean_data():
    df = pd.DataFrame({
        "a": [1.0] * 30 + [2.0] * 30,
        "target": [0] * 30 + [1] * 30,
    })
    assert validate(df) is True

def test_validate_rejects_nulls():
    df = pd.DataFrame({"a": [1.0, None], "target": [0, 1]})
    assert validate(df) is False

def test_validate_rejects_unknown_class():
    df = pd.DataFrame({"a": range(60), "target": [9] * 60})
    assert validate(df) is False
