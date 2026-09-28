from src.data_pipeline import validate
import pandas as pd

def test_validate_rejects_nulls():
    df = pd.DataFrame({"a": [1, None], "target": [0, 1]})
    assert validate(df) is False

def test_validate_rejects_unknown_class():
    df = pd.DataFrame({"a": range(60), "target": [9] * 60})
    assert validate(df) is False