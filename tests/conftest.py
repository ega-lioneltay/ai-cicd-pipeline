import joblib
import pandas as pd
import pytest

@pytest.fixture(scope="session")
def trained_model():
    return joblib.load("src/model.joblib")

@pytest.fixture(scope="session")
def processed_data():
    df = pd.read_csv("data/processed.csv")
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y
