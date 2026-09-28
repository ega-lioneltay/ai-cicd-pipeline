import time
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

def test_model_meets_accuracy_and_latency_targets():
    model = joblib.load("src/model.joblib")
    df = pd.read_csv("data/processed.csv")
    X, y = df.drop(columns=["target"]), df["target"]

    start = time.time()
    predictions = model.predict(X)
    latency_ms = (time.time() - start) / len(X) * 1000

    assert accuracy_score(y, predictions) >= 0.85
    assert latency_ms < 50, f"Per-prediction latency {latency_ms:.2f}ms exceeds 50ms budget"