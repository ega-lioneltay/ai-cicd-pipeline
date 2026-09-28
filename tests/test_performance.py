import time
import pytest
from sklearn.metrics import accuracy_score

def test_model_accuracy_gate(trained_model, processed_data):
    X, y = processed_data
    predictions = trained_model.predict(X)
    assert accuracy_score(y, predictions) >= 0.85

@pytest.mark.benchmark
def test_single_sample_latency_budget(trained_model, processed_data):
    X, _ = processed_data
    sample = X.iloc[[0]]

    # Warmup
    for _ in range(5):
        _ = trained_model.predict(sample)

    iterations = 50
    start = time.perf_counter()
    for _ in range(iterations):
        _ = trained_model.predict(sample)
    avg_latency_ms = ((time.perf_counter() - start) / iterations) * 1000

    assert avg_latency_ms < 50, f"Single-prediction latency {avg_latency_ms:.2f}ms exceeds 50ms"
