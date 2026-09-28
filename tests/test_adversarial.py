import joblib
import numpy as np
import pytest

model = joblib.load("src/model.joblib")
n_features = model.n_features_in_

def test_handles_all_zero_input():
    prediction = model.predict(np.zeros((1, n_features)))
    assert prediction[0] in model.classes_

def test_handles_extreme_values_without_crashing():
    extreme = np.full((1, n_features), 1e6)
    prediction = model.predict(extreme)
    assert prediction[0] in model.classes_

def test_rejects_wrong_shape_gracefully():
    with pytest.raises(ValueError):
        model.predict(np.zeros((1, n_features - 1)))