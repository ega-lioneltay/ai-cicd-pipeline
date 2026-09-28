import numpy as np
import pytest

def test_handles_boundary_values(trained_model):
    n_features = trained_model.n_features_in_
    
    zero_input = np.zeros((1, n_features))
    extreme_input = np.full((1, n_features), 1e6)

    assert trained_model.predict(zero_input)[0] in trained_model.classes_
    assert trained_model.predict(extreme_input)[0] in trained_model.classes_

def test_rejects_wrong_shape_gracefully(trained_model):
    n_features = trained_model.n_features_in_
    with pytest.raises(ValueError):
        trained_model.predict(np.zeros((1, n_features - 1)))

@pytest.mark.parametrize("corrupt_val", [np.nan, np.inf, -np.inf])
def test_rejects_or_handles_nan_and_inf(trained_model, corrupt_val):
    n_features = trained_model.n_features_in_
    corrupt_input = np.full((1, n_features), corrupt_val)

    try:
        prediction = trained_model.predict(corrupt_input)
        # If the model natively handles missing/infinite values, check that output is valid
        assert prediction[0] in trained_model.classes_
    except (ValueError, TypeError):
        # Graceful rejection is also valid behavior
        pass
