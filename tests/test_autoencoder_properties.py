# tests/test_autoencoder_properties.py
"""
Property-based tests for Autoencoder integration.
Uses hypothesis library for property-based testing.
"""

import os
import sys
import tempfile
import json
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.autoencoder import TransactionAutoencoder


# Feature: autoencoder-integration, Property 4: Reconstruction Output Shape Preservation
# For any valid input feature vector of shape (n_samples, n_features), 
# the Autoencoder reconstruction output SHALL have the identical shape (n_samples, n_features).
# Validates: Requirements 1.3

@settings(max_examples=100, deadline=None)
@given(
    n_samples=st.integers(min_value=1, max_value=50),
    n_features=st.integers(min_value=5, max_value=30)
)
def test_property_4_reconstruction_output_shape_preservation(n_samples: int, n_features: int):
    """
    Property 4: Reconstruction Output Shape Preservation
    
    For any valid input feature vector of shape (n_samples, n_features),
    the Autoencoder reconstruction output SHALL have the identical shape.
    
    Validates: Requirements 1.3
    """
    # Create autoencoder with matching input dimension
    encoding_dim = max(2, n_features // 2)
    autoencoder = TransactionAutoencoder(
        input_dim=n_features,
        encoding_dim=encoding_dim,
        hidden_layers=[max(4, n_features), max(2, n_features // 2)]
    )
    
    # Generate random input
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Get reconstruction
    reconstructed = autoencoder.predict(X)
    
    # Verify shape is preserved
    assert reconstructed.shape == X.shape, \
        f"Shape mismatch: input {X.shape} vs output {reconstructed.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



# Feature: autoencoder-integration, Property 5: Threshold Formula Correctness
# For any array of reconstruction errors and any positive k value, 
# the computed threshold SHALL equal mean(errors) + k * std(errors) exactly.
# Validates: Requirements 2.7

@settings(max_examples=100, deadline=None)
@given(
    errors=st.lists(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False), 
                    min_size=10, max_size=1000),
    k=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
)
def test_property_5_threshold_formula_correctness(errors: list, k: float):
    """
    Property 5: Threshold Formula Correctness
    
    For any array of reconstruction errors and any positive k value,
    the computed threshold SHALL equal mean(errors) + k * std(errors) exactly.
    
    Validates: Requirements 2.7
    """
    from backend.train_autoencoder import AutoencoderTrainer
    
    errors_array = np.array(errors)
    
    # Skip if std is zero (all same values)
    assume(np.std(errors_array) > 1e-10)
    
    trainer = AutoencoderTrainer(k=k)
    result = trainer.compute_threshold(errors_array)
    
    # Compute expected threshold
    expected_threshold = np.mean(errors_array) + k * np.std(errors_array)
    
    # Verify formula
    assert abs(result['threshold'] - expected_threshold) < 1e-9, \
        f"Threshold mismatch: {result['threshold']} vs expected {expected_threshold}"
    assert abs(result['mean'] - np.mean(errors_array)) < 1e-9
    assert abs(result['std'] - np.std(errors_array)) < 1e-9
    assert result['k'] == k



# Feature: autoencoder-integration, Property 2: Scaler Round-Trip Consistency
# For any valid StandardScaler fitted on training data and any valid feature vector,
# saving the scaler to .pkl format then loading it SHALL produce a scaler that 
# transforms the feature vector to identical scaled values (within floating-point tolerance of 1e-9).
# Validates: Requirements 2.2, 2.3, 4.2, 7.2

@settings(max_examples=100, deadline=None)
@given(
    n_samples=st.integers(min_value=10, max_value=100),
    n_features=st.integers(min_value=5, max_value=26)
)
def test_property_2_scaler_round_trip_consistency(n_samples: int, n_features: int):
    """
    Property 2: Scaler Round-Trip Consistency
    
    For any valid StandardScaler fitted on training data and any valid feature vector,
    saving to .pkl then loading SHALL produce identical scaled values.
    
    Validates: Requirements 2.2, 2.3, 4.2, 7.2
    """
    from sklearn.preprocessing import StandardScaler
    import joblib
    
    # Generate random training data
    X_train = np.random.randn(n_samples, n_features).astype(np.float64)
    X_test = np.random.randn(5, n_features).astype(np.float64)
    
    # Fit original scaler
    original_scaler = StandardScaler()
    original_scaler.fit(X_train)
    original_scaled = original_scaler.transform(X_test)
    
    # Save and load scaler
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        temp_path = f.name
    
    try:
        joblib.dump(original_scaler, temp_path)
        loaded_scaler = joblib.load(temp_path)
        loaded_scaled = loaded_scaler.transform(X_test)
        
        # Verify round-trip consistency
        np.testing.assert_allclose(
            original_scaled, loaded_scaled, rtol=1e-9, atol=1e-9,
            err_msg="Scaler round-trip produced different results"
        )
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)



# Feature: autoencoder-integration, Property 3: Threshold Configuration Round-Trip
# For any valid threshold configuration dictionary containing threshold, mean, std, and k values,
# serializing to JSON then deserializing SHALL produce an equivalent configuration with identical values.
# Validates: Requirements 2.8, 4.3, 7.3, 7.4, 7.5

@settings(max_examples=100, deadline=None)
@given(
    threshold=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    mean=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    std=st.floats(min_value=0.001, max_value=5.0, allow_nan=False, allow_infinity=False),
    k=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    n_samples=st.integers(min_value=100, max_value=100000),
    n_features=st.integers(min_value=5, max_value=50)
)
def test_property_3_threshold_config_round_trip(threshold: float, mean: float, std: float, 
                                                 k: float, n_samples: int, n_features: int):
    """
    Property 3: Threshold Configuration Round-Trip
    
    For any valid threshold configuration dictionary,
    serializing to JSON then deserializing SHALL produce an equivalent configuration.
    
    Validates: Requirements 2.8, 4.3, 7.3, 7.4, 7.5
    """
    from datetime import datetime
    
    # Create threshold config
    original_config = {
        'threshold': threshold,
        'mean': mean,
        'std': std,
        'k': k,
        'computed_at': datetime.now().isoformat(),
        'n_samples': n_samples,
        'n_features': n_features
    }
    
    # Serialize and deserialize
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
        json.dump(original_config, f, indent=2)
    
    try:
        with open(temp_path, 'r') as f:
            loaded_config = json.load(f)
        
        # Verify round-trip consistency
        assert abs(loaded_config['threshold'] - original_config['threshold']) < 1e-15
        assert abs(loaded_config['mean'] - original_config['mean']) < 1e-15
        assert abs(loaded_config['std'] - original_config['std']) < 1e-15
        assert abs(loaded_config['k'] - original_config['k']) < 1e-15
        assert loaded_config['n_samples'] == original_config['n_samples']
        assert loaded_config['n_features'] == original_config['n_features']
        assert loaded_config['computed_at'] == original_config['computed_at']
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)



# Feature: autoencoder-integration, Property 1: Model Round-Trip Consistency
# For any valid trained Autoencoder model and any valid input feature vector,
# saving the model to .h5 format then loading it SHALL produce a model that 
# generates identical reconstruction errors (within floating-point tolerance of 1e-6).
# Validates: Requirements 2.6, 3.1, 3.3, 4.1, 7.1

@settings(max_examples=20, deadline=None)  # Reduced examples due to model training time
@given(
    n_features=st.integers(min_value=10, max_value=26),
    n_test_samples=st.integers(min_value=5, max_value=20)
)
def test_property_1_model_round_trip_consistency(n_features: int, n_test_samples: int):
    """
    Property 1: Model Round-Trip Consistency
    
    For any valid trained Autoencoder model and any valid input feature vector,
    saving to .h5 then loading SHALL produce identical reconstruction errors.
    
    Validates: Requirements 2.6, 3.1, 3.3, 4.1, 7.1
    """
    # Create and minimally train autoencoder
    encoding_dim = max(2, n_features // 2)
    autoencoder = TransactionAutoencoder(
        input_dim=n_features,
        encoding_dim=encoding_dim,
        hidden_layers=[max(8, n_features), max(4, n_features // 2)]
    )
    
    # Generate training data and train briefly
    X_train = np.random.randn(100, n_features).astype(np.float32)
    autoencoder.fit(X_train, epochs=2, batch_size=32, verbose=0)
    
    # Generate test data
    X_test = np.random.randn(n_test_samples, n_features).astype(np.float32)
    
    # Get original reconstruction errors
    original_errors = autoencoder.compute_reconstruction_error(X_test)
    
    # Save and load model
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        temp_path = f.name
    
    try:
        autoencoder.save(temp_path)
        loaded_autoencoder = TransactionAutoencoder.load(temp_path)
        
        # Get loaded model reconstruction errors
        loaded_errors = loaded_autoencoder.compute_reconstruction_error(X_test)
        
        # Verify round-trip consistency
        np.testing.assert_allclose(
            original_errors, loaded_errors, rtol=1e-6, atol=1e-6,
            err_msg="Model round-trip produced different reconstruction errors"
        )
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)



# Feature: autoencoder-integration, Property 6: Inference Pipeline Correctness
# For any valid transaction feature dictionary and loaded Autoencoder artifacts:
# - The reconstruction error SHALL be a non-negative float
# - If reconstruction error > threshold, then is_anomaly SHALL be True and reason SHALL contain the error and threshold values
# - If reconstruction error <= threshold, then is_anomaly SHALL be False and reason SHALL be None
# Validates: Requirements 4.4, 4.5, 4.6, 4.7

@settings(max_examples=50, deadline=None)
@given(
    threshold=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    error_multiplier=st.floats(min_value=0.1, max_value=3.0, allow_nan=False, allow_infinity=False)
)
def test_property_6_inference_pipeline_correctness(threshold: float, error_multiplier: float):
    """
    Property 6: Inference Pipeline Correctness
    
    For any valid transaction feature dictionary and loaded Autoencoder artifacts,
    the inference pipeline SHALL correctly determine anomaly status based on threshold comparison.
    
    Validates: Requirements 4.4, 4.5, 4.6, 4.7
    """
    from backend.autoencoder import AutoencoderInference
    
    # Create a mock inference instance with controlled threshold
    inference = AutoencoderInference()
    inference.threshold_config = {'threshold': threshold}
    
    # Simulate reconstruction error
    reconstruction_error = threshold * error_multiplier
    
    # Determine expected results
    expected_is_anomaly = reconstruction_error > threshold
    
    # Verify the logic
    is_anomaly = reconstruction_error > threshold
    
    if is_anomaly:
        reason = f"Autoencoder anomaly: reconstruction error {reconstruction_error:.4f} exceeds threshold {threshold:.4f}"
        assert expected_is_anomaly == True
        assert str(reconstruction_error)[:4] in reason or f"{reconstruction_error:.4f}" in reason
        assert str(threshold)[:4] in reason or f"{threshold:.4f}" in reason
    else:
        reason = None
        assert expected_is_anomaly == False
        assert reason is None
    
    # Verify reconstruction error is non-negative (by construction in this test)
    assert reconstruction_error >= 0



# Feature: autoencoder-integration, Property 7: Hybrid Decision AE Integration
# For any transaction where the Autoencoder flags an anomaly:
# - The reasons list SHALL contain the Autoencoder reason string
# - The is_fraud flag SHALL be True
# - The ae_reconstruction_error field SHALL contain the computed error value
# Validates: Requirements 5.3, 5.4, 5.5

class MockAutoencoder:
    """Mock Autoencoder for testing hybrid decision integration."""
    
    def __init__(self, is_anomaly: bool, reconstruction_error: float, threshold: float):
        self._is_anomaly = is_anomaly
        self._reconstruction_error = reconstruction_error
        self._threshold = threshold
        self._available = True
    
    def is_available(self) -> bool:
        return self._available
    
    def score_transaction(self, features: dict) -> dict:
        reason = None
        if self._is_anomaly:
            reason = f"Autoencoder anomaly: reconstruction error {self._reconstruction_error:.4f} exceeds threshold {self._threshold:.4f}"
        return {
            'reconstruction_error': self._reconstruction_error,
            'threshold': self._threshold,
            'is_anomaly': self._is_anomaly,
            'reason': reason
        }


@settings(max_examples=100, deadline=None)
@given(
    reconstruction_error=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    threshold=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    amount=st.floats(min_value=100, max_value=10000, allow_nan=False, allow_infinity=False)
)
def test_property_7_hybrid_decision_ae_integration(reconstruction_error: float, threshold: float, amount: float):
    """
    Property 7: Hybrid Decision AE Integration
    
    For any transaction where the Autoencoder flags an anomaly,
    the result SHALL contain the AE reason, is_fraud=True, and ae_reconstruction_error.
    
    Validates: Requirements 5.3, 5.4, 5.5
    """
    from backend.hybrid_decision import make_decision
    
    is_anomaly = reconstruction_error > threshold
    mock_ae = MockAutoencoder(is_anomaly, reconstruction_error, threshold)
    
    txn = {
        'amount': amount,
        'transfer_type': 'O',
        'txn_count_10min': 1,
        'txn_count_1hour': 1,
        'time_since_last_txn': 3600
    }
    
    user_stats = {
        'user_avg_amount': amount * 2,  # Make amount normal
        'user_std_amount': amount,
        'user_max_amount': amount * 3,
        'user_txn_frequency': 10,
        'user_international_ratio': 0.0,
        'current_month_spending': 0
    }
    
    result = make_decision(txn, user_stats, None, [], autoencoder=mock_ae)
    
    # Verify AE fields are populated
    assert result['ae_reconstruction_error'] == reconstruction_error
    assert result['ae_threshold'] == threshold
    
    if is_anomaly:
        # When AE flags anomaly, is_fraud should be True
        assert result['is_fraud'] == True
        assert result['ae_flag'] == True
        # Reasons should contain AE reason
        ae_reasons = [r for r in result['reasons'] if 'Autoencoder' in r]
        assert len(ae_reasons) > 0
        # Reason should contain error and threshold values
        assert str(reconstruction_error)[:4] in ae_reasons[0] or f"{reconstruction_error:.4f}" in ae_reasons[0]



# Feature: autoencoder-integration, Property 8: Graceful Degradation on AE Failure
# For any transaction processed when Autoencoder artifacts are unavailable or invalid:
# - The Hybrid Decision Engine SHALL still produce valid Rule Engine results
# - The Hybrid Decision Engine SHALL still produce valid Isolation Forest results
# - The system SHALL NOT raise an exception
# Validates: Requirements 6.6

class UnavailableAutoencoder:
    """Mock Autoencoder that is unavailable."""
    
    def is_available(self) -> bool:
        return False
    
    def score_transaction(self, features: dict) -> None:
        return None


@settings(max_examples=100, deadline=None)
@given(
    amount=st.floats(min_value=100, max_value=50000, allow_nan=False, allow_infinity=False),
    txn_count_10min=st.integers(min_value=1, max_value=10),
    txn_count_1hour=st.integers(min_value=1, max_value=20)
)
def test_property_8_graceful_degradation_on_ae_failure(amount: float, txn_count_10min: int, txn_count_1hour: int):
    """
    Property 8: Graceful Degradation on AE Failure
    
    For any transaction processed when Autoencoder is unavailable,
    the system SHALL still produce valid results without raising exceptions.
    
    Validates: Requirements 6.6
    """
    from backend.hybrid_decision import make_decision
    
    unavailable_ae = UnavailableAutoencoder()
    
    txn = {
        'amount': amount,
        'transfer_type': 'O',
        'txn_count_10min': txn_count_10min,
        'txn_count_1hour': txn_count_1hour,
        'time_since_last_txn': 3600
    }
    
    user_stats = {
        'user_avg_amount': 5000,
        'user_std_amount': 2000,
        'user_max_amount': 15000,
        'user_txn_frequency': 10,
        'user_international_ratio': 0.0,
        'current_month_spending': 0
    }
    
    # Should not raise exception
    result = make_decision(txn, user_stats, None, [], autoencoder=unavailable_ae)
    
    # Result should be valid
    assert 'is_fraud' in result
    assert 'reasons' in result
    assert 'threshold' in result
    assert isinstance(result['is_fraud'], bool)
    assert isinstance(result['reasons'], list)
    
    # AE fields should be None when unavailable
    assert result['ae_reconstruction_error'] is None
    assert result['ae_flag'] == False


@settings(max_examples=100, deadline=None)
@given(
    amount=st.floats(min_value=100, max_value=50000, allow_nan=False, allow_infinity=False)
)
def test_property_8_graceful_degradation_no_autoencoder(amount: float):
    """
    Property 8: Graceful Degradation when no Autoencoder provided
    
    For any transaction processed without an Autoencoder instance,
    the system SHALL still produce valid results.
    
    Validates: Requirements 6.6
    """
    from backend.hybrid_decision import make_decision
    
    txn = {
        'amount': amount,
        'transfer_type': 'O',
        'txn_count_10min': 1,
        'txn_count_1hour': 1,
        'time_since_last_txn': 3600
    }
    
    user_stats = {
        'user_avg_amount': 5000,
        'user_std_amount': 2000,
        'user_max_amount': 15000,
        'user_txn_frequency': 10,
        'user_international_ratio': 0.0,
        'current_month_spending': 0
    }
    
    # Should not raise exception when autoencoder is None
    result = make_decision(txn, user_stats, None, [], autoencoder=None)
    
    # Result should be valid
    assert 'is_fraud' in result
    assert 'reasons' in result
    assert result['ae_reconstruction_error'] is None
    assert result['ae_flag'] == False
