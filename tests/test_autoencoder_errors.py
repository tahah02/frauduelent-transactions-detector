# tests/test_autoencoder_errors.py
"""
Unit tests for Autoencoder error handling edge cases.
Tests missing files, invalid inputs, and error conditions.
"""

import os
import sys
import tempfile
import json
import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.autoencoder import AutoencoderInference


class TestMissingFiles:
    """Tests for missing artifact file scenarios."""
    
    def test_missing_model_file(self, tmp_path, monkeypatch):
        """Test handling when model file is missing."""
        # Point to non-existent paths
        monkeypatch.setattr(AutoencoderInference, 'MODEL_PATH', str(tmp_path / 'nonexistent.h5'))
        monkeypatch.setattr(AutoencoderInference, 'SCALER_PATH', str(tmp_path / 'scaler.pkl'))
        monkeypatch.setattr(AutoencoderInference, 'THRESHOLD_PATH', str(tmp_path / 'threshold.json'))
        
        inference = AutoencoderInference()
        result = inference.load_artifacts()
        
        assert result == False
        assert inference.is_available() == False
    
    def test_missing_scaler_file(self, tmp_path, monkeypatch):
        """Test handling when scaler file is missing."""
        # Create model file but not scaler
        model_path = tmp_path / 'model.h5'
        model_path.touch()
        
        monkeypatch.setattr(AutoencoderInference, 'MODEL_PATH', str(model_path))
        monkeypatch.setattr(AutoencoderInference, 'SCALER_PATH', str(tmp_path / 'nonexistent.pkl'))
        monkeypatch.setattr(AutoencoderInference, 'THRESHOLD_PATH', str(tmp_path / 'threshold.json'))
        
        inference = AutoencoderInference()
        result = inference.load_artifacts()
        
        assert result == False
        assert inference.is_available() == False
    
    def test_missing_threshold_file(self, tmp_path, monkeypatch):
        """Test handling when threshold file is missing."""
        # Create model and scaler files but not threshold
        model_path = tmp_path / 'model.h5'
        scaler_path = tmp_path / 'scaler.pkl'
        model_path.touch()
        scaler_path.touch()
        
        monkeypatch.setattr(AutoencoderInference, 'MODEL_PATH', str(model_path))
        monkeypatch.setattr(AutoencoderInference, 'SCALER_PATH', str(scaler_path))
        monkeypatch.setattr(AutoencoderInference, 'THRESHOLD_PATH', str(tmp_path / 'nonexistent.json'))
        
        inference = AutoencoderInference()
        result = inference.load_artifacts()
        
        assert result == False
        assert inference.is_available() == False


class TestFeatureValidation:
    """Tests for feature validation scenarios."""
    
    def test_missing_features(self):
        """Test handling when input features are missing."""
        inference = AutoencoderInference()
        inference._loaded = True  # Pretend loaded
        
        # Incomplete features
        incomplete_features = {
            'transaction_amount': 1000,
            'hour': 12
            # Missing many required features
        }
        
        result = inference._validate_features(incomplete_features)
        assert result == False
    
    def test_complete_features(self):
        """Test validation passes with complete features."""
        inference = AutoencoderInference()
        
        # Complete features
        complete_features = {f: 0 for f in inference.FEATURES}
        
        result = inference._validate_features(complete_features)
        assert result == True


class TestInvalidReconstructionErrors:
    """Tests for invalid reconstruction error handling."""
    
    def test_nan_error_handling(self):
        """Test handling of NaN reconstruction error."""
        inference = AutoencoderInference()
        
        error, reason = inference._handle_invalid_error(float('nan'))
        
        assert error == 999.0
        assert reason is not None
        assert 'invalid' in reason.lower() or 'clipped' in reason.lower()
    
    def test_inf_error_handling(self):
        """Test handling of infinite reconstruction error."""
        inference = AutoencoderInference()
        
        error, reason = inference._handle_invalid_error(float('inf'))
        
        assert error == 999.0
        assert reason is not None
        assert 'invalid' in reason.lower() or 'clipped' in reason.lower()
    
    def test_negative_inf_error_handling(self):
        """Test handling of negative infinite reconstruction error."""
        inference = AutoencoderInference()
        
        error, reason = inference._handle_invalid_error(float('-inf'))
        
        assert error == 999.0
        assert reason is not None
    
    def test_valid_error_passthrough(self):
        """Test that valid errors pass through unchanged."""
        inference = AutoencoderInference()
        
        valid_error = 0.05
        error, reason = inference._handle_invalid_error(valid_error)
        
        assert error == valid_error
        assert reason is None


class TestScoreTransactionEdgeCases:
    """Tests for score_transaction edge cases."""
    
    def test_score_without_loading(self):
        """Test scoring when artifacts not loaded returns None."""
        inference = AutoencoderInference()
        inference._loaded = False
        
        # Mock load_artifacts to return False
        inference.load_artifacts = lambda: False
        
        features = {f: 0 for f in inference.FEATURES}
        result = inference.score_transaction(features)
        
        assert result is None
    
    def test_score_with_invalid_features(self):
        """Test scoring with invalid features returns None."""
        inference = AutoencoderInference()
        inference._loaded = True
        inference.threshold_config = {'threshold': 0.05}
        
        # Empty features
        result = inference.score_transaction({})
        
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
