# Implementation Plan: Autoencoder Integration

## Overview

This implementation plan integrates an Autoencoder-based anomaly detection model into the existing Banking Anomaly Detection System. The tasks are organized to build incrementally: core model → training pipeline → inference → integration → frontend → testing.

## Tasks

- [x] 1. Set up project structure and dependencies
  - Add TensorFlow/Keras to `pyproject.toml` dependencies
  - Create `models/` directory if not exists
  - Verify `data/engineered_transaction_features.csv` exists
  - _Requirements: 8.1, 8.3, 8.4, 8.5_

- [x] 2. Implement Autoencoder model class
  - [x] 2.1 Create `backend/autoencoder.py` with `TransactionAutoencoder` class
    - Implement `__init__` with configurable input_dim, encoding_dim, hidden_layers
    - Implement `build_model()` to create Keras Sequential model with encoder/decoder layers
    - Implement `fit()` method for training with MSE loss
    - Implement `predict()` for reconstruction
    - Implement `compute_reconstruction_error()` for MSE calculation
    - Implement `save()` and `load()` for .h5 persistence
    - _Requirements: 1.1, 1.2, 1.3, 1.5_

  - [x] 2.2 Write property test for reconstruction output shape preservation
    - **Property 4: Reconstruction Output Shape Preservation**
    - **Validates: Requirements 1.3**

- [x] 3. Implement training pipeline
  - [x] 3.1 Create `backend/train_autoencoder.py` with `AutoencoderTrainer` class
    - Define FEATURES list (same 26 features as Isolation Forest)
    - Define file paths: DATA_PATH, MODEL_PATH, SCALER_PATH, THRESHOLD_PATH
    - Implement `load_data()` to load CSV
    - Implement `fit_scaler()` to fit StandardScaler and save to .pkl
    - Implement `compute_threshold()` with formula: mean + k * std
    - Implement `save_threshold()` to save JSON config
    - Implement `train()` orchestration method
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10_

  - [x] 3.2 Write property test for threshold formula correctness
    - **Property 5: Threshold Formula Correctness**
    - **Validates: Requirements 2.7**

  - [x] 3.3 Write property test for scaler round-trip consistency
    - **Property 2: Scaler Round-Trip Consistency**
    - **Validates: Requirements 2.2, 2.3, 4.2, 7.2**

  - [x] 3.4 Write property test for threshold configuration round-trip
    - **Property 3: Threshold Configuration Round-Trip**
    - **Validates: Requirements 2.8, 4.3, 7.3, 7.4, 7.5**

- [x] 4. Implement post-training validation
  - [x] 4.1 Add `validate_saved_model()` method to `AutoencoderTrainer`
    - Reload saved .h5 model
    - Verify feature shape acceptance
    - Compute reconstruction errors on sample inputs
    - Compare with training metrics (fail if >1% difference)
    - Log success/failure messages
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [x] 4.2 Write property test for model round-trip consistency
    - **Property 1: Model Round-Trip Consistency**
    - **Validates: Requirements 2.6, 3.1, 3.3, 4.1, 7.1**

- [x] 5. Checkpoint - Verify training pipeline
  - Run `python -m backend.train_autoencoder`
  - Verify `models/autoencoder.h5` is created
  - Verify `backend/autoencoder_scaler.pkl` is created
  - Verify `models/autoencoder_threshold.json` is created
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement inference module
  - [x] 6.1 Add `AutoencoderInference` class to `backend/autoencoder.py`
    - Implement `__init__` with lazy loading
    - Implement `load_artifacts()` to load model, scaler, threshold
    - Implement `is_available()` check
    - Implement `_validate_features()` for shape validation
    - Implement `_handle_invalid_error()` for NaN/inf handling
    - Implement `score_transaction()` with human-readable reason generation
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

  - [x] 6.2 Write property test for inference pipeline correctness
    - **Property 6: Inference Pipeline Correctness**
    - **Validates: Requirements 4.4, 4.5, 4.6, 4.7**

- [x] 7. Implement error handling and safeguards
  - [x] 7.1 Add error handling to `AutoencoderInference`
    - Handle missing model file with warning log
    - Handle missing scaler file with warning log
    - Handle missing threshold file with warning log
    - Handle feature shape mismatch with warning log
    - Handle NaN/inf reconstruction errors with clipping and error log
    - Add timestamps and transaction context to all logs
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.7, 6.8_

  - [x] 7.2 Write unit tests for error handling edge cases
    - Test missing model file scenario
    - Test missing scaler file scenario
    - Test missing threshold file scenario
    - Test feature shape mismatch scenario
    - Test NaN/inf reconstruction error scenario
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 8. Integrate Autoencoder into hybrid decision engine
  - [x] 8.1 Update `backend/hybrid_decision.py` `make_decision()` function
    - Add optional `autoencoder` parameter
    - Add `ae_flag` and `ae_reconstruction_error` to result dict
    - Invoke Autoencoder inference after Isolation Forest
    - Add AE reason to reasons list when anomaly detected
    - Set is_fraud=True when AE flags anomaly
    - Preserve existing Rule Engine and Isolation Forest logic unchanged
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

  - [x] 8.2 Write property test for hybrid decision AE integration
    - **Property 7: Hybrid Decision AE Integration**
    - **Validates: Requirements 5.3, 5.4, 5.5**

  - [x] 8.3 Write property test for graceful degradation
    - **Property 8: Graceful Degradation on AE Failure**
    - **Validates: Requirements 6.6**

- [x] 9. Checkpoint - Verify backend integration
  - Test hybrid decision with AE enabled
  - Test hybrid decision with AE disabled/unavailable
  - Verify Rule Engine and Isolation Forest still work correctly
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Update Streamlit frontend
  - [x] 10.1 Update `app.py` with Autoencoder integration
    - Import `AutoencoderInference` from backend
    - Add `get_autoencoder()` function with `@st.cache_resource`
    - Update `dashboard()` to initialize AE and pass to `make_decision()`
    - Add AE status indicator to sidebar
    - Display AE reconstruction error and threshold in result panel
    - Display AE anomaly reason in reasons list
    - Handle graceful degradation when AE unavailable
    - _Requirements: 5.3, 5.4, 5.5, 6.6_

  - [x] 10.2 Write unit tests for frontend AE display
    - Test AE status indicator display
    - Test AE result fields display
    - Test graceful degradation UI
    - _Requirements: 5.3, 5.4, 5.5_

- [x] 11. Final checkpoint - End-to-end validation
  - Run full training pipeline: `python -m backend.train_autoencoder`
  - Start Streamlit app: `streamlit run app.py`
  - Process test transaction and verify AE output appears
  - Verify existing Rule Engine and Isolation Forest still work
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- All tasks including property-based tests are required for comprehensive validation
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties using `hypothesis`
- Unit tests validate specific examples and edge cases
- The implementation preserves all existing functionality (non-breaking integration)
