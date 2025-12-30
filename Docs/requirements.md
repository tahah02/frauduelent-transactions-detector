# Requirements Document

## Introduction

This document specifies the requirements for integrating an Autoencoder-based anomaly detection model into the existing Banking Anomaly Detection System. The Autoencoder will complement the existing Rule Engine and Isolation Forest models by providing behavioral anomaly scoring based on reconstruction error. The integration must preserve all existing functionality while adding a new layer of ML-based risk assessment.

## Glossary

- **Autoencoder**: A neural network architecture that learns to compress and reconstruct input data, used here to detect anomalies via reconstruction error
- **Reconstruction_Error**: The difference between the original input and the Autoencoder's reconstructed output, measured as mean squared error
- **Threshold**: A configurable boundary value (mean + k*std of reconstruction errors) above which a transaction is flagged as anomalous
- **Hybrid_Decision_Engine**: The component that combines Rule Engine, Isolation Forest, and Autoencoder outputs to produce a final fraud decision
- **Feature_Scaler**: A StandardScaler that normalizes input features before Autoencoder processing
- **Engineered_Features**: The pre-computed transaction features stored in `data/engineered_transaction_features.csv`

## Requirements

### Requirement 1: Autoencoder Model Architecture

**User Story:** As a fraud analyst, I want an Autoencoder model that learns normal transaction behavior, so that I can detect anomalies based on how well transactions can be reconstructed.

#### Acceptance Criteria

1. THE Autoencoder SHALL be implemented as a Keras neural network with encoder and decoder layers
2. THE Autoencoder SHALL accept the same feature set used by the Isolation Forest model
3. THE Autoencoder SHALL output a reconstruction of the input features
4. THE Autoencoder SHALL be defined in `backend/autoencoder.py` as a separate module
5. WHEN the Autoencoder is instantiated, THE Autoencoder SHALL support configurable hidden layer dimensions

### Requirement 2: Autoencoder Training Pipeline

**User Story:** As a data scientist, I want to train the Autoencoder on historical transaction data, so that it learns the patterns of normal behavior.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL load data from `data/engineered_transaction_features.csv`
2. THE Training_Pipeline SHALL fit a StandardScaler on the training features and save it to `backend/autoencoder_scaler.pkl`
3. THE Training_Pipeline SHALL use the saved scaler (`autoencoder_scaler.pkl`) to guarantee metric consistency between training and inference
4. THE Training_Pipeline SHALL scale features before training the Autoencoder
5. THE Training_Pipeline SHALL train the Autoencoder to minimize reconstruction error (mean squared error)
6. WHEN training completes, THE Training_Pipeline SHALL save the model in `.h5` format to `models/autoencoder.h5`
7. WHEN training completes, THE Training_Pipeline SHALL compute the threshold using the formula: `threshold = mean(reconstruction_error) + k * std(reconstruction_error)` where k is a configurable parameter (default k=3)
8. THE Training_Pipeline SHALL save the threshold configuration to `models/autoencoder_threshold.json` including the threshold value, k parameter, mean, and std used
9. THE Training_Pipeline SHALL be executable via `python -m backend.train_autoencoder`
10. THE Training_Pipeline SHALL log the number of features, training samples, and computed threshold upon completion

### Requirement 3: Post-Training Validation

**User Story:** As a data scientist, I want to validate that the saved model produces consistent results, so that I can trust the model in production.

#### Acceptance Criteria

1. WHEN training completes, THE Training_Pipeline SHALL reload the saved `.h5` model
2. WHEN validation runs, THE Training_Pipeline SHALL verify the loaded model accepts the expected feature shape
3. WHEN validation runs, THE Training_Pipeline SHALL compute reconstruction errors on sample inputs
4. IF the reloaded model produces metrics that differ from training metrics by more than 1%, THEN THE Training_Pipeline SHALL abort and log a metrics mismatch error
5. WHEN validation succeeds, THE Training_Pipeline SHALL log a success message confirming model integrity
6. THE Training_Pipeline SHALL validate that the scaler loads successfully and transforms features correctly

### Requirement 4: Autoencoder Inference

**User Story:** As a fraud detection system, I want to score transactions using the Autoencoder, so that I can identify behavioral anomalies.

#### Acceptance Criteria

1. THE Inference_Module SHALL load the Autoencoder model from `models/autoencoder.h5`
2. THE Inference_Module SHALL load the scaler from `backend/autoencoder_scaler.pkl`
3. THE Inference_Module SHALL load the threshold from `models/autoencoder_threshold.json`
4. WHEN a transaction is scored, THE Inference_Module SHALL scale the input features using the loaded scaler
5. WHEN a transaction is scored, THE Inference_Module SHALL compute the reconstruction error
6. WHEN a transaction is scored, THE Inference_Module SHALL compare the reconstruction error to the threshold
7. WHEN the reconstruction error exceeds the threshold, THE Inference_Module SHALL return an anomaly flag with a human-readable reason in the format: "Autoencoder anomaly: reconstruction error {error:.3f} exceeds threshold {threshold:.3f}"

### Requirement 5: Hybrid Decision Integration

**User Story:** As a fraud analyst, I want the Autoencoder results integrated into the existing decision flow, so that I get a comprehensive risk assessment.

#### Acceptance Criteria

1. THE Hybrid_Decision_Engine SHALL maintain the existing decision priority: Rule Engine (block) → Isolation Forest (risk) → Autoencoder (risk)
2. THE Hybrid_Decision_Engine SHALL invoke Autoencoder inference after Isolation Forest scoring
3. WHEN the Autoencoder flags an anomaly, THE Hybrid_Decision_Engine SHALL add the Autoencoder reason to the reasons list
4. WHEN the Autoencoder flags an anomaly, THE Hybrid_Decision_Engine SHALL set is_fraud to True
5. THE Hybrid_Decision_Engine SHALL include the Autoencoder reconstruction error in the result output
6. THE Hybrid_Decision_Engine SHALL NOT modify existing Rule Engine or Isolation Forest logic

### Requirement 6: Error Handling and Safeguards

**User Story:** As a system operator, I want the system to handle missing or corrupted Autoencoder artifacts gracefully, so that the existing detection continues to work.

#### Acceptance Criteria

1. IF the Autoencoder model file is missing, THEN THE Inference_Module SHALL skip Autoencoder scoring, log a warning with the missing file path, and continue
2. IF the Autoencoder scaler file is missing, THEN THE Inference_Module SHALL skip Autoencoder scoring, log a warning with the missing file path, and continue
3. IF the threshold file is missing, THEN THE Inference_Module SHALL skip Autoencoder scoring, log a warning with the missing file path, and continue
4. IF the input features do not match the expected feature shape, THEN THE Inference_Module SHALL skip Autoencoder scoring, log a warning with expected vs actual shape, and continue
5. IF the reconstruction error is NaN or infinite, THEN THE Inference_Module SHALL clip the value to a maximum anomaly score, flag as anomaly, and log the incident for investigation
6. WHEN any Autoencoder error occurs, THE Hybrid_Decision_Engine SHALL continue processing with Rule Engine and Isolation Forest results only
7. THE Inference_Module SHALL log all warnings and errors with timestamps and transaction context for debugging
8. THE Inference_Module SHALL validate model load success before attempting inference

### Requirement 7: Model Serialization Round-Trip

**User Story:** As a data scientist, I want to ensure that saved models can be correctly loaded and produce identical results, so that I can trust model persistence.

#### Acceptance Criteria

1. FOR ALL valid Autoencoder models, saving to `.h5` then loading SHALL produce an equivalent model
2. FOR ALL valid scalers, saving to `.pkl` then loading SHALL produce an equivalent scaler
3. FOR ALL valid threshold configurations, saving to `.json` then loading SHALL produce an equivalent threshold value
4. THE Pretty_Printer SHALL format threshold configurations as valid JSON
5. FOR ALL valid threshold JSON objects, parsing then printing then parsing SHALL produce an equivalent object (round-trip property)

### Requirement 8: File Structure Compliance

**User Story:** As a developer, I want the Autoencoder integration to follow the established project structure, so that the codebase remains maintainable.

#### Acceptance Criteria

1. THE Autoencoder module SHALL be located at `backend/autoencoder.py`
2. THE Training script SHALL be located at `backend/train_autoencoder.py`
3. THE Autoencoder model SHALL be saved to `models/autoencoder.h5`
4. THE Autoencoder scaler SHALL be saved to `backend/autoencoder_scaler.pkl`
5. THE Autoencoder threshold SHALL be saved to `models/autoencoder_threshold.json`
6. THE Hybrid_Decision_Engine modifications SHALL remain in `backend/hybrid_decision.py`
