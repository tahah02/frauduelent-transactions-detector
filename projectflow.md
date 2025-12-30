# Banking Anomaly Detection System - Project Flow

## 🚀 **System Execution Flow**

### **1. System Startup**
```
🔄 Application Launch
├── 📊 Load Isolation Forest Model (models/isolation_forest.pkl)
├── 🧠 Load Autoencoder Model (models/autoencoder.h5)
├── ⚖️ Load Feature Scalers (models/*.pkl)
├── 🎯 Load Threshold Configurations (models/autoencoder_threshold.json)
└── 🌐 Start Streamlit Web Interface (Port 9000)
```

### **2. Transaction Processing Pipeline**

#### **Step 1: Data Input**
```
📥 New Transaction Received
├── Customer ID
├── Transaction Amount
├── Transfer Type (S/I/L/Q/O)
├── Channel (Mobile/Web/ATM)
├── Timestamp
└── Additional Metadata
```

#### **Step 2: Feature Engineering**
```
🔧 Feature Generation (26 Features)
├── 💰 Transaction Features
│   ├── transaction_amount
│   ├── flag_amount (overseas indicator)
│   ├── transfer_type_encoded
│   └── transfer_type_risk
├── 👤 User Behavior Features  
│   ├── user_avg_amount
│   ├── user_std_amount
│   ├── deviation_from_avg
│   └── amount_to_max_ratio
├── ⏰ Temporal Features
│   ├── hour, day_of_week
│   ├── is_weekend, is_night
│   └── time_since_last
└── 🚀 Velocity Features
    ├── txn_count_30s/10min/1hour
    ├── hourly_total/count
    └── daily_total/count
```

#### **Step 3: Triple-Layer Detection**

##### **Layer 1: Rule Engine (Hard Blocks)**
```
🚫 Business Rule Validation
├── Velocity Check
│   ├── Max 5 transactions in 10 minutes
│   └── Max 15 transactions in 1 hour
├── Amount Limits
│   ├── Dynamic threshold = user_avg + (multiplier × user_std)
│   ├── Multipliers: S=2.0, Q=2.5, L=3.0, I=3.5, O=4.0
│   └── Minimum floors: S=5000, Q=3000, L=2000, I=1500, O=1000
└── Decision: BLOCK (if violated) or CONTINUE
```

##### **Layer 2: Isolation Forest (ML Anomaly Detection)**
```
🌲 Isolation Forest Analysis
├── Feature Scaling (StandardScaler)
├── Anomaly Score Calculation
├── Threshold Comparison
├── Risk Score Generation
└── Decision: FLAG (if anomalous) or CONTINUE
```

##### **Layer 3: Autoencoder (Behavioral Analysis)**
```
🧠 Autoencoder Behavioral Analysis
├── Feature Scaling (StandardScaler)
├── Neural Network Reconstruction
│   ├── Encoder: [64, 32] → 13 (bottleneck)
│   └── Decoder: 13 → [32, 64] → 26 (reconstruction)
├── Reconstruction Error Calculation (MSE)
├── Threshold Comparison (mean + 3×std = 1.914)
└── Decision: FLAG (if error > threshold) or PASS
```

### **3. Decision Integration**

#### **Hybrid Decision Logic**
```
🎯 Final Decision Process
├── Priority 1: Rule Engine Result
│   └── If BLOCKED → Return "FRAUD: Rule Violation"
├── Priority 2: Isolation Forest Result  
│   └── If FLAGGED → Add "ML Anomaly" to reasons
├── Priority 3: Autoencoder Result
│   └── If FLAGGED → Add "Behavioral Anomaly" to reasons
└── Final Output:
    ├── is_fraud: boolean
    ├── reasons: list of explanations
    ├── risk_score: numerical score
    └── model_details: individual results
```

### **4. User Interface Flow**

#### **Web Dashboard Process**
```
🌐 Streamlit Web Interface
├── 🔐 User Authentication
├── 📊 Dashboard Loading
│   ├── Load transaction data
│   ├── Initialize models
│   └── Display system status
├── 📥 Transaction Input
│   ├── Manual entry form
│   ├── CSV file upload
│   └── Real-time processing
├── 🔍 Analysis Display
│   ├── Rule engine results
│   ├── ML model scores
│   ├── Autoencoder metrics
│   └── Combined decision
└── 📈 Results Visualization
    ├── Risk score charts
    ├── Feature importance
    └── Decision explanations
```

## 🔄 **Data Flow Architecture**

### **Training Phase (Offline)**
```
📚 Model Training Pipeline
├── 📊 Data Loading
│   └── data/engineered_transaction_features.csv (3,502 samples)
├── 🔧 Feature Processing
│   ├── StandardScaler fitting
│   └── Feature validation (26 features)
├── 🌲 Isolation Forest Training
│   ├── Anomaly detection learning
│   └── Model saving (models/isolation_forest.pkl)
├── 🧠 Autoencoder Training
│   ├── Neural network training (100 epochs)
│   ├── Reconstruction error analysis
│   ├── Threshold calculation (mean + 3×std)
│   └── Model saving (models/autoencoder.h5)
└── 💾 Artifact Storage
    ├── Trained models
    ├── Feature scalers
    └── Configuration files
```

### **Inference Phase (Online)**
```
⚡ Real-time Processing
├── 📥 Transaction Input
├── 🔧 Feature Engineering
├── 🚫 Rule Engine Check
├── 🌲 Isolation Forest Scoring
├── 🧠 Autoencoder Analysis
├── 🎯 Decision Integration
└── 📤 Result Output
```

## 🛠 **System Components Interaction**

### **Backend Services**
```
🏗 Backend Architecture
├── rule_engine.py → Business logic validation
├── model_training.py → Isolation Forest management
├── autoencoder.py → Neural network operations
├── hybrid_decision.py → Decision integration
├── feature_engineering.py → Data preprocessing
└── utils.py → Shared utilities
```

### **Frontend Interface**
```
🖥 Frontend Components
├── app.py → Main Streamlit application
├── Authentication → User login system
├── Dashboard → Transaction analysis interface
├── Visualization → Charts and metrics display
└── Results → Decision explanation panel
```

### **Data Storage**
```
💾 Data Management
├── models/ → Trained ML models and configurations
├── data/ → Training datasets and features
├── backend/ → Source code and business logic
└── tests/ → Quality assurance and validation
```

## 🔍 **Error Handling Flow**

### **Graceful Degradation**
```
🛡 Error Recovery Process
├── Model Loading Failure
│   ├── Log warning message
│   ├── Continue with available models
│   └── Notify user of reduced functionality
├── Feature Processing Error
│   ├── Use default values
│   ├── Log incident for investigation
│   └── Continue processing
└── Decision Engine Failure
    ├── Fall back to rule engine only
    ├── Alert system administrators
    └── Maintain basic fraud protection
```

## 📊 **Performance Monitoring**

### **System Metrics**
```
📈 Performance Tracking
├── Transaction Processing Time
├── Model Accuracy Metrics
├── System Resource Usage
├── Error Rate Monitoring
└── User Experience Metrics
```

This flow ensures robust, scalable, and reliable fraud detection with multiple layers of protection and comprehensive error handling.