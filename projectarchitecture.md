# Banking Anomaly Detection System - Project Architecture

## 🏗 **System Architecture Overview**

The Banking Anomaly Detection System follows a **layered microservices architecture** with clear separation of concerns, enabling scalability, maintainability, and robust fraud detection capabilities.

## 📐 **High-Level Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    🌐 PRESENTATION LAYER                    │
├─────────────────────────────────────────────────────────────┤
│  Streamlit Web Interface (app.py)                          │
│  ├── Authentication & Session Management                    │
│  ├── Dashboard & Visualization                             │
│  ├── Transaction Input Forms                               │
│  └── Results Display & Analytics                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   🧠 BUSINESS LOGIC LAYER                   │
├─────────────────────────────────────────────────────────────┤
│  Hybrid Decision Engine (hybrid_decision.py)               │
│  ├── 🚫 Rule Engine Integration                            │
│  ├── 🌲 Isolation Forest Integration                       │
│  ├── 🧠 Autoencoder Integration                            │
│  └── 🎯 Decision Aggregation Logic                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   🔍 DETECTION SERVICES LAYER               │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │🚫 Rule      │  │🌲 Isolation │  │🧠 Autoencoder│        │
│  │  Engine     │  │  Forest     │  │  Neural Net │        │
│  │             │  │             │  │             │        │
│  │• Velocity   │  │• Anomaly    │  │• Behavioral │        │
│  │• Limits     │  │  Detection  │  │  Analysis   │        │
│  │• Thresholds │  │• Risk Score │  │• Pattern    │        │
│  │             │  │             │  │  Learning   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   🔧 DATA PROCESSING LAYER                  │
├─────────────────────────────────────────────────────────────┤
│  Feature Engineering (feature_engineering.py)              │
│  ├── Transaction Feature Extraction                        │
│  ├── User Behavior Analysis                                │
│  ├── Temporal Pattern Recognition                          │
│  ├── Velocity Calculations                                 │
│  └── Data Normalization & Scaling                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    💾 DATA STORAGE LAYER                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │📊 Training  │  │🤖 ML Models │  │⚙️ Config    │        │
│  │   Data      │  │             │  │   Files     │        │
│  │             │  │• isolation_ │  │             │        │
│  │• Raw Trans  │  │  forest.pkl │  │• thresholds │        │
│  │• Features   │  │• autoencoder│  │• scalers    │        │
│  │• History    │  │  .h5        │  │• params     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## 🧩 **Component Architecture**

### **1. Presentation Layer Components**

#### **Streamlit Web Application (app.py)**
```
🌐 Web Interface Architecture
├── 🔐 Authentication Module
│   ├── Session management
│   ├── User validation
│   └── Security controls
├── 📊 Dashboard Components
│   ├── Transaction input forms
│   ├── Real-time processing display
│   ├── Results visualization
│   └── System status monitoring
└── 🎨 UI/UX Elements
    ├── Responsive design
    ├── Interactive charts
    └── User-friendly navigation
```

### **2. Business Logic Layer Components**

#### **Hybrid Decision Engine (hybrid_decision.py)**
```
🎯 Decision Integration Architecture
├── 🔄 Processing Pipeline
│   ├── Sequential layer execution
│   ├── Priority-based decision making
│   └── Result aggregation
├── 🚫 Rule Engine Interface
│   ├── Business rule validation
│   ├── Hard limit enforcement
│   └── Immediate blocking logic
├── 🌲 ML Model Interface
│   ├── Isolation Forest integration
│   ├── Anomaly score processing
│   └── Risk assessment
└── 🧠 Neural Network Interface
    ├── Autoencoder integration
    ├── Behavioral analysis
    └── Pattern recognition
```

### **3. Detection Services Layer**

#### **Rule Engine Service (rule_engine.py)**
```
🚫 Business Rules Architecture
├── 📏 Threshold Calculations
│   ├── Dynamic limit computation
│   ├── User-specific adjustments
│   └── Transfer type multipliers
├── ⚡ Velocity Monitoring
│   ├── Transaction frequency tracking
│   ├── Time-window analysis
│   └── Burst detection
└── 🎯 Decision Logic
    ├── Hard blocking rules
    ├── Violation detection
    └── Reason generation
```

#### **Isolation Forest Service (model_training.py)**
```
🌲 ML Anomaly Detection Architecture
├── 🤖 Model Management
│   ├── Training pipeline
│   ├── Model persistence
│   └── Version control
├── 📊 Feature Processing
│   ├── Data preprocessing
│   ├── Feature scaling
│   └── Anomaly scoring
└── 🎯 Decision Making
    ├── Threshold comparison
    ├── Risk score calculation
    └── Anomaly flagging
```

#### **Autoencoder Service (autoencoder.py)**
```
🧠 Neural Network Architecture
├── 🏗 Model Structure
│   ├── Encoder: Input(26) → [64,32] → Bottleneck(13)
│   ├── Decoder: Bottleneck(13) → [32,64] → Output(26)
│   └── Loss Function: Mean Squared Error
├── 🔧 Training Pipeline
│   ├── Data preprocessing
│   ├── Model training (100 epochs)
│   ├── Threshold calculation
│   └── Model validation
└── ⚡ Inference Engine
    ├── Real-time reconstruction
    ├── Error calculation
    ├── Anomaly detection
    └── Behavioral analysis
```

## 🗂 **File Structure Architecture**

```
banking_anomaly_detector/
├── 📱 Frontend Layer
│   └── app.py                          # Streamlit web interface
├── 🧠 Business Logic Layer  
│   └── backend/
│       ├── hybrid_decision.py          # Decision integration
│       ├── rule_engine.py              # Business rules
│       ├── model_training.py           # Isolation Forest
│       ├── autoencoder.py              # Neural network
│       ├── feature_engineering.py     # Data processing
│       └── utils.py                    # Shared utilities
├── 💾 Data Layer
│   ├── data/                           # Training datasets
│   │   ├── engineered_transaction_features.csv
│   │   └── feature_Engineered.csv
│   └── models/                         # Trained models
│       ├── isolation_forest.pkl
│       ├── isolation_forest_scaler.pkl
│       ├── autoencoder.h5
│       ├── autoencoder_scaler.pkl
│       └── autoencoder_threshold.json
├── 🧪 Testing Layer
│   └── tests/
│       ├── test_autoencoder_properties.py
│       ├── test_autoencoder_errors.py
│       └── test_frontend_ae.py
└── 📚 Documentation
    ├── BRD.md                          # Business requirements
    ├── projectflow.md                  # Process flow
    └── projectarchitecture.md         # This document
```

## 🔄 **Data Flow Architecture**

### **Training Data Flow**
```
📊 Training Pipeline
Raw Data → Feature Engineering → Model Training → Model Storage
    ↓              ↓                    ↓              ↓
CSV Files → 26 Features → IF + AE Models → PKL/H5 Files
```

### **Inference Data Flow**
```
⚡ Real-time Processing
Transaction → Features → Rule Check → ML Analysis → Decision
     ↓           ↓          ↓           ↓            ↓
  Input Data → 26 Dims → Block/Pass → Scores → Final Result
```

## 🛡 **Security Architecture**

### **Data Protection**
```
🔒 Security Layers
├── 🔐 Authentication
│   ├── Session-based login
│   ├── User validation
│   └── Access control
├── 🛡 Data Security
│   ├── Input validation
│   ├── SQL injection prevention
│   └── XSS protection
└── 🔍 Audit Logging
    ├── Decision tracking
    ├── User activity logs
    └── System monitoring
```

## ⚡ **Performance Architecture**

### **Optimization Strategies**
```
🚀 Performance Design
├── 💾 Caching Layer
│   ├── Model caching (@st.cache_resource)
│   ├── Feature caching
│   └── Result caching
├── 🔄 Lazy Loading
│   ├── Model initialization on demand
│   ├── Feature computation optimization
│   └── Memory management
└── 📊 Scalability
    ├── Stateless design
    ├── Horizontal scaling ready
    └── Load balancing support
```

## 🔧 **Technology Stack**

### **Core Technologies**
```
🛠 Technology Architecture
├── 🐍 Backend Framework
│   ├── Python 3.13
│   ├── Streamlit (Web UI)
│   └── NumPy/Pandas (Data processing)
├── 🤖 Machine Learning
│   ├── Scikit-learn (Isolation Forest)
│   ├── TensorFlow/Keras (Autoencoder)
│   └── Joblib (Model persistence)
├── 💾 Data Storage
│   ├── CSV files (Training data)
│   ├── PKL files (ML models)
│   └── JSON files (Configuration)
└── 🧪 Testing & Quality
    ├── Hypothesis (Property-based testing)
    ├── Pytest (Unit testing)
    └── Custom validation
```

## 🔌 **Integration Architecture**

### **External System Integration**
```
🔗 Integration Points
├── 📊 Data Sources
│   ├── Transaction databases
│   ├── User behavior data
│   └── Historical patterns
├── 🚨 Alerting Systems
│   ├── Fraud notifications
│   ├── System monitoring
│   └── Performance alerts
└── 📈 Analytics Platforms
    ├── Business intelligence
    ├── Reporting systems
    └── Compliance tracking
```

## 🎯 **Deployment Architecture**

### **Environment Strategy**
```
🚀 Deployment Design
├── 🧪 Development Environment
│   ├── Local development
│   ├── Unit testing
│   └── Feature development
├── 🔍 Testing Environment
│   ├── Integration testing
│   ├── Performance testing
│   └── User acceptance testing
└── 🏭 Production Environment
    ├── High availability setup
    ├── Load balancing
    ├── Monitoring & alerting
    └── Backup & recovery
```

This architecture ensures scalability, maintainability, and robust fraud detection while maintaining clear separation of concerns and enabling future enhancements.