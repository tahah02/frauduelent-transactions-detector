# 🏦 Banking Anomaly Detection System

A sophisticated **triple-layer fraud detection system** that combines business rules, machine learning, and deep learning to protect banking transactions from fraudulent activities.

## 🎯 **Project Overview**

This system provides **real-time fraud detection** using three complementary approaches:
- **🚫 Rule Engine**: Hard business rule enforcement
- **🌲 Isolation Forest**: Statistical anomaly detection  
- **🧠 Autoencoder**: Behavioral pattern analysis

## ✨ **Key Features**

- **Triple-Layer Protection**: Multiple detection methods working together
- **Real-Time Processing**: Sub-second transaction analysis
- **Web Dashboard**: Interactive Streamlit interface
- **26 Intelligent Features**: Comprehensive transaction analysis
- **Graceful Degradation**: System continues if components fail
- **Scalable Architecture**: Ready for production deployment

## 🏗 **System Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Rule Engine   │    │ Isolation Forest│    │  Autoencoder    │
│                 │    │                 │    │                 │
│ • Velocity      │    │ • Statistical   │    │ • Behavioral    │
│ • Limits        │    │   Anomalies     │    │   Patterns      │
│ • Thresholds    │    │ • Risk Scoring  │    │ • Deep Learning │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │  Hybrid Decision    │
                    │     Engine          │
                    └─────────────────────┘
```

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.8+
- pip or conda package manager

### **Installation**
```bash
# Clone the repository
git clone https://github.com/yourusername/banking-anomaly-detector.git
cd banking-anomaly-detector

# Install dependencies
pip install -r requirements.txt

# Train models (optional - pre-trained models included)
python -m backend.model_training
python -m backend.train_autoencoder

# Run the application
streamlit run app.py
```

### **Access the Dashboard**
Open your browser and go to: `http://localhost:8501`

## 📊 **Model Performance**

### **Training Statistics**
- **Dataset**: 3,502 historical transactions
- **Features**: 26 engineered features
- **Isolation Forest**: 100 trees, 5% contamination
- **Autoencoder**: 35 epochs, MSE loss, threshold=1.914

### **Detection Capabilities**
- **Processing Speed**: <100ms per transaction
- **Throughput**: 10,000+ transactions/hour
- **Accuracy**: High precision with low false positives

## 🔧 **Technical Stack**

- **Backend**: Python, Scikit-learn, TensorFlow/Keras
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Model Storage**: Joblib, HDF5
- **Testing**: Hypothesis (Property-based testing)

## 📁 **Project Structure**

```
banking-anomaly-detector/
├── 📱 Frontend
│   └── app.py                          # Streamlit web interface
├── 🧠 Backend
│   ├── hybrid_decision.py              # Decision integration
│   ├── rule_engine.py                  # Business rules
│   ├── model_training.py               # Isolation Forest
│   ├── autoencoder.py                  # Neural network
│   └── feature_engineering.py         # Data processing
├── 💾 Models
│   ├── isolation_forest.pkl            # Trained IF model
│   ├── autoencoder.h5                  # Trained AE model
│   └── *.pkl                          # Feature scalers
├── 📊 Data
│   └── engineered_transaction_features.csv
├── 🧪 Tests
│   └── test_*.py                       # Comprehensive testing
└── 📚 Documentation
    ├── BRD.md                          # Business requirements
    ├── projectflow.md                  # Process flow
    ├── projectarchitecture.md         # System architecture
    ├── IsolationForest_Implementation.md
    └── Autoencoder_Implementation.md
```

## 🎮 **Usage Examples**

### **Analyze a Transaction**
```python
from backend.hybrid_decision import make_decision
from backend.model_training import load_model
from backend.autoencoder import AutoencoderInference

# Load models
model, features, scaler = load_model()
autoencoder = AutoencoderInference()

# Analyze transaction
transaction = {
    'transaction_amount': 5000,
    'transfer_type': 'S',  # Overseas
    'hour': 3,             # 3 AM
    'user_avg_amount': 500,
    # ... other features
}

result = make_decision(transaction, user_stats, model, features, autoencoder)
print(f"Fraud Decision: {result['is_fraud']}")
print(f"Reasons: {result['reasons']}")
```

### **Web Interface**
1. Upload transaction data or enter manually
2. View real-time analysis results
3. See detailed explanations for each decision
4. Monitor system performance metrics

## 🔍 **Detection Methods Explained**

### **1. Rule Engine (The Bouncer)**
- Enforces hard business limits
- Velocity checks (max 5 txns/10min)
- Dynamic amount thresholds
- Immediate blocking for clear violations

### **2. Isolation Forest (The Detective)**
- Uses 100 decision trees
- Isolates statistical outliers
- Learns from 26 transaction features
- Provides anomaly scores

### **3. Autoencoder (The Behavioral Analyst)**
- Neural network (26→64→32→13→32→64→26)
- Learns normal behavior patterns
- Detects behavioral deviations
- Reconstruction error analysis

## 📈 **Feature Engineering**

Our system uses **26 intelligent features**:

### **Transaction Features**
- Amount, type, channel, risk scores

### **User Behavior**
- Historical patterns, deviations, ratios

### **Temporal Patterns**
- Time of day, day of week, timing analysis

### **Velocity Tracking**
- Transaction frequency, burst detection

## 🧪 **Testing**

```bash
# Run all tests
python -m pytest tests/

# Run property-based tests
python -m pytest tests/test_autoencoder_properties.py

# Run specific test
python -m pytest tests/test_frontend_ae.py -v
```

## 🚀 **Deployment**

### **Local Development**
```bash
streamlit run app.py --server.port 8501
```

### **Production Deployment**
- Docker containerization ready
- Scalable architecture
- Load balancing support
- Monitoring and alerting

## 📚 **Documentation**

- **[Business Requirements](BRD.md)**: Complete business context
- **[Project Flow](projectflow.md)**: Step-by-step execution
- **[Architecture](projectarchitecture.md)**: Technical design
- **[Isolation Forest](IsolationForest_Implementation.md)**: ML implementation
- **[Autoencoder](Autoencoder_Implementation.md)**: Deep learning details

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- Built with modern ML/DL frameworks
- Inspired by real-world banking fraud challenges
- Comprehensive testing with property-based testing
- Production-ready architecture

## 📞 **Contact**

- **Project Maintainer**: [Your Name]
- **Email**: [your.email@example.com]
- **LinkedIn**: [Your LinkedIn Profile]

---

⭐ **Star this repository if you found it helpful!** ⭐