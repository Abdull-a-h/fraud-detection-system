🛡️ Real-Time Fraud Detection System
A production-ready fraud detection system using Machine Learning to identify fraudulent transactions in real-time with high accuracy and low false positives.

📊 Performance Metrics

Precision: 88.51% (highly accurate fraud flags)
Recall: 78.57% (catches most fraudulent transactions)
False Positive Rate: 0.02% (only 10 false alarms per 56,864 transactions)
Latency: <70ms (real-time detection)
ROC-AUC: 0.9726

✨ Features

✅ Real-time fraud detection with <100ms latency
✅ Multiple ML models (Random Forest, XGBoost, Logistic Regression)
✅ Optimized threshold for minimal false positives
✅ RESTful API with health checks and statistics
✅ Dockerized for easy deployment
✅ Production-ready with Gunicorn
✅ Comprehensive evaluation and visualization tools

🏗️ System Architecture
┌─────────────────┐
│  Transaction    │
│     Data        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocessing   │
│  - Scaling      │
│  - Features     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Random Forest  │
│     Model       │
│  (Threshold:    │
│    0.7814)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Fraud/Legit    │
│   Prediction    │
└─────────────────┘
📁 Project Structure
fraud_detection/
├── data/
│   ├── creditcard.csv          # Original dataset
│   ├── X_train.npy             # Preprocessed training features
│   ├── X_test.npy              # Preprocessed test features
│   ├── y_train.npy             # Training labels
│   └── y_test.npy              # Test labels
├── models/
│   ├── random_forest_model.pkl # Best performing model
│   ├── xgboost_model.pkl       # XGBoost model
│   └── logistic_regression_model.pkl
├── src/
│   ├── data_preprocessing.py   # Data preprocessing pipeline
│   ├── model_training.py       # Model training script
│   ├── model_evaluation.py     # Model evaluation and metrics
│   └── threshold_optimization.py # Threshold tuning
├── api/
│   ├── app.py                  # Flask API application
├── results/
│   ├── *_confusion_matrix.png  # Model visualizations
│   ├── *_roc_curve.png
│   └── *_precision_recall.png
├── Dockerfile                   # Docker configuration
├── docker-compose.yml          # Docker Compose setup
├── requirements.txt            # Python dependencies
├── test_api.py                 # API testing script
└── README.md                   # This file
🚀 Quick Start
Prerequisites

Python 3.9+
Docker (optional, for containerized deployment)
4GB RAM minimum

Installation

Clone the repository

bashgit clone https://github.com/Abdull-a-h/fraud-detection-system.git
cd fraud-detection-system

Create virtual environment

bash python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies

bash pip install -r requirements.txt

Download the dataset

Download from Kaggle Credit Card Fraud Detection
Place creditcard.csv in the data/ directory



Training the Model
bash # 1. Preprocess data
cd src
python data_preprocessing.py

# 2. Train models
python model_training.py

# 3. Evaluate models
python model_evaluation.py

# 4. Optimize threshold (optional)
python threshold_optimization.py
Running the API
Option 1: Local Development
bash cd api
python app.py
Option 2: Production (Gunicorn)
bash cd api
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 30 app:app
Option 3: Docker
bash # Build the image
docker build -t fraud-detection-api .

# Run the container
docker run -d -p 5000:5000 --name fraud-api fraud-detection-api

# Check logs
docker logs fraud-api
Option 4: Docker Compose
bashdocker-compose up -d
📡 API Endpoints
Health Check
bashGET /health

Response:
{
  "status": "healthy",
  "model": "Random Forest",
  "threshold": 0.7814
}
Single Transaction Prediction
bashPOST /predict

Request:
{
  "transaction_id": "TXN_12345",
  "features": [0.1, 0.2, ..., 0.5]  # 30 features
}

Response:
{
  "transaction_id": "TXN_12345",
  "is_fraud": false,
  "fraud_probability": 0.0234,
  "confidence": "high",
  "risk_level": "low",
  "latency_ms": 45.67,
  "timestamp": 1735689234.567
}
Batch Prediction
bashPOST /batch_predict

Request:
{
  "transactions": [
    {"transaction_id": "TXN_001", "features": [...]},
    {"transaction_id": "TXN_002", "features": [...]}
  ]
}

Response:
{
  "results": [...],
  "total_processed": 2,
  "total_time_ms": 89.45,
  "avg_time_per_transaction_ms": 44.72
}
Statistics
bashGET /stats

Response:
{
  "total_predictions": 1234,
  "fraud_detected": 15,
  "legitimate": 1219,
  "avg_latency_ms": 42.5,
  "started_at": "2025-12-31T10:00:00"
}
🧪 Testing
Run the test suite:
bashpython test_api.py
Expected output:
Testing Fraud Detection API
==================================================
1. Legitimate Transaction Test:
  ✓ Correctly identified as legitimate
  ✓ Low fraud probability (0.05%)
  ✓ Latency: 45ms

2. Fraudulent Transaction Test:
  ✓ Correctly identified as fraud
  ✓ High fraud probability (99.99%)
  ✓ Latency: 41ms
📊 Model Comparison
ModelPrecisionRecallF1-ScoreFalse Positive RateROC-AUCRandom Forest88.51%78.57%0.83240.02%0.9726XGBoost35.00%87.00%0.49420.28%0.9760Logistic Regression6.00%92.00%0.10942.56%0.9698
Winner: Random Forest with optimized threshold (0.7814)
🎯 Threshold Optimization
The system uses an optimized threshold of 0.7814 (instead of default 0.5) to achieve:

Higher precision: Fewer false positives (better customer experience)
Good recall: Still catches 78.57% of fraudulent transactions
Optimal F1: Best balance between precision and recall

You can adjust the threshold in api/app.py:
python FRAUD_THRESHOLD = 0.7814  # Lower = catch more fraud, Higher = fewer false alarms

Current Performance

Latency: 41-67ms per prediction
Throughput: ~20-30 requests/second (single worker)
Memory: ~500MB

Scaling Tips

Increase workers: gunicorn -w 8 (2-4x CPU cores)
Use caching: Redis for frequently checked accounts
Batch processing: Process multiple transactions together
Load balancing: Deploy multiple instances behind a load balancer
GPU acceleration: For deep learning models (if implemented)

🔧 Configuration
Environment variables (optional):
bashexport FRAUD_THRESHOLD=0.7814
export MODEL_PATH=../models/random_forest_model.pkl
export LOG_LEVEL=INFO
export GUNICORN_WORKERS=4
📝 Dataset Information
Credit Card Fraud Detection Dataset

Source: Kaggle
Transactions: 284,807
Fraudulent: 492 (0.172%)
Features: 30 (28 PCA-transformed + Time + Amount)
Class: 0 (Legitimate) / 1 (Fraud)

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request
