# 🛡️ Real-Time Fraud Detection System

A **production-ready fraud detection system** using Machine Learning to identify fraudulent transactions **in real time** with **high accuracy**, **low false positives**, and **low latency**.

---

## 📊 Performance Metrics

| Metric | Value |
|------|------|
| **Precision** | 88.51% (highly accurate fraud flags) |
| **Recall** | 78.57% (catches most fraudulent transactions) |
| **False Positive Rate** | 0.02% (≈10 false alarms per 56,864 transactions) |
| **Latency** | \< 70 ms (real-time detection) |
| **ROC-AUC** | 0.9726 |

---

## ✨ Features

- ✅ Real-time fraud detection with **\<100ms latency**
- ✅ Multiple ML models (Random Forest, XGBoost, Logistic Regression)
- ✅ Optimized classification threshold for **minimal false positives**
- ✅ RESTful API with health checks & statistics
- ✅ Dockerized for easy deployment
- ✅ Production-ready with **Gunicorn**
- ✅ Comprehensive evaluation & visualization tools

---

## 🏗️ System Architecture

┌─────────────────┐
│ Transaction │
│ Data │
└────────┬────────┘
│
▼
┌─────────────────┐
│ Preprocessing │
│ - Scaling │
│ - Feature Eng. │
└────────┬────────┘
│
▼
┌─────────────────┐
│ Random Forest │
│ Model │
│ (Threshold: │
│ 0.7814) │
└────────┬────────┘
│
▼
┌─────────────────┐
│ Fraud / Legit │
│ Prediction │
└─────────────────┘


---

## 📁 Project Structure

fraud_detection/
├── data/
│ ├── creditcard.csv
│ ├── X_train.npy
│ ├── X_test.npy
│ ├── y_train.npy
│ └── y_test.npy
│
├── models/
│ ├── random_forest_model.pkl
│ ├── xgboost_model.pkl
│ └── logistic_regression_model.pkl
│
├── src/
│ ├── data_preprocessing.py
│ ├── model_training.py
│ ├── model_evaluation.py
│ └── threshold_optimization.py
│
├── api/
│ └── app.py
│
├── results/
│ ├── *_confusion_matrix.png
│ ├── *_roc_curve.png
│ └── *_precision_recall.png
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── test_api.py
└── README.md


---

## 🚀 Quick Start

### Prerequisites
- Python **3.9+**
- Docker (optional)
- Minimum **4GB RAM**

---

## 🔧 Installation

### 1️⃣ Clone the Repository


git clone https://github.com/Abdull-a-h/fraud-detection-system.git
cd fraud-detection-system

### 2️⃣ Create Virtual Environment

python -m venv venv

Activate:

# Linux / Mac
source venv/bin/activate

# Windows
venv\Scripts\activate

### 3️⃣ Install Dependencies

pip install -r requirements.txt

### 4️⃣ Download Dataset

Download from Kaggle – Credit Card Fraud Detection

Place creditcard.csv inside the data/ directory

## 🧠 Model Training Pipeline
### 1️⃣ Preprocess Data

cd src
python data_preprocessing.py

### 2️⃣ Train Models

python model_training.py

### 3️⃣ Evaluate Models

python model_evaluation.py

### 4️⃣ Optimize Threshold (Optional)

python threshold_optimization.py

## 🧪 Model Comparison
Model	Precision	Recall	F1-Score	False Positive Rate	ROC-AUC
Random Forest	88.51%	78.57%	0.8324	0.02%	0.9726
XGBoost	35.00%	87.00%	0.4942	0.28%	0.9760
Logistic Regression	6.00%	92.00%	0.1094	2.56%	0.9698

🏆 Winner: Random Forest with optimized threshold 0.7814

## 🎯 Threshold Optimization

The system uses a custom threshold (0.7814) instead of the default 0.5 to achieve:

Higher precision (fewer false alarms)

Good recall (still catches fraud)

Optimal F1-score

FRAUD_THRESHOLD = 0.7814

## 🌐 Running the API
Option 1: Local Development

cd api
python app.py

Option 2: Docker
docker build -t fraud-detection-api .
docker run -d -p 5000:5000 --name fraud-api fraud-detection-api

Docker Compose
docker-compose up -d

📡 API Endpoints
🔍 Health Check
GET /health


Response:

{
  "status": "healthy",
  "model": "Random Forest",
  "threshold": 0.7814
}

🔮 Single Transaction Prediction
POST /predict


Request:

{
  "transaction_id": "TXN_12345",
  "features": [0.1, 0.2, "..."]
}

📦 Batch Prediction
POST /batch_predict

📊 Statistics
GET /stats

🧪 Testing
python test_api.py


Expected output:

✓ Legitimate transaction detected
✓ Fraudulent transaction detected
✓ Latency < 50ms

⚙️ Configuration

Environment variables:

export FRAUD_THRESHOLD=0.7814
export MODEL_PATH=../models/random_forest_model.pkl
export LOG_LEVEL=INFO
export GUNICORN_WORKERS=4

📚 Dataset Information

Source: Kaggle Credit Card Fraud Dataset

Transactions: 284,807

Fraudulent: 492 (0.172%)

Features: 30 (PCA + Time + Amount)

🤝 Contributing

Contributions are welcome!

git checkout -b feature/YourFeature
git commit -m "Add feature"
git push origin feature/YourFeature


Open a Pull Request 🚀