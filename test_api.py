import requests
import numpy as np
import json

# Load test data
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

print("Testing Fraud Detection API")
print("="*50)

# Test with a legitimate transaction
legit_idx = np.where(y_test == 0)[0][0]
legit_transaction = X_test[legit_idx].tolist()

response = requests.post(
    'http://localhost:5000/predict',
    json={
        'transaction_id': 'LEGIT_TEST_001',
        'features': legit_transaction
    }
)

print("\n1. Legitimate Transaction Test:")
print(json.dumps(response.json(), indent=2))

# Test with a fraudulent transaction
fraud_idx = np.where(y_test == 1)[0][0]
fraud_transaction = X_test[fraud_idx].tolist()

response = requests.post(
    'http://localhost:5000/predict',
    json={
        'transaction_id': 'FRAUD_TEST_002',
        'features': fraud_transaction
    }
)

print("\n2. Fraudulent Transaction Test:")
print(json.dumps(response.json(), indent=2))