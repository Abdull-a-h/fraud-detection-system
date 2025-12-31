from flask import Flask, request, jsonify
import joblib
import numpy as np
import time

app = Flask(__name__)

# Load the Random Forest model (CHANGED from XGBoost)
model = joblib.load('../models/random_forest_model.pkl')

# Use optimal threshold (CHANGED from 0.5 to 0.7814)
FRAUD_THRESHOLD = 0.7814  # Optimal F1 threshold

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'model': 'Random Forest',
        'threshold': FRAUD_THRESHOLD
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Real-time fraud prediction endpoint"""
    
    start_time = time.time()
    
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        
        fraud_probability = model.predict_proba(features)[0][1]
        is_fraud = fraud_probability > FRAUD_THRESHOLD
        
        latency = (time.time() - start_time) * 1000
        
        response = {
            'transaction_id': data.get('transaction_id', 'unknown'),
            'is_fraud': bool(is_fraud),
            'fraud_probability': float(fraud_probability),
            'confidence': 'high' if abs(fraud_probability - FRAUD_THRESHOLD) > 0.2 else 'medium',
            'risk_level': get_risk_level(fraud_probability),
            'latency_ms': round(latency, 2),
            'timestamp': time.time()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Prediction failed'
        }), 400

def get_risk_level(probability):
    """Categorize risk level based on probability"""
    if probability < 0.3:
        return 'low'
    elif probability < 0.7:
        return 'medium'
    else:
        return 'high'

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint for multiple transactions"""
    
    try:
        data = request.get_json()
        transactions = data['transactions']
        
        results = []
        for transaction in transactions:
            features = np.array(transaction['features']).reshape(1, -1)
            fraud_probability = model.predict_proba(features)[0][1]
            is_fraud = fraud_probability > FRAUD_THRESHOLD
            
            results.append({
                'transaction_id': transaction.get('transaction_id', 'unknown'),
                'is_fraud': bool(is_fraud),
                'fraud_probability': float(fraud_probability),
                'risk_level': get_risk_level(fraud_probability)
            })
        
        return jsonify({'results': results, 'total_processed': len(results)}), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Batch prediction failed'
        }), 400

if __name__ == '__main__':
    print("="*60)
    print("Starting Fraud Detection API...")
    print("="*60)
    print(f"Model: Random Forest")
    print(f"Fraud threshold: {FRAUD_THRESHOLD}")
    print(f"Expected performance:")
    print(f"  - Precision: 88.51%")
    print(f"  - Recall: 78.57%")
    print(f"  - False Positive Rate: 0.02%")
    print("="*60)
    app.run(host='0.0.0.0', port=5000, debug=False)