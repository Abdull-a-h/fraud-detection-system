import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

def handle_imbalance(X_train, y_train, method='smote'):
    """Handle class imbalance using various techniques"""
    
    if method == 'smote':
        # SMOTE: Synthetic Minority Over-sampling
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
    elif method == 'undersample':
        # Random undersampling of majority class
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
        
    elif method == 'combined':
        # Combination of over and undersampling
        over = SMOTE(sampling_strategy=0.5, random_state=42)
        under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
        X_resampled, y_resampled = over.fit_resample(X_train, y_train)
        X_resampled, y_resampled = under.fit_resample(X_resampled, y_resampled)
    
    print(f"Original dataset shape: {X_train.shape}")
    print(f"Resampled dataset shape: {X_resampled.shape}")
    print(f"Fraud ratio after resampling: {y_resampled.mean()*100:.2f}%")
    
    return X_resampled, y_resampled

def train_random_forest(X_train, y_train):
    """Train Random Forest classifier"""
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',  # Handle imbalance
        random_state=42,
        n_jobs=-1
    )
    
    print("Training Random Forest...")
    model.fit(X_train, y_train)
    print("Training complete!")
    
    return model

def train_xgboost(X_train, y_train):
    """Train XGBoost classifier"""
    
    # Calculate scale_pos_weight for imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    print("Training XGBoost...")
    model.fit(X_train, y_train)
    print("Training complete!")
    
    return model

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression classifier"""
    
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    
    print("Training Logistic Regression...")
    model.fit(X_train, y_train)
    print("Training complete!")
    
    return model

if __name__ == "__main__":
    # Load preprocessed data
    X_train = np.load('../data/X_train.npy')
    y_train = np.load('../data/y_train.npy')
    
    # Option 1: Use SMOTE for resampling
    X_resampled, y_resampled = handle_imbalance(X_train, y_train, method='smote')
    
    # Train multiple models
    rf_model = train_random_forest(X_resampled, y_resampled)
    xgb_model = train_xgboost(X_resampled, y_resampled)
    lr_model = train_logistic_regression(X_resampled, y_resampled)
    
    # Save models
    joblib.dump(rf_model, '../models/random_forest_model.pkl')
    joblib.dump(xgb_model, '../models/xgboost_model.pkl')
    joblib.dump(lr_model, '../models/logistic_regression_model.pkl')
    
    print("Models saved successfully!")