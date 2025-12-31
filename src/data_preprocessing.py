import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath):
    """Load and preprocess the fraud detection dataset"""
    
    # Load data
    df = pd.read_csv(filepath)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
    
    # Check for missing values
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Scale Time and Amount features
    scaler = StandardScaler()
    X['scaled_amount'] = scaler.fit_transform(X['Amount'].values.reshape(-1, 1))
    X['scaled_time'] = scaler.fit_transform(X['Time'].values.reshape(-1, 1))
    
    # Drop original Time and Amount
    X = X.drop(['Time', 'Amount'], axis=1)
    
    return X, y, scaler

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets with stratification"""
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Maintain class distribution
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Training fraud ratio: {y_train.mean()*100:.2f}%")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X, y, scaler = load_and_preprocess_data('../data/creditcard.csv')
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Save preprocessed data
    np.save('../data/X_train.npy', X_train)
    np.save('../data/X_test.npy', X_test)
    np.save('../data/y_train.npy', y_train)
    np.save('../data/y_test.npy', y_test)