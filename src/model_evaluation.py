import numpy as np
import joblib
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Comprehensive model evaluation"""
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n{'='*60}")
    print(f"{model_name} Evaluation Results")
    print(f"{'='*60}")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    print(f"\nTrue Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    
    # False Positive Rate
    fpr = fp / (fp + tn)
    print(f"\nFalse Positive Rate: {fpr*100:.2f}%")
    
    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    # F1 Score
    f1 = f1_score(y_test, y_pred)
    print(f"F1 Score: {f1:.4f}")
    
    return {
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'f1_score': f1,
        'false_positive_rate': fpr
    }

def plot_confusion_matrix(cm, model_name):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'../results/{model_name}_confusion_matrix.png')
    print(f"Saved confusion matrix plot: ../results/{model_name}_confusion_matrix.png")
    plt.close()

def plot_roc_curve(y_test, y_pred_proba, model_name):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'../results/{model_name}_roc_curve.png')
    print(f"Saved ROC curve plot: ../results/{model_name}_roc_curve.png")
    plt.close()

def plot_precision_recall_curve(y_test, y_pred_proba, model_name):
    """Plot Precision-Recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.tight_layout()
    plt.savefig(f'../results/{model_name}_precision_recall.png')
    print(f"Saved precision-recall plot: ../results/{model_name}_precision_recall.png")
    plt.close()

def compare_models(results_dict):
    """Compare multiple models"""
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    
    comparison = []
    for name, results in results_dict.items():
        comparison.append({
            'Model': name,
            'ROC-AUC': results['roc_auc'],
            'F1-Score': results['f1_score'],
            'FPR': results['false_positive_rate']
        })
    
    import pandas as pd
    df_comparison = pd.DataFrame(comparison)
    print("\n", df_comparison.to_string(index=False))
    
    # Find best model (highest F1, lowest FPR)
    best_model = df_comparison.loc[df_comparison['F1-Score'].idxmax(), 'Model']
    print(f"\n{'='*60}")
    print(f"RECOMMENDED MODEL: {best_model}")
    print(f"{'='*60}")
    
    return best_model

if __name__ == "__main__":
    print("Starting Model Evaluation...")
    print("="*60)
    
    # Create results directory if it doesn't exist
    os.makedirs('../results', exist_ok=True)
    
    # Load test data
    print("\nLoading test data...")
    X_test = np.load('../data/X_test.npy')
    y_test = np.load('../data/y_test.npy')
    print(f"Test data loaded: {X_test.shape[0]} samples")
    print(f"Fraud cases in test set: {y_test.sum()} ({y_test.mean()*100:.2f}%)")
    
    # Load models
    print("\nLoading trained models...")
    models = {}
    
    model_files = {
        'Random Forest': '../models/random_forest_model.pkl',
        'XGBoost': '../models/xgboost_model.pkl',
        'Logistic Regression': '../models/logistic_regression_model.pkl'
    }
    
    for name, filepath in model_files.items():
        try:
            models[name] = joblib.load(filepath)
            print(f"✓ Loaded {name}")
        except FileNotFoundError:
            print(f"✗ Could not find {filepath}")
    
    if not models:
        print("\nError: No models found! Please run model_training.py first.")
        exit(1)
    
    # Evaluate each model
    print("\n" + "="*60)
    print("EVALUATING MODELS")
    print("="*60)
    
    results = {}
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        result = evaluate_model(model, X_test, y_test, name)
        results[name] = result
        
        # Generate plots
        print(f"\nGenerating plots for {name}...")
        plot_confusion_matrix(result['confusion_matrix'], name)
        plot_roc_curve(y_test, result['y_pred_proba'], name)
        plot_precision_recall_curve(y_test, result['y_pred_proba'], name)
        print()
    
    # Compare models
    best_model = compare_models(results)
    
    # Final summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"\nTotal models evaluated: {len(results)}")
    print(f"Best performing model: {best_model}")
    print(f"\nAll plots saved in: ../results/")
    print("\nNext steps:")
    print("1. Review the plots in the results/ directory")
    print("2. Check the model comparison table above")
    print("3. Update api/app.py to use the best model")
    print("4. Run threshold optimization if needed")
    print("="*60)