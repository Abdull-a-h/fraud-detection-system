import numpy as np
import joblib
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

def calculate_metrics_at_threshold(y_test, y_pred_proba, threshold):
    """Calculate metrics at a specific threshold"""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Confusion matrix components
    tn = ((y_test == 0) & (y_pred == 0)).sum()
    fp = ((y_test == 0) & (y_pred == 1)).sum()
    fn = ((y_test == 1) & (y_pred == 0)).sum()
    tp = ((y_test == 1) & (y_pred == 1)).sum()
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'false_positive_rate': fpr,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn
    }

def find_optimal_thresholds(y_test, y_pred_proba):
    """Find optimal thresholds for different objectives"""
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Calculate F1 scores for all thresholds
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Find optimal threshold for maximum F1
    optimal_f1_idx = np.argmax(f1_scores[:-1])
    optimal_f1_threshold = thresholds[optimal_f1_idx]
    
    # Find threshold for high precision (>90%)
    high_precision_idx = np.where(precision[:-1] >= 0.90)[0]
    if len(high_precision_idx) > 0:
        high_precision_threshold = thresholds[high_precision_idx[-1]]
    else:
        high_precision_threshold = None
    
    # Find threshold for high recall (>90%)
    high_recall_idx = np.where(recall[:-1] >= 0.90)[0]
    if len(high_recall_idx) > 0:
        high_recall_threshold = thresholds[high_recall_idx[0]]
    else:
        high_recall_threshold = None
    
    return {
        'optimal_f1': optimal_f1_threshold,
        'high_precision': high_precision_threshold,
        'high_recall': high_recall_threshold,
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds,
        'f1_scores': f1_scores
    }

def plot_threshold_analysis(optimal_thresholds, model_name):
    """Plot threshold analysis"""
    precision = optimal_thresholds['precision']
    recall = optimal_thresholds['recall']
    thresholds = optimal_thresholds['thresholds']
    f1_scores = optimal_thresholds['f1_scores']
    
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Precision, Recall, F1 vs Threshold
    plt.subplot(1, 2, 1)
    plt.plot(thresholds, precision[:-1], label='Precision', linewidth=2, color='blue')
    plt.plot(thresholds, recall[:-1], label='Recall', linewidth=2, color='green')
    plt.plot(thresholds, f1_scores[:-1], label='F1 Score', linewidth=2, color='red')
    
    # Mark optimal points
    if optimal_thresholds['optimal_f1'] is not None:
        plt.axvline(optimal_thresholds['optimal_f1'], color='red', linestyle='--', 
                   label=f'Optimal F1: {optimal_thresholds["optimal_f1"]:.3f}', alpha=0.7)
    
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Metrics vs Threshold - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    
    # Plot 2: Precision-Recall Curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, linewidth=2, color='purple')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(f'../results/{model_name}_threshold_optimization.png', dpi=300)
    print(f"\nPlot saved: ../results/{model_name}_threshold_optimization.png")
    plt.close()

if __name__ == "__main__":
    print("="*70)
    print("THRESHOLD OPTIMIZATION FOR FRAUD DETECTION")
    print("="*70)
    
    # Load the best model (Random Forest)
    print("\nLoading Random Forest model...")
    model = joblib.load('../models/random_forest_model.pkl')
    
    # Load test data
    print("Loading test data...")
    X_test = np.load('../data/X_test.npy')
    y_test = np.load('../data/y_test.npy')
    
    print(f"Test samples: {len(y_test)}")
    print(f"Fraud cases: {y_test.sum()} ({y_test.mean()*100:.2f}%)")
    
    # Get probability predictions
    print("\nGenerating predictions...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Find optimal thresholds
    print("\nAnalyzing thresholds...")
    optimal_thresholds = find_optimal_thresholds(y_test, y_pred_proba)
    
    print("\n" + "="*70)
    print("THRESHOLD RECOMMENDATIONS")
    print("="*70)
    
    # Test different threshold strategies
    strategies = {
        'Default (0.5)': 0.5,
        'Optimal F1': optimal_thresholds['optimal_f1'],
        'High Precision (90%+)': optimal_thresholds['high_precision'],
        'High Recall (90%+)': optimal_thresholds['high_recall'],
        'Conservative (0.7)': 0.7,
        'Aggressive (0.3)': 0.3
    }
    
    results_summary = []
    
    for strategy_name, threshold in strategies.items():
        if threshold is None:
            print(f"\n{strategy_name}: Not achievable with this model")
            continue
        
        metrics = calculate_metrics_at_threshold(y_test, y_pred_proba, threshold)
        results_summary.append(metrics)
        
        print(f"\n{strategy_name} (Threshold: {threshold:.4f})")
        print("-" * 70)
        print(f"  Precision:           {metrics['precision']:.2%} ({metrics['true_positives']}/{metrics['true_positives'] + metrics['false_positives']} flagged are actually fraud)")
        print(f"  Recall:              {metrics['recall']:.2%} ({metrics['true_positives']}/{metrics['true_positives'] + metrics['false_negatives']} actual frauds caught)")
        print(f"  F1 Score:            {metrics['f1_score']:.4f}")
        print(f"  False Positive Rate: {metrics['false_positive_rate']:.2%} ({metrics['false_positives']} legitimate flagged as fraud)")
        print(f"  True Positives:      {metrics['true_positives']}")
        print(f"  False Positives:     {metrics['false_positives']}")
        print(f"  False Negatives:     {metrics['false_negatives']}")
    
    # Generate visualization
    plot_threshold_analysis(optimal_thresholds, 'Random Forest')
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    print("\n1. BALANCED APPROACH (Recommended for most cases):")
    print(f"   Use threshold: {optimal_thresholds['optimal_f1']:.4f}")
    print("   This maximizes the F1 score, balancing precision and recall.")
    
    print("\n2. LOW FALSE POSITIVES (Minimize customer friction):")
    if optimal_thresholds['high_precision'] is not None:
        print(f"   Use threshold: {optimal_thresholds['high_precision']:.4f}")
        print("   This ensures >90% precision, fewer false alarms.")
    else:
        print("   Use threshold: 0.7-0.8")
        print("   Higher threshold = fewer false positives, but may miss some fraud.")
    
    print("\n3. CATCH MORE FRAUD (Maximize fraud detection):")
    if optimal_thresholds['high_recall'] is not None:
        print(f"   Use threshold: {optimal_thresholds['high_recall']:.4f}")
        print("   This ensures >90% recall, catches more fraud but more false alarms.")
    else:
        print("   Use threshold: 0.3-0.4")
        print("   Lower threshold = catch more fraud, but more false positives.")
    
    print("\n4. CURRENT MODEL (0.5 threshold):")
    default_metrics = calculate_metrics_at_threshold(y_test, y_pred_proba, 0.5)
    print(f"   Precision: {default_metrics['precision']:.2%}")
    print(f"   Recall: {default_metrics['recall']:.2%}")
    print(f"   False Positives: {default_metrics['false_positives']}")
    
    print("\n" + "="*70)
    print("TO USE IN API:")
    print("="*70)
    print(f"\nUpdate api/app.py with your chosen threshold:")
    print(f"FRAUD_THRESHOLD = {optimal_thresholds['optimal_f1']:.4f}  # Optimal F1")
    print(f"# or")
    print(f"FRAUD_THRESHOLD = 0.7  # Conservative (fewer false positives)")
    print(f"# or")
    print(f"FRAUD_THRESHOLD = 0.3  # Aggressive (catch more fraud)")
    print("="*70)