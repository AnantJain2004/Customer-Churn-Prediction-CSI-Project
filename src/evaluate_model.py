import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)
from preprocess import preprocess_pipeline

def load_model():
    """Load the trained model"""
    with open('app/model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate various evaluation metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    
    # ROC AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    metrics['roc_auc'] = auc(fpr, tpr)
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix_final.png')
    plt.show()
    
    return cm

def plot_roc_curve(y_true, y_pred_proba):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    plt.show()

def plot_precision_recall_curve(y_true, y_pred_proba):
    """Plot Precision-Recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('precision_recall_curve.png')
    plt.show()

def analyze_predictions(y_true, y_pred, y_pred_proba):
    """Analyze predictions in detail"""
    results_df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred,
        'Probability': y_pred_proba
    })
    
    # Add prediction correctness
    results_df['Correct'] = results_df['Actual'] == results_df['Predicted']
    
    print("=== Prediction Analysis ===")
    print(f"Total predictions: {len(results_df)}")
    print(f"Correct predictions: {results_df['Correct'].sum()}")
    print(f"Incorrect predictions: {len(results_df) - results_df['Correct'].sum()}")
    
    # Analyze by prediction confidence
    high_confidence = results_df[
        (results_df['Probability'] > 0.8) | (results_df['Probability'] < 0.2)
    ]
    print(f"\nHigh confidence predictions: {len(high_confidence)}")
    print(f"High confidence accuracy: {high_confidence['Correct'].mean():.4f}")
    
    # Analyze false positives and false negatives
    false_positives = results_df[
        (results_df['Actual'] == 0) & (results_df['Predicted'] == 1)
    ]
    false_negatives = results_df[
        (results_df['Actual'] == 1) & (results_df['Predicted'] == 0)
    ]
    
    print(f"\nFalse Positives: {len(false_positives)}")
    print(f"False Negatives: {len(false_negatives)}")
    
    return results_df

def generate_evaluation_report(metrics, cm):
    """Generate a comprehensive evaluation report"""
    print("\n" + "="*50)
    print("         CUSTOMER CHURN PREDICTION")
    print("            MODEL EVALUATION REPORT")
    print("="*50)
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1-Score:  {metrics['f1_score']:.4f}")
    print(f"   ROC AUC:   {metrics['roc_auc']:.4f}")
    
    print(f"\n📈 CONFUSION MATRIX:")
    print(f"   True Negatives:  {cm[0][0]}")
    print(f"   False Positives: {cm[0][1]}")
    print(f"   False Negatives: {cm[1][0]}")
    print(f"   True Positives:  {cm[1][1]}")
    
    # Business interpretation
    print(f"\n💼 BUSINESS INTERPRETATION:")
    churn_rate = (cm[1][0] + cm[1][1]) / cm.sum()
    print(f"   Actual churn rate: {churn_rate:.2%}")
    
    if metrics['precision'] > 0.7:
        print("   ✅ Good precision - Low false positive rate")
    else:
        print("   ⚠️  Consider improving precision to reduce false alarms")
        
    if metrics['recall'] > 0.7:
        print("   ✅ Good recall - Catching most churning customers")
    else:
        print("   ⚠️  Consider improving recall to catch more churning customers")

def evaluate_model_performance():
    """Main function to evaluate model performance"""
    print("Loading data and model...")
    
    # Load preprocessed data
    X_train, X_test, y_train, y_test, scaler, label_encoders = preprocess_pipeline()
    
    # Load trained model
    model = load_model()
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    
    # Generate visualizations
    print("Generating visualizations...")
    cm = plot_confusion_matrix(y_test, y_pred, "Final Model - Confusion Matrix")
    plot_roc_curve(y_test, y_pred_proba)
    plot_precision_recall_curve(y_test, y_pred_proba)
    
    # Analyze predictions
    results_df = analyze_predictions(y_test, y_pred, y_pred_proba)
    
    # Generate comprehensive report
    generate_evaluation_report(metrics, cm)
    
    # Save results
    results_df.to_csv('prediction_results.csv', index=False)
    
    return metrics, results_df

if __name__ == "__main__":
    metrics, results_df = evaluate_model_performance()