import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import preprocess_pipeline

def train_random_forest(X_train, y_train):
    """Train Random Forest model with hyperparameter tuning"""
    print("Training Random Forest...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestClassifier(random_state=42)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model"""
    print("Training Logistic Regression...")
    
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    
    lr = LogisticRegression(random_state=42, max_iter=1000)
    
    grid_search = GridSearchCV(
        lr, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def train_svm(X_train, y_train):
    """Train SVM model"""
    print("Training SVM...")
    
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
    
    svm = SVC(random_state=42, probability=True)
    
    grid_search = GridSearchCV(
        svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    print(f"\n=== {model_name} Evaluation ===")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'], 
                yticklabels=['No Churn', 'Churn'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.show()
    
    return accuracy, y_pred, y_pred_proba

def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'{model_name} - Feature Importance')
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig(f'{model_name.lower().replace(" ", "_")}_feature_importance.png')
        plt.show()

def save_model(model, filename):
    """Save trained model"""
    with open(f'app/{filename}', 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved as {filename}")

def train_models():
    """Main function to train all models"""
    # Load preprocessed data
    print("Loading preprocessed data...")
    X_train, X_test, y_train, y_test, scaler, label_encoders = preprocess_pipeline()
    
    # Dictionary to store models and their performance
    models = {}
    results = {}
    
    # Train Random Forest
    rf_model = train_random_forest(X_train, y_train)
    models['Random Forest'] = rf_model
    accuracy, _, _ = evaluate_model(rf_model, X_test, y_test, 'Random Forest')
    results['Random Forest'] = accuracy
    plot_feature_importance(rf_model, X_train.columns, 'Random Forest')
    
    # Train Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train)
    models['Logistic Regression'] = lr_model
    accuracy, _, _ = evaluate_model(lr_model, X_test, y_test, 'Logistic Regression')
    results['Logistic Regression'] = accuracy
    
    # Train SVM
    svm_model = train_svm(X_train, y_train)
    models['SVM'] = svm_model
    accuracy, _, _ = evaluate_model(svm_model, X_test, y_test, 'SVM')
    results['SVM'] = accuracy
    
    # Find best model
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    
    print(f"\n=== Model Comparison ===")
    for model_name, accuracy in results.items():
        print(f"{model_name}: {accuracy:.4f}")
    
    print(f"\nBest Model: {best_model_name} with accuracy: {results[best_model_name]:.4f}")
    
    # Save the best model
    save_model(best_model, 'model.pkl')
    
    # Save scaler and encoders
    with open('app/scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)
    
    with open('app/label_encoders.pkl', 'wb') as file:
        pickle.dump(label_encoders, file)
    
    return best_model, scaler, label_encoders

if __name__ == "__main__":
    best_model, scaler, label_encoders = train_models()