import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import pickle

def load_pickle(filename):
    """Load pickle file"""
    with open(filename, 'rb') as file:
        return pickle.load(file)

def save_pickle(obj, filename):
    """Save object as pickle file"""
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

def plot_categorical_distribution(df, column, target='Churn', figsize=(10, 6)):
    """Plot distribution of categorical variable by target"""
    plt.figure(figsize=figsize)
    
    # Create crosstab
    ct = pd.crosstab(df[column], df[target], normalize='index') * 100
    
    # Plot
    ct.plot(kind='bar', ax=plt.gca())
    plt.title(f'{column} Distribution by {target}')
    plt.xlabel(column)
    plt.ylabel('Percentage')
    plt.legend(['No Churn', 'Churn'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_numerical_distribution(df, column, target='Churn', figsize=(12, 5)):
    """Plot distribution of numerical variable by target"""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    for value in df[target].unique():
        axes[0].hist(df[df[target] == value][column], alpha=0.7, 
                    label=f'{target}={value}', bins=20)
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{column} Distribution')
    axes[0].legend()
    
    # Box plot
    df.boxplot(column=column, by=target, ax=axes[1])
    axes[1].set_title(f'{column} by {target}')
    
    plt.tight_layout()
    plt.show()

def create_correlation_heatmap(df, figsize=(12, 10)):
    """Create correlation heatmap"""
    plt.figure(figsize=figsize)
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.show()

def encode_categorical_features(df, categorical_columns):
    """Encode categorical features using Label Encoder"""
    df_encoded = df.copy()
    label_encoders = {}
    
    for column in categorical_columns:
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df_encoded[column])
        label_encoders[column] = le
    
    return df_encoded, label_encoders

def get_feature_importance_df(model, feature_names):
    """Get feature importance as DataFrame"""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    else:
        print("Model doesn't have feature_importances_ attribute")
        return None

def plot_feature_importance(importance_df, top_n=15, figsize=(10, 8)):
    """Plot top N feature importances"""
    if importance_df is None:
        return
    
    plt.figure(figsize=figsize)
    top_features = importance_df.head(top_n)
    
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def calculate_churn_rate_by_feature(df, feature, target='Churn'):
    """Calculate churn rate by feature values"""
    churn_rate = df.groupby(feature)[target].agg(['count', 'sum', 'mean'])
    churn_rate.columns = ['total_customers', 'churned_customers', 'churn_rate']
    churn_rate['churn_percentage'] = churn_rate['churn_rate'] * 100
    
    return churn_rate.sort_values('churn_rate', ascending=False)

def create_churn_summary_report(df):
    """Create a summary report of churn patterns"""
    print("="*60)
    print("           CUSTOMER CHURN ANALYSIS REPORT")
    print("="*60)
    
    total_customers = len(df)
    churned_customers = df['Churn'].sum()
    churn_rate = churned_customers / total_customers
    
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"   Total Customers: {total_customers:,}")
    print(f"   Churned Customers: {churned_customers:,}")
    print(f"   Overall Churn Rate: {churn_rate:.2%}")
    
    # Analyze by key features
    print(f"\n🔍 CHURN ANALYSIS BY KEY FEATURES:")
    
    key_features = ['Contract', 'InternetService', 'PaymentMethod', 'tenure']
    
    for feature in key_features:
        if feature in df.columns:
            churn_by_feature = calculate_churn_rate_by_feature(df, feature)
            print(f"\n   {feature}:")
            for idx, row in churn_by_feature.head(3).iterrows():
                print(f"     {idx}: {row['churn_percentage']:.1f}% "
                      f"({row['churned_customers']}/{row['total_customers']})")

def prepare_data_for_prediction(customer_data, label_encoders, scaler):
    """Prepare new customer data for prediction"""
    # Convert to DataFrame if it's a dictionary
    if isinstance(customer_data, dict):
        df = pd.DataFrame([customer_data])
    else:
        df = customer_data.copy()
    
    # Apply label encoders
    for column, encoder in label_encoders.items():
        if column in df.columns:
            df[column] = encoder.transform(df[column])
    
    # Scale numerical features
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numerical_cols:
        if col in df.columns:
            df[col] = scaler.transform(df[[col]])
    
    return df

def validate_input_data(data, required_columns):
    """Validate input data for prediction"""
    missing_columns = []
    for col in required_columns:
        if col not in data.columns:
            missing_columns.append(col)
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

def get_model_insights(model, feature_names):
    """Get insights from the trained model"""
    insights = {}
    
    if hasattr(model, 'feature_importances_'):
        # Feature importance
        importance_dict = dict(zip(feature_names, model.feature_importances_))
        insights['top_features'] = sorted(importance_dict.items(), 
                                        key=lambda x: x[1], reverse=True)[:10]
    
    return insights

def format_prediction_output(prediction, probability, customer_id=None):
    """Format prediction output for better readability"""
    result = {
        'customer_id': customer_id,
        'churn_prediction': 'Yes' if prediction == 1 else 'No',
        'churn_probability': f"{probability:.2%}",
        'risk_level': get_risk_level(probability)
    }
    
    return result

def get_risk_level(probability):
    """Categorize customer risk level based on churn probability"""
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.6:
        return "Medium Risk"
    elif probability < 0.8:
        return "High Risk"
    else:
        return "Very High Risk"

def generate_prediction_explanation(probability, top_features):
    """Generate explanation for prediction"""
    risk_level = get_risk_level(probability)
    
    explanation = f"Customer is classified as {risk_level} for churn "
    explanation += f"with {probability:.1%} probability. "
    
    if probability > 0.5:
        explanation += "Key factors contributing to churn risk include: "
        explanation += ", ".join([f[0] for f in top_features[:3]])
    else:
        explanation += "Customer shows good retention indicators."
    
    return explanation

def save_results_to_csv(results, filename):
    """Save results to CSV file"""
    if isinstance(results, dict):
        df = pd.DataFrame([results])
    else:
        df = pd.DataFrame(results)
    
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def load_and_validate_data(filepath):
    """Load and validate dataset"""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        
        # Basic validation
        if df.empty:
            raise ValueError("Dataset is empty")
        
        if 'Churn' not in df.columns:
            raise ValueError("Target column 'Churn' not found")
        
        return df
    
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None