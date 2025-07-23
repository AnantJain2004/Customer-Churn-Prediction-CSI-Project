import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    # Load the Telco Customer Churn dataset
    df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    return df

def explore_data(df):
    # Basic data exploration
    print("Dataset Shape:", df.shape)
    print("\nColumn Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nChurn Distribution:")
    print(df['Churn'].value_counts())
    
    # Plot churn distribution
    plt.figure(figsize=(8, 6))
    df['Churn'].value_counts().plot(kind='bar')
    plt.title('Customer Churn Distribution')
    plt.xlabel('Churn')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('churn_distribution.png')
    plt.show()

def clean_data(df):
    # Clean and preprocess the data
    df_clean = df.copy()
    
    # Convert TotalCharges to numeric
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    
    # Fill missing values in TotalCharges with median
    df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median(), inplace=True)
    
    # Convert binary categorical variables to numeric
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        df_clean[col] = df_clean[col].map({'Yes': 1, 'No': 0})
    
    # Handle categorical variables with more than 2 categories
    categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                       'Contract', 'PaymentMethod']
    
    # Apply Label Encoding to categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])
        label_encoders[col] = le
    
    # Drop customerID as it's not useful for prediction
    df_clean = df_clean.drop('customerID', axis=1)
    
    return df_clean, label_encoders

def feature_engineering(df):
    # Create new features
    df_features = df.copy()
    
    # Create tenure groups
    df_features['tenure_group'] = pd.cut(df_features['tenure'], 
                                       bins=[0, 12, 24, 48, 72], 
                                       labels=['0-1 year', '1-2 years', '2-4 years', '4+ years'])
    df_features['tenure_group'] = LabelEncoder().fit_transform(df_features['tenure_group'])
    
    # Create monthly charges groups
    df_features['monthly_charges_group'] = pd.cut(df_features['MonthlyCharges'], 
                                                bins=[0, 35, 65, 95, 120], 
                                                labels=['Low', 'Medium', 'High', 'Very High'])
    df_features['monthly_charges_group'] = LabelEncoder().fit_transform(df_features['monthly_charges_group'])
    
    # Create total charges groups
    df_features['total_charges_group'] = pd.cut(df_features['TotalCharges'], 
                                              bins=[0, 1000, 3000, 5000, 8500], 
                                              labels=['Low', 'Medium', 'High', 'Very High'])
    df_features['total_charges_group'] = LabelEncoder().fit_transform(df_features['total_charges_group'])
    
    return df_features

def prepare_features(df):
    # Prepare features for model training
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    return X, y, scaler

def preprocess_pipeline():
    print("Loading data...")
    df = load_data()
    
    print("Exploring data...")
    explore_data(df)
    
    print("Cleaning data...")
    df_clean, label_encoders = clean_data(df)
    
    print("Feature engineering...")
    df_features = feature_engineering(df_clean)
    
    print("Preparing features...")
    X, y, scaler = prepare_features(df_features)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler, label_encoders

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, label_encoders = preprocess_pipeline()
    
    # Save processed data
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)