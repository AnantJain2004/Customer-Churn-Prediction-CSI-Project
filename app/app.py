import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from utils import format_prediction_output, get_risk_level, generate_prediction_explanation

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model_and_preprocessors():
    """Load trained model and preprocessors"""
    try:
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        
        with open('label_encoders.pkl', 'rb') as file:
            label_encoders = pickle.load(file)
        
        return model, scaler, label_encoders
    except FileNotFoundError:
        st.error("Model files not found. Please train the model first.")
        return None, None, None

def create_input_features():
    """Create input widgets for customer features"""
    st.sidebar.header("Customer Information")
    
    # Customer demographics
    st.sidebar.subheader("Demographics")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
    dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
    
    # Account information
    st.sidebar.subheader("Account Information")
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.sidebar.selectbox(
        "Payment Method", 
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
    total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 8500.0, monthly_charges * tenure)
    
    # Services
    st.sidebar.subheader("Services")
    phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.sidebar.selectbox(
        "Multiple Lines", 
        ["No phone service", "No", "Yes"] if phone_service == "No" else ["No", "Yes"]
    )
    
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    if internet_service != "No":
        online_security = st.sidebar.selectbox("Online Security", ["No", "Yes"])
        online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes"])
        device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes"])
        tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
        streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes"])
        streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes"])
    else:
        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        tech_support = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"
    
    # Create feature dictionary
    features = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    return features

def preprocess_input(features, label_encoders, scaler):
    """Preprocess input features for prediction"""
    # Convert to DataFrame
    df = pd.DataFrame([features])
    
    # Apply label encoders
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                          'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                          'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                          'PaperlessBilling', 'PaymentMethod']
    
    for col in categorical_columns:
        if col in label_encoders:
            df[col] = label_encoders[col].transform(df[col])
    
    # Create additional features (matching training pipeline)
    df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], 
                               labels=[0, 1, 2, 3]).astype(int)
    df['monthly_charges_group'] = pd.cut(df['MonthlyCharges'], bins=[0, 35, 65, 95, 120], 
                                        labels=[0, 1, 2, 3]).astype(int)
    df['total_charges_group'] = pd.cut(df['TotalCharges'], bins=[0, 1000, 3000, 5000, 8500], 
                                      labels=[0, 1, 2, 3]).astype(int)
    
    # Scale numerical features
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    return df

def create_gauge_chart(probability):
    """Create a gauge chart for churn probability"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Probability (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 80], 'color': "orange"},
                {'range': [80, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def display_feature_importance(model, feature_names):
    """Display feature importance"""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True).tail(15)
        
        fig = px.bar(importance_df, x='importance', y='feature', 
                     orientation='h', title="Top 15 Feature Importances")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main Streamlit application"""
    st.title("🔮 Customer Churn Prediction")
    st.markdown("Predict whether a customer will churn based on their profile and usage patterns")
    
    # Load model and preprocessors
    model, scaler, label_encoders = load_model_and_preprocessors()
    
    if model is None:
        st.stop()
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Prediction Results")
        
        # Get input features
        features = create_input_features()
        
        # Make prediction button
        if st.button("🔍 Predict Churn", type="primary"):
            try:
                # Preprocess input
                processed_features = preprocess_input(features, label_encoders, scaler)
                
                # Make prediction
                prediction = model.predict(processed_features)[0]
                probability = model.predict_proba(processed_features)[0][1]
                
                # Display results
                risk_level = get_risk_level(probability)
                
                # Create metric cards
                col_metric1, col_metric2, col_metric3 = st.columns(3)
                
                with col_metric1:
                    st.metric(
                        "Churn Prediction", 
                        "Will Churn" if prediction == 1 else "Will Stay",
                        delta=f"{probability:.1%} probability"
                    )
                
                with col_metric2:
                    st.metric("Risk Level", risk_level)
                
                with col_metric3:
                    retention_score = (1 - probability) * 100
                    st.metric("Retention Score", f"{retention_score:.0f}/100")
                
                # Display gauge chart
                fig = create_gauge_chart(probability)
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.subheader("💡 Recommendations")
                if probability > 0.7:
                    st.error("🚨 High churn risk! Immediate action recommended.")
                    st.write("**Suggested Actions:**")
                    st.write("- Offer retention discount or incentive")
                    st.write("- Schedule a personal call with customer service")
                    st.write("- Provide upgrade options or better service plans")
                elif probability > 0.4:
                    st.warning("⚠️ Medium churn risk. Monitor closely.")
                    st.write("**Suggested Actions:**")
                    st.write("- Send targeted retention email campaign")
                    st.write("- Offer loyalty rewards or benefits")
                    st.write("- Conduct customer satisfaction survey")
                else:
                    st.success("✅ Low churn risk. Customer likely to stay.")
                    st.write("**Suggested Actions:**")
                    st.write("- Continue providing excellent service")
                    st.write("- Consider upselling opportunities")
                    st.write("- Use as reference for customer testimonials")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    with col2:
        st.header("📈 Model Insights")
        
        # Display feature importance
        if hasattr(model, 'feature_importances_'):
            # Get feature names (this should match your training features)
            feature_names = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                           'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                           'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                           'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                           'MonthlyCharges', 'TotalCharges', 'tenure_group', 'monthly_charges_group',
                           'total_charges_group']
            
            display_feature_importance(model, feature_names)
        
        # Model information
        st.subheader("🤖 Model Information")
        st.info(f"**Model Type:** {type(model).__name__}")
        st.info("**Training Features:** 22 features including customer demographics, account info, and services")
        st.info("**Performance:** Optimized for balanced precision and recall")

if __name__ == "__main__":
    main()