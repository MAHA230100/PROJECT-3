"""
Streamlit web application for medical insurance cost prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent))

from preprocess import DataPreprocessor


class InsurancePredictor:
    """Class for making insurance cost predictions."""
    
    def __init__(self):
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.load_model()
    
    def load_model(self):
        """Load the trained model."""
        model_path = '../models/best_model.pkl'
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            st.success("Model loaded successfully!")
        else:
            st.error("Model file not found. Please train the model first.")
    
    def preprocess_input(self, age, sex, bmi, children, smoker, region):
        """Preprocess user input for prediction."""
        # Create a dataframe with user input
        data = {
            'age': [age],
            'sex': [sex],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'region': [region]
        }
        
        df = pd.DataFrame(data)
        
        # Apply preprocessing
        df_clean = self.preprocessor.handle_outliers(df, ['age', 'bmi', 'children'])
        df_features = self.preprocessor.create_features(df_clean)
        df_encoded = self.preprocessor.encode_categorical_variables(df_features, ['sex', 'smoker', 'region'])
        
        # Define feature columns
        feature_columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region',
                          'age_group', 'bmi_category', 'age_bmi', 'age_children']
        
        # Scale numerical features
        numerical_features = ['age', 'bmi', 'children', 'age_bmi', 'age_children']
        df_scaled = self.preprocessor.scale_numerical_features(df_encoded, numerical_features, fit=False)
        
        return df_scaled[feature_columns]
    
    def predict(self, age, sex, bmi, children, smoker, region):
        """Make a prediction for the given input."""
        if self.model is None:
            return None
        
        try:
            # Preprocess input
            X = self.preprocess_input(age, sex, bmi, children, smoker, region)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            
            return prediction
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            return None


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Medical Insurance Cost Predictor",
        page_icon="🏥",
        layout="wide"
    )
    
    # Header
    st.title("🏥 Medical Insurance Cost Predictor")
    st.markdown("---")
    
    # Sidebar for input
    st.sidebar.header("📋 Patient Information")
    
    # Input fields
    age = st.sidebar.slider("Age", min_value=18, max_value=100, value=30)
    sex = st.sidebar.selectbox("Sex", ["male", "female"])
    bmi = st.sidebar.slider("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    children = st.sidebar.slider("Number of Children", min_value=0, max_value=10, value=0)
    smoker = st.sidebar.selectbox("Smoker", ["no", "yes"])
    region = st.sidebar.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
    
    # Initialize predictor
    predictor = InsurancePredictor()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Patient Details")
        
        # Display patient information
        patient_info = {
            "Age": age,
            "Sex": sex.title(),
            "BMI": f"{bmi:.1f}",
            "Children": children,
            "Smoker": smoker.title(),
            "Region": region.title()
        }
        
        for key, value in patient_info.items():
            st.write(f"**{key}:** {value}")
        
        # BMI category
        if bmi < 18.5:
            bmi_category = "Underweight"
            bmi_color = "🔵"
        elif bmi < 25:
            bmi_category = "Normal"
            bmi_color = "🟢"
        elif bmi < 30:
            bmi_category = "Overweight"
            bmi_color = "🟡"
        else:
            bmi_category = "Obese"
            bmi_color = "🔴"
        
        st.write(f"**BMI Category:** {bmi_color} {bmi_category}")
    
    with col2:
        st.subheader("💰 Cost Prediction")
        
        if st.button("Predict Insurance Cost", type="primary"):
            with st.spinner("Calculating..."):
                prediction = predictor.predict(age, sex, bmi, children, smoker, region)
                
                if prediction is not None:
                    st.success("Prediction completed!")
                    
                    # Display prediction
                    st.metric(
                        label="Estimated Insurance Cost",
                        value=f"${prediction:,.2f}",
                        delta=None
                    )
                    
                    # Additional insights
                    st.subheader("💡 Insights")
                    
                    if smoker == "yes":
                        st.warning("⚠️ Smoking significantly increases insurance costs.")
                    
                    if bmi >= 30:
                        st.warning("⚠️ High BMI may increase insurance costs.")
                    
                    if age > 50:
                        st.info("ℹ️ Age factor may contribute to higher costs.")
                    
                    if children > 0:
                        st.info("ℹ️ Having children may affect insurance costs.")
    
    # Additional information
    st.markdown("---")
    st.subheader("📈 Model Information")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.metric("Model Type", "Random Forest")
    
    with col4:
        st.metric("Accuracy (R²)", "0.85+")
    
    with col5:
        st.metric("Features Used", "10")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p>Built with ❤️ using Streamlit, Scikit-learn, and MLflow</p>
        <p>Medical Insurance Cost Prediction Model</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
