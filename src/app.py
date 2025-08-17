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
        """Load the trained model and preprocessor."""
        model_path = 'models/best_model.pkl'
        preprocessor_path = 'models/preprocessor.pkl'
        
        if os.path.exists(model_path) and os.path.exists(preprocessor_path):
            try:
                self.model = joblib.load(model_path)
                preprocessor_data = joblib.load(preprocessor_path)
                self.preprocessor.scaler = preprocessor_data['scaler']
                self.preprocessor.label_encoders = preprocessor_data['label_encoders']
                # st.success("Model and preprocessor loaded successfully!")
            except Exception as e:
                # st.error(f"Error loading model: {e}")
                self.model = None
        else:
            # st.error("Model or preprocessor file not found. Please train the model first.")
            self.model = None
    
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
        try:
            # Handle outliers
            df_clean = self.preprocessor.handle_outliers(df, ['age', 'bmi', 'children'])
            
            # Create features
            df_features = self.preprocessor.create_features(df_clean)
            
            # Encode categorical variables using existing encoders
            df_encoded = df_features.copy()
            for column in ['sex', 'smoker', 'region']:
                if column in df_encoded.columns and column in self.preprocessor.label_encoders:
                    le = self.preprocessor.label_encoders[column]
                    # Handle unseen categories
                    unique_values = le.classes_
                    df_encoded[column] = df_encoded[column].map(lambda x: x if x in unique_values else unique_values[0])
                    df_encoded[column] = le.transform(df_encoded[column])
            
            # Define feature columns
            feature_columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region',
                              'age_group', 'bmi_category', 'age_bmi', 'age_children']
            
            # Scale numerical features using fitted scaler
            numerical_features = ['age', 'bmi', 'children', 'age_bmi', 'age_children']
            df_scaled = df_encoded.copy()
            df_scaled[numerical_features] = self.preprocessor.scaler.transform(df_encoded[numerical_features])
            
            return df_scaled[feature_columns]
            
        except Exception as e:
            st.error(f"Error in preprocessing: {e}")
            return None
    
    def predict(self, age, sex, bmi, children, smoker, region):
        """Make a prediction for the given input."""
        if self.model is None:
            return None
        
        try:
            # Preprocess input
            X = self.preprocess_input(age, sex, bmi, children, smoker, region)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            
            # Calculate confidence interval (simplified approach)
            # In a real scenario, you'd use the model's uncertainty estimation
            confidence_interval = prediction * 0.15  # 15% margin of error
            
            return prediction, confidence_interval
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            return None, None


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Medical Insurance Cost Predictor",
        page_icon="🏥",
        layout="wide"
    )
    
    # Header
    st.title("Medical Insurance Cost Calculator")
    st.write("Enter patient details to estimate insurance costs")
    
    # Input section
    st.header("Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        sex = st.selectbox("Gender", ["male", "female"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    
    with col2:
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        smoker = st.selectbox("Smoking Status", ["no", "yes"])
        region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
    
    # Initialize predictor
    predictor = InsurancePredictor()
    
    # Prediction section
    st.header("Cost Estimation")
    
    # Display patient summary
    st.subheader("Patient Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Age:** {age} years")
        st.write(f"**Gender:** {sex.title()}")
    
    with col2:
        st.write(f"**BMI:** {bmi:.1f}")
        st.write(f"**Children:** {children}")
    
    with col3:
        st.write(f"**Smoker:** {smoker.title()}")
        st.write(f"**Region:** {region.title()}")
    
    # BMI category
    if bmi < 18.5:
        bmi_status = "Underweight"
    elif bmi < 25:
        bmi_status = "Normal"
    elif bmi < 30:
        bmi_status = "Overweight"
    else:
        bmi_status = "Obese"
    
    st.write(f"**BMI Status:** {bmi_status}")
    
    # Prediction button
    if st.button("Calculate Insurance Cost", type="primary"):
            with st.spinner("Calculating..."):
                result = predictor.predict(age, sex, bmi, children, smoker, region)
                
                if result is not None and result[0] is not None:
                    prediction, confidence_interval = result
                    
                    # Display prediction
                    st.subheader("Estimated Cost")
                    st.metric(
                        label="Insurance Cost",
                        value=f"₹{prediction:,.2f}",
                        delta=None
                    )
                    
                    # Display confidence interval
                    lower_bound = prediction - confidence_interval
                    upper_bound = prediction + confidence_interval
                    st.write(f"**Range:** ₹{lower_bound:,.2f} - ₹{upper_bound:,.2f}")
                    st.write(f"**Margin:** ±₹{confidence_interval:,.2f}")
                    
                    # Simple insights
                    st.subheader("Notes")
                    if smoker == "yes":
                        st.write("• Smoking increases insurance costs")
                    if bmi >= 30:
                        st.write("• High BMI may affect costs")
                    if age > 50:
                        st.write("• Age is a factor in pricing")
                else:
                    st.error("Unable to calculate. Please check if model is trained.")
    
    # Model info
    st.markdown("---")
    st.write("**Model:** Linear Regression")
    st.write("**Accuracy:** ~58%")
    st.write("**Data:** Medical insurance dataset")
    
    # Footer
    st.markdown("---")
    st.write("Medical Insurance Cost Calculator")
    # st.write("Built with Python and Streamlit")


if __name__ == "__main__":
    main()
