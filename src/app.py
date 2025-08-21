"""
Streamlit web application for medical insurance cost prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
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
        model_path = r"C:\Users\Iniyavan\Documents\github\PROJECT-3\models\best_model.pkl"
        preprocessor_path = r"C:\Users\Iniyavan\Documents\github\PROJECT-3\models\preprocessor.pkl"
        
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
        page_title="Insurance Cost Predictor",
        layout="wide"
    )
    
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Intro", "EDA Analysis", "Prediction", "Conclusion"]
    )
    
    if page == "Intro":
        show_intro_page()
    elif page == "EDA Analysis":
        show_eda_analysis_page()
    elif page == "Prediction":
        show_prediction_page()
    elif page == "Conclusion":
        show_conclusion_page()

def show_intro_page():
    """Display the introduction page."""
    st.title("Medical Insurance Cost Prediction")
    
    st.write("A simple tool to estimate medical insurance costs using machine learning")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("What this does")
        st.write("""
        This tool estimates medical insurance costs using patient information. 
        It's based on real insurance data and gives you a rough idea of what costs might be.
        """)
        
        st.subheader("Features")
        st.write("• Get cost estimates for insurance")
        st.write("• Look at the data and patterns")
        st.write("• Find answers to common questions")
        st.write("• Understand what affects pricing")
        
        st.subheader("How it works")
        st.write("The tool looks at several factors:")
        st.write("• Age and gender")
        st.write("• BMI (weight/height ratio)")
        st.write("• Number of children")
        st.write("• Whether you smoke")
        st.write("• Where you live")
    
    with col2:
        st.subheader("Data Info")
        st.metric("Records", "2772")
        st.metric("Model", "Linear Regression")
        st.metric("Accuracy", "58%")
        st.metric("Factors", "10")
        

def show_eda_analysis_page():
    """Display the combined EDA and Questions page."""
    st.title("Data and Questions")
    
    # EDA Section
    st.header("Data Overview")
    st.write("Here's what the insurance data looks like.")
    
    # Load sample data for display
    try:
        import pandas as pd
        data_path = r"C:\Users\Iniyavan\Documents\github\PROJECT-3\data\medical_insurance.csv"
        if not os.path.exists(data_path):
            data_path = r"C:\Users\Iniyavan\Documents\github\PROJECT-3\data\medical_insurance.csv"
        df = pd.read_csv(data_path)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Info")
            st.write(f"**Records:** {len(df)}")
            st.write(f"**Columns:** {len(df.columns)}")
            st.write(f"**Missing data:** {df.isnull().sum().sum()}")
            
            st.subheader("Data Types")
            st.write("**Numbers:** age, bmi, children, charges")
            st.write("**Categories:** sex, smoker, region")
        
        with col2:
            st.subheader("Cost Info")
            st.write(f"**Average:** ₹{df['charges'].mean():,.2f}")
            st.write(f"**Lowest:** ₹{df['charges'].min():,.2f}")
            st.write(f"**Highest:** ₹{df['charges'].max():,.2f}")
            
            st.subheader("Age Info")
            st.write(f"**Average age:** {df['age'].mean():.1f} years")
            st.write(f"**Age range:** {df['age'].min()} - {df['age'].max()} years")
        
        # Show sample data
        st.subheader("Sample Data")
        st.dataframe(df.head(10))

        # Visual analysis tabs
        st.markdown("---")
        st.subheader("Visual Analysis")
        tab_uni, tab_bi, tab_multi, tab_out, tab_corr = st.tabs([
            "Univariate", "Bivariate", "Multivariate", "Outliers", "Correlation"
        ])

        with tab_uni:
            colu1, colu2 = st.columns(2)
            with colu1:
                st.caption("Distribution of charges")
                fig, ax = plt.subplots(figsize=(5,3))
                sns.histplot(df['charges'], kde=True, ax=ax)
                ax.set_xlabel("Charges (₹)")
                st.pyplot(fig, clear_figure=True)
            with colu2:
                st.caption("Age distribution")
                fig, ax = plt.subplots(figsize=(5,3))
                sns.histplot(df['age'], bins=20, kde=False, ax=ax, color="#4C78A8")
                ax.set_xlabel("Age")
                st.pyplot(fig, clear_figure=True)
            colu3, colu4 = st.columns(2)
            with colu3:
                st.caption("Smokers vs Non-smokers")
                fig, ax = plt.subplots(figsize=(5,3))
                sns.countplot(x='smoker', data=df, ax=ax)
                ax.set_xlabel("Smoker")
                st.pyplot(fig, clear_figure=True)
            with colu4:
                st.caption("Policyholders by region")
                fig, ax = plt.subplots(figsize=(5,3))
                sns.countplot(x='region', data=df, ax=ax)
                ax.set_xlabel("Region")
                ax.tick_params(axis='x', rotation=20)
                st.pyplot(fig, clear_figure=True)

        with tab_bi:
            colb1, colb2 = st.columns(2)
            with colb1:
                st.caption("Charges vs Age")
                fig, ax = plt.subplots(figsize=(5,3))
                sns.regplot(x='age', y='charges', data=df, ax=ax, scatter_kws={'s':10})
                ax.set_ylabel("Charges (₹)")
                st.pyplot(fig, clear_figure=True)
            with colb2:
                st.caption("Charges by Smoker")
                fig, ax = plt.subplots(figsize=(5,3))
                sns.boxplot(x='smoker', y='charges', data=df, ax=ax)
                ax.set_ylabel("Charges (₹)")
                st.pyplot(fig, clear_figure=True)
            colb3, colb4 = st.columns(2)
            with colb3:
                st.caption("Charges vs BMI")
                fig, ax = plt.subplots(figsize=(5,3))
                sns.regplot(x='bmi', y='charges', data=df, ax=ax, scatter_kws={'s':10}, color="#F58518")
                ax.set_ylabel("Charges (₹)")
                st.pyplot(fig, clear_figure=True)
            with colb4:
                st.caption("Avg charges by gender")
                fig, ax = plt.subplots(figsize=(5,3))
                sns.barplot(x='sex', y='charges', data=df, estimator=np.mean, ax=ax)
                ax.set_ylabel("Avg Charges (₹)")
                st.pyplot(fig, clear_figure=True)

        with tab_multi:
            colm1, colm2 = st.columns(2)
            with colm1:
                st.caption("Smoker vs Age: Avg charges")
                df_age_bin = df.copy()
                df_age_bin['age_bin'] = pd.cut(df_age_bin['age'], bins=[18,25,35,45,55,65], include_lowest=True)
                pivot = df_age_bin.groupby(['smoker','age_bin'])['charges'].mean().reset_index()
                fig, ax = plt.subplots(figsize=(5,3))
                for smk, sub in pivot.groupby('smoker'):
                    ax.plot(sub['age_bin'].astype(str), sub['charges'], marker='o', label=smk)
                ax.set_ylabel("Avg Charges (₹)")
                ax.legend(title='Smoker')
                ax.tick_params(axis='x', rotation=20)
                st.pyplot(fig, clear_figure=True)
            with colm2:
                st.caption("Smokers: Gender x Region (avg charges)")
                smokers = df[df['smoker']=="yes"].copy()
                if not smokers.empty:
                    heat = smokers.pivot_table(index='sex', columns='region', values='charges', aggfunc='mean')
                    fig, ax = plt.subplots(figsize=(5,3))
                    sns.heatmap(heat, annot=True, fmt='.0f', cmap='Blues', ax=ax)
                    ax.set_ylabel("")
                    ax.set_xlabel("")
                    st.pyplot(fig, clear_figure=True)
                else:
                    st.info("No smoker records to display.")
            st.caption("Age, BMI, Smoking together")
            fig, ax = plt.subplots(figsize=(6,3.5))
            sns.scatterplot(data=df, x='age', y='charges', hue='smoker', size='bmi', sizes=(10,80), ax=ax)
            ax.set_ylabel("Charges (₹)")
            st.pyplot(fig, clear_figure=True)
            # Obese smokers vs non-obese non-smokers
            obese_smokers = df[(df['bmi']>30) & (df['smoker']=="yes")]['charges'].mean()
            lean_nonsmokers = df[(df['bmi']<=30) & (df['smoker']=="no")]['charges'].mean()
            if not np.isnan(obese_smokers) and not np.isnan(lean_nonsmokers):
                st.write(f"Avg charges: Obese smokers ₹{obese_smokers:,.0f} vs Non-obese non-smokers ₹{lean_nonsmokers:,.0f}")

        with tab_out:
            st.caption("Top 10 highest charges")
            st.dataframe(df.sort_values('charges', ascending=False).head(10))
            st.caption("Charges boxplot")
            fig, ax = plt.subplots(figsize=(6,3))
            sns.boxplot(x=df['charges'], ax=ax)
            ax.set_xlabel("Charges (₹)")
            st.pyplot(fig, clear_figure=True)
            st.caption("BMI boxplot")
            fig, ax = plt.subplots(figsize=(6,3))
            sns.boxplot(x=df['bmi'], ax=ax, color="#72B7B2")
            ax.set_xlabel("BMI")
            st.pyplot(fig, clear_figure=True)

        with tab_corr:
            st.caption("Correlation between numeric features")
            num_cols = df.select_dtypes(include=[np.number])
            fig, ax = plt.subplots(figsize=(6,4))
            sns.heatmap(num_cols.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            st.pyplot(fig, clear_figure=True)
        
    except Exception as e:
        st.error(f"Could not load data: {e}")
    
    # Questions Section
    st.header("Common Questions")
    
    with st.expander("What affects insurance costs?"):
        st.write("""
        Several things affect insurance costs:
        • **Age** - Older people usually pay more
        • **BMI** - Higher BMI means higher costs
        • **Smoking** - Smokers pay a lot more
        • **Region** - Where you live matters
        • **Children** - More kids = higher costs
        • **Gender** - Some differences between men and women
        """)
    
    with st.expander("How accurate is this?"):
        st.write("""
        The model is about 58% accurate. This means:
        • Good for rough estimates
        • Real costs might be ±15% different
        • Works best for typical cases
        • Extreme cases might be off more
        """)
    
    with st.expander("Why do smokers pay more?"):
        st.write("""
        Smokers pay more because:
        • Higher health risks
        • More likely to need medical care
        • Insurance companies see them as risky
        • They adjust prices based on risk
        """)
    
    with st.expander("What is BMI and why does it matter?"):
        st.write("""
        BMI = weight (kg) / height (m)²
        
        It matters because:
        • High BMI = health risks
        • Linked to diabetes, heart problems
        • Insurance companies use it for pricing
        • Normal BMI (18.5-25) gets better rates
        """)
    
    with st.expander("Can I get a better estimate?"):
        st.write("""
        For better estimates:
        • Contact insurance companies directly
        • Get quotes from multiple companies
        • This tool is just a rough guide
        • Real rates depend on specific policies
        """)

def show_conclusion_page():
    """Display the conclusion page."""
    st.title("Summary")
    
    st.header("What we built")
    st.write("""
    A simple tool that estimates insurance costs using patient information. 
    It shows how machine learning can help predict healthcare costs.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Main findings")
        st.write("""
        • **Age** - Costs go up with age
        • **Smoking** - Smokers pay much more
        • **BMI** - Higher BMI = higher costs
        • **Region** - Different areas have different costs
        • **Model** - Linear Regression works best (58% accuracy)
        """)
        
        st.subheader("What we did")
        st.write("""
        • Built a data processing pipeline
        • Trained several ML models
        • Used MLflow to track experiments
        • Made a web app with Streamlit
        • Got decent prediction accuracy
        """)
    
    with col2:
        st.subheader("Model results")
        st.metric("Best Model", "Linear Regression")
        st.metric("Accuracy", "58%")
        st.metric("Error", "₹5,847")
        st.metric("Avg Error", "₹4,123")
        
        st.subheader("Data info")
        st.metric("Records", "159")
        st.metric("Features", "10")
        st.metric("Age Range", "18-64")
        st.metric("Cost Range", "₹1,121 - ₹63,770")
    
    st.header("What this means")
    
    st.subheader("For insurance companies")
    st.write("""
    • Age and smoking are key pricing factors
    • BMI is a good risk indicator
    • Regional pricing can be optimized
    • ML models can improve pricing
    """)
    
    st.subheader("For customers")
    st.write("""
    • Quitting smoking saves money
    • Healthy BMI lowers costs
    • Age affects pricing
    • Location matters for costs
    """)
    
    st.subheader("For healthcare")
    st.write("""
    • Preventive care reduces insurance costs
    • Smoking programs save money
    • BMI management helps patients
    • Regional analysis is useful
    """)
    
    st.header("Future work")
    st.write("""
    • Add more health factors (medical history, family history)
    • Try better models (Neural Networks, etc.)
    • Use real-time data
    • Make it more personalized
    • Build a mobile app
    """)
    
    st.header("Impact")
    st.write("""
    This shows how machine learning can help with insurance pricing. 
    It gives a starting point for better cost prediction models.
    """)
    
    st.markdown("---")
    st.write("**Thanks for checking out the Insurance Cost Calculator!**")
    st.write("A simple tool showing how data science can help with healthcare costs.")

def show_prediction_page():
    """Display the prediction page."""
    st.title("Cost Calculator")
    st.write("Enter your details to get an insurance cost estimate")
    
    # Input section
    st.header("Your Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        sex = st.selectbox("Gender", ["male", "female"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    
    with col2:
        children = st.number_input("Children", min_value=0, max_value=10, value=0)
        smoker = st.selectbox("Smoker", ["no", "yes"])
        region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
    
    # Initialize predictor
    predictor = InsurancePredictor()
    
    # Prediction section
    st.header("Cost Estimate")
    
    # Display patient summary
    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Age:** {age}")
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
    if st.button("Calculate Cost", type="primary"):
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
                        st.write("• Smoking increases costs")
                    if bmi >= 30:
                        st.write("• High BMI affects costs")
                    if age > 50:
                        st.write("• Age affects pricing")
                else:
                    st.error("Could not calculate. Check if model is trained.")
    
    # Model info
    st.markdown("---")
    st.write("**Model:** Linear Regression")
    st.write("**Accuracy:** ~58%")
    st.write("**Data:** Insurance dataset")
    
    # Footer
    st.markdown("---")
    st.write("Insurance Cost Calculator")


if __name__ == "__main__":
    main()
