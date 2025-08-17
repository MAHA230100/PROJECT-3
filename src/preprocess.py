"""
Data preprocessing module for medical insurance cost prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy import stats
import joblib
import os


class DataPreprocessor:
    """
    A class for preprocessing medical insurance data.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        self.target_column = 'charges'
        
    def load_data(self, file_path):
        """
        Load the medical insurance dataset.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def check_missing_values(self, df):
        """
        Check for missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            dict: Missing values summary
        """
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100
        
        missing_summary = pd.DataFrame({
            'Missing_Count': missing_values,
            'Missing_Percentage': missing_percentage
        })
        
        print("Missing values summary:")
        print(missing_summary[missing_summary['Missing_Count'] > 0])
        
        return missing_summary
    
    def handle_outliers(self, df, columns, method='iqr'):
        """
        Handle outliers in numerical columns.
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (list): List of numerical columns to process
            method (str): Method to handle outliers ('iqr' or 'zscore')
            
        Returns:
            pd.DataFrame: Dataframe with outliers handled
        """
        df_clean = df.copy()
        
        for column in columns:
            if column in df.columns and df[column].dtype in ['int64', 'float64']:
                if method == 'iqr':
                    Q1 = df[column].quantile(0.25)
                    Q3 = df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Cap outliers instead of removing them
                    df_clean[column] = np.where(df_clean[column] < lower_bound, lower_bound, df_clean[column])
                    df_clean[column] = np.where(df_clean[column] > upper_bound, upper_bound, df_clean[column])
                    
                elif method == 'zscore':
                    z_scores = np.abs(stats.zscore(df[column]))
                    df_clean[column] = np.where(z_scores > 3, df[column].median(), df[column])
        
        return df_clean
    
    def encode_categorical_variables(self, df, categorical_columns):
        """
        Encode categorical variables using Label Encoding.
        
        Args:
            df (pd.DataFrame): Input dataframe
            categorical_columns (list): List of categorical columns
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical variables
        """
        df_encoded = df.copy()
        
        for column in categorical_columns:
            if column in df.columns:
                le = LabelEncoder()
                df_encoded[column] = le.fit_transform(df_encoded[column])
                self.label_encoders[column] = le
                
        return df_encoded
    
    def scale_numerical_features(self, df, numerical_columns, fit=True):
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df (pd.DataFrame): Input dataframe
            numerical_columns (list): List of numerical columns to scale
            fit (bool): Whether to fit the scaler or use existing one
            
        Returns:
            pd.DataFrame: Dataframe with scaled numerical features
        """
        df_scaled = df.copy()
        
        if fit:
            df_scaled[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])
        else:
            df_scaled[numerical_columns] = self.scaler.transform(df[numerical_columns])
            
        return df_scaled
    
    def create_features(self, df):
        """
        Create additional features for better model performance.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with additional features
        """
        df_features = df.copy()
        
        # Age groups
        df_features['age_group'] = pd.cut(df_features['age'], 
                                        bins=[0, 25, 35, 45, 55, 100], 
                                        labels=[0, 1, 2, 3, 4])
        
        # BMI categories
        df_features['bmi_category'] = pd.cut(df_features['bmi'], 
                                           bins=[0, 18.5, 25, 30, 100], 
                                           labels=[0, 1, 2, 3])
        
        # Interaction features
        df_features['age_bmi'] = df_features['age'] * df_features['bmi']
        df_features['age_children'] = df_features['age'] * df_features['children']
        
        # Convert categorical features to numeric
        df_features['age_group'] = df_features['age_group'].astype(int)
        df_features['bmi_category'] = df_features['bmi_category'].astype(int)
        
        return df_features
    
    def prepare_data(self, df, target_column='charges', test_size=0.2, random_state=42):
        """
        Complete data preparation pipeline.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of the target column
            test_size (float): Proportion of test set
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, feature_columns)
        """
        # Check for missing values
        self.check_missing_values(df)
        
        # Handle outliers in numerical columns
        numerical_cols = ['age', 'bmi', 'children', 'charges']
        df_clean = self.handle_outliers(df, numerical_cols)
        
        # Create additional features
        df_features = self.create_features(df_clean)
        
        # Encode categorical variables
        categorical_cols = ['sex', 'smoker', 'region']
        df_encoded = self.encode_categorical_variables(df_features, categorical_cols)
        
        # Define feature columns (including new features)
        feature_columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region',
                          'age_group', 'bmi_category', 'age_bmi', 'age_children']
        
        # Scale numerical features
        numerical_features = ['age', 'bmi', 'children', 'age_bmi', 'age_children']
        df_scaled = self.scale_numerical_features(df_encoded, numerical_features)
        
        # Prepare X and y
        X = df_scaled[feature_columns]
        y = df_scaled[target_column]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, feature_columns
    
    def save_preprocessor(self, file_path):
        """
        Save the preprocessor objects for later use.
        
        Args:
            file_path (str): Path to save the preprocessor
        """
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(preprocessor_data, file_path)
        print(f"Preprocessor saved to {file_path}")
    
    def load_preprocessor(self, file_path):
        """
        Load the preprocessor objects.
        
        Args:
            file_path (str): Path to the saved preprocessor
        """
        if os.path.exists(file_path):
            preprocessor_data = joblib.load(file_path)
            self.scaler = preprocessor_data['scaler']
            self.label_encoders = preprocessor_data['label_encoders']
            self.feature_columns = preprocessor_data['feature_columns']
            print(f"Preprocessor loaded from {file_path}")
        else:
            print(f"Preprocessor file {file_path} not found.")


def main():
    """
    Main function to demonstrate the preprocessing pipeline.
    """
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data
    data_path = '../data/medical_insurance.csv'
    df = preprocessor.load_data(data_path)
    
    if df is not None:
        # Prepare data
        X_train, X_test, y_train, y_test, feature_columns = preprocessor.prepare_data(df)
        
        # Save preprocessor
        preprocessor.save_preprocessor('../models/preprocessor.pkl')
        
        print("Data preprocessing completed successfully!")
        print(f"Feature columns: {feature_columns}")


if __name__ == "__main__":
    main()
