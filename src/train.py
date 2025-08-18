"""
Model training module for medical insurance cost prediction.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib
import os
import numpy as np
from scipy import stats

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

from preprocess import DataPreprocessor
from mlflow_utils import log_experiment_results


class ModelTrainer:
    """A class for training insurance cost prediction models."""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = float('-inf')
        self.preprocessor = DataPreprocessor()
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models and select the best one."""
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.001, max_iter=10000),
            'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42)
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=42)
        
        results = []
        
        for model_name, model in models.items():
            print(f"Training {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate confidence intervals for predictions
            residuals = y_test - y_pred
            std_residuals = np.std(residuals)
            confidence_interval = 1.96 * std_residuals  # 95% confidence interval
            
            metrics = {
                'Model': model_name,
                'MSE': float(mse),
                'RMSE': float(rmse),
                'MAE': float(mae),
                'R2': float(r2),
                'Confidence_Interval': float(confidence_interval)
            }
            
            results.append(metrics)
            self.models[model_name] = model
            
            print(f"{model_name} - R2: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")

            # Log to MLflow
            try:
                params = getattr(model, 'get_params', lambda: {})()
                # Only keep simple serializable params
                serializable_params = {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))}
                log_experiment_results(model_name, model, {
                    'r2': float(r2),
                    'rmse': float(rmse),
                    'mae': float(mae)
                }, parameters=serializable_params)
            except Exception as e:
                print(f"Could not log {model_name} to MLflow: {e}")
            
            # Update best model
            if r2 > self.best_score:
                self.best_score = r2
                self.best_model = model
        
        return results
    
    def save_models(self, models_dir='../models'):
        """Save trained models."""
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        for model_name, model in self.models.items():
            model_path = os.path.join(models_dir, f"{model_name.lower().replace(' ', '_')}.pkl")
            joblib.dump(model, model_path)
            print(f"Saved {model_name} to {model_path}")
        
        if self.best_model is not None:
            best_model_path = os.path.join(models_dir, 'best_model.pkl')
            joblib.dump(self.best_model, best_model_path)
            print(f"Saved best model to {best_model_path}")
    
    def get_prediction_confidence(self, model, X_test, y_test):
        """Calculate confidence intervals for model predictions."""
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        std_residuals = np.std(residuals)
        confidence_interval = 1.96 * std_residuals  # 95% confidence interval
        return confidence_interval


def main():
    """Main function to run the training pipeline."""
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Load and prepare data
    data_path = '../data/medical_insurance.csv'
    df = trainer.preprocessor.load_data(data_path)
    
    if df is not None:
        X_train, X_test, y_train, y_test, feature_columns = trainer.preprocessor.prepare_data(df)
        
        # Train models
        results = trainer.train_models(X_train, X_test, y_train, y_test)
        
        # Display results
        results_df = pd.DataFrame(results)
        print("\nModel Evaluation Results:")
        print(results_df.to_string(index=False))
        
        # Save models
        trainer.save_models()
        
        print(f"\nBest model: {trainer.best_score:.4f} R2 score")
        print("Training completed successfully!")


if __name__ == "__main__":
    main()
