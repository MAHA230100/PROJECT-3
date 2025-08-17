"""
MLflow utilities for experiment tracking and model management.
"""

import mlflow
import mlflow.sklearn
import os
from datetime import datetime
import pandas as pd


class MLflowManager:
    """Manager class for MLflow operations."""
    
    def __init__(self, experiment_name="insurance_cost_prediction"):
        self.experiment_name = experiment_name
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Setup MLflow tracking."""
        # Set the tracking URI (local filesystem)
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Set the experiment
        mlflow.set_experiment(self.experiment_name)
        
        print(f"MLflow experiment set to: {self.experiment_name}")
    
    def log_model_metrics(self, model_name, metrics, parameters=None):
        """Log model metrics to MLflow."""
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            if parameters:
                for param, value in parameters.items():
                    mlflow.log_param(param, value)
            
            # Log metrics
            for metric, value in metrics.items():
                mlflow.log_metric(metric, value)
            
            print(f"Logged metrics for {model_name}")
    
    def log_model(self, model, model_name, artifact_path=None):
        """Log a trained model to MLflow."""
        if artifact_path is None:
            artifact_path = model_name.lower().replace(' ', '_')
        
        mlflow.sklearn.log_model(model, artifact_path)
        print(f"Logged model {model_name} to MLflow")
    
    def save_model_locally(self, model, model_name, models_dir="../models"):
        """Save model locally and log to MLflow."""
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        # Save locally
        model_path = os.path.join(models_dir, f"{model_name.lower().replace(' ', '_')}.pkl")
        mlflow.sklearn.save_model(model, model_path)
        
        # Log to MLflow
        self.log_model(model, model_name)
        
        print(f"Saved and logged model {model_name}")
    
    def get_experiment_runs(self):
        """Get all runs from the current experiment."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment:
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            return runs
        return pd.DataFrame()
    
    def compare_models(self):
        """Compare all models in the experiment."""
        runs = self.get_experiment_runs()
        
        if not runs.empty:
            # Extract metrics
            comparison = runs[['metrics.r2', 'metrics.rmse', 'metrics.mae']].copy()
            comparison['model'] = runs['tags.mlflow.runName']
            
            print("Model Comparison:")
            print(comparison.to_string(index=False))
            
            return comparison
        else:
            print("No runs found in the experiment.")
            return None


def log_experiment_results(model_name, model, metrics, parameters=None):
    """Convenience function to log experiment results."""
    mlflow_manager = MLflowManager()
    
    # Log metrics
    mlflow_manager.log_model_metrics(model_name, metrics, parameters)
    
    # Log model
    mlflow_manager.log_model(model, model_name)
    
    print(f"Experiment results logged for {model_name}")


if __name__ == "__main__":
    # Example usage
    mlflow_manager = MLflowManager()
    
    # Get experiment runs
    runs = mlflow_manager.get_experiment_runs()
    print(f"Found {len(runs)} runs in experiment")
    
    # Compare models
    comparison = mlflow_manager.compare_models()
