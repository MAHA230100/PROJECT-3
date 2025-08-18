#!/usr/bin/env python3
"""
Complete Insurance Cost Prediction Pipeline

This script runs the entire pipeline:
1. Data preprocessing
2. Model training with MLflow tracking
3. Launch Streamlit app

Usage: python run_complete_pipeline.py
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def print_step(step_num, title):
    """Print a step header."""
    print(f"\nStep {step_num}: {title}")

def check_dependencies():
    """Check if all required dependencies are installed."""
    print_step(1, "Checking dependencies")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 
        'seaborn', 'streamlit', 'mlflow', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  {package} - OK")
        except ImportError:
            missing_packages.append(package)
            print(f"  {package} - Missing")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("All packages installed")
        except subprocess.CalledProcessError:
            print("Failed to install some packages. Please install manually:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    
    return True

def run_data_preprocessing():
    """Run data preprocessing step."""
    print_step(2, "Data preprocessing")
    
    try:
        # Import and run preprocessing
        sys.path.append('src')
        from preprocess import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        
        # Load data
        print("  Loading dataset...")
        df = preprocessor.load_data('data/medical_insurance.csv')
        
        if df is None:
            print("  Failed to load dataset")
            return False
        
        # Prepare data
        print("  Preparing data...")
        X_train, X_test, y_train, y_test, feature_columns = preprocessor.prepare_data(df)

        # Save a cleaned dataset for EDA
        print("  Saving cleaned dataset for EDA...")
        try:
            clean_df = preprocessor.basic_clean(df)
            os.makedirs('data', exist_ok=True)
            preprocessor.save_clean_dataset(clean_df, 'data/medical_insurance_clean.csv')
        except Exception as e:
            print(f"  Could not save cleaned dataset: {e}")
        
        # Save preprocessor
        print("  Saving preprocessor...")
        preprocessor.save_preprocessor('models/preprocessor.pkl')
        
        print("  Data preprocessing completed")
        return True
        
    except Exception as e:
        print(f"  Error in data preprocessing: {e}")
        return False

def run_model_training():
    """Run model training step."""
    print_step(3, "Model training")
    
    try:
        # Import and run training
        sys.path.append('src')
        from train import ModelTrainer
        
        trainer = ModelTrainer()
        
        # Load and prepare data
        print("  Loading preprocessed data...")
        df = trainer.preprocessor.load_data('data/medical_insurance.csv')
        
        if df is None:
            print("  Failed to load dataset")
            return False
        
        X_train, X_test, y_train, y_test, feature_columns = trainer.preprocessor.prepare_data(df)
        
        # Train models
        print("  Training models...")
        results = trainer.train_models(X_train, X_test, y_train, y_test)
        
        # Display results
        print("\n  Model Performance Results:")
        print("  " + "-" * 50)
        for result in results:
            print(f"  {result['Model']:20} | R²: {result['R2']:.4f} | RMSE: {result['RMSE']:.2f} | MAE: {result['MAE']:.2f}")
        
        # Save models
        print("\n  Saving models...")
        trainer.save_models('models')
        
        print("  Model training completed")
        return True
        
    except Exception as e:
        print(f"  Error in model training: {e}")
        return False

def run_mlflow_tracking():
    """Run MLflow experiment tracking."""
    print_step(4, "MLflow tracking")
    
    try:
        # Import MLflow utilities
        sys.path.append('src')
        from mlflow_utils import MLflowManager
        
        mlflow_manager = MLflowManager()
        
        # Get experiment runs
        print("  Getting experiment runs...")
        runs = mlflow_manager.get_experiment_runs()
        
        if not runs.empty:
            print(f"  Found {len(runs)} experiment runs")
            
            # Compare models
            print("\n  Model Comparison:")
            comparison = mlflow_manager.compare_models()
            
            if comparison is not None:
                print(comparison.to_string(index=False))
        else:
            print("  No experiment runs found yet")
        
        print("  MLflow tracking completed")
        return True
        
    except Exception as e:
        print(f"  Error in MLflow tracking: {e}")
        return False

def launch_streamlit_app():
    """Launch the Streamlit web application."""
    print_step(5, "Starting web app")
    
    try:
        print("  Starting the web app...")
        print("  URL: http://localhost:8501")
        print("  Waiting for app to start...")
        
        # Start Streamlit app in background
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "src/app.py",
            "--server.port", "8501",
            "--server.headless", "true"
        ])
        
        # Wait a bit for the app to start
        time.sleep(5)
        
        # Try to open browser
        try:
            webbrowser.open("http://localhost:8501")
            print("  Browser opened")
        except:
            print("  Please manually open: http://localhost:8501")
        
        print("\n  Setup complete!")
        print("  The web app is now running.")
        print("  Press Ctrl+C to stop the app.")
        
        # Keep the process running
        try:
            process.wait()
        except KeyboardInterrupt:
                    print("\n  Stopping the app...")
        process.terminate()
        process.wait()
        print("  App stopped.")
        
        return True
        
    except Exception as e:
        print(f"  Error launching Streamlit app: {e}")
        return False

def main():
    """Main function to run the complete pipeline."""
    print("Insurance Cost Calculator")
    print("Setting up...")
    print("This will:")
    print("1. Check packages")
    print("2. Process data")
    print("3. Train models")
    print("4. Start web app")
    
    # Check if we're in the right directory
    if not os.path.exists('data/medical_insurance.csv'):
        print("Error: data file not found")
        print("Run this from the project directory")
        return False
    
    # Step 1: Check dependencies
    if not check_dependencies():
        return False
    
    # Step 2: Data preprocessing
    if not run_data_preprocessing():
        return False
    
    # Step 3: Model training
    if not run_model_training():
        return False
    
    # Step 4: MLflow tracking
    if not run_mlflow_tracking():
        return False
    
    # Step 5: Launch Streamlit app
    if not launch_streamlit_app():
        return False
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\nSetup complete!")
        else:
            print("\nSetup failed. Check the errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nSetup stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
