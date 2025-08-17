# Medical Insurance Cost Calculator

A simple tool to estimate medical insurance costs based on patient information.

## What it does

This project takes patient details like age, BMI, smoking status, etc. and predicts insurance costs using machine learning. It includes:

- Data cleaning and preprocessing
- Multiple ML models (Linear Regression, Random Forest, XGBoost)
- A web interface for easy cost estimation
- Confidence intervals for predictions

## Files

```
├── data/
│   └── medical_insurance.csv          # Sample insurance data
├── notebooks/
│   └── EDA.ipynb                      # Data analysis notebook
├── src/
│   ├── preprocess.py                  # Data cleaning and feature creation
│   ├── train.py                       # Model training code
│   ├── mlflow_utils.py                # Experiment tracking
│   └── app.py                         # Web app
├── models/                            # Trained models
├── requirements.txt                   # Python packages needed
├── run_complete_pipeline.py          # Run everything at once
└── README.md                          # This file
```

## Quick start

Run this command to set everything up and start the app:

```bash
python run_complete_pipeline.py
```

This will:
1. Install required packages
2. Clean and prepare the data
3. Train the models
4. Start the web app

## Manual setup

If you want to run things step by step:

```bash
# Install packages
pip install -r requirements.txt

# Look at the data
jupyter notebook notebooks/EDA.ipynb

# Train models
python src/train.py

# Start the app
streamlit run src/app.py
```

## Model performance

| Model | Accuracy (R²) | RMSE |
|-------|---------------|------|
| Linear Regression | 58.6% | 4,467 |
| Ridge Regression | 55.9% | 4,610 |
| Random Forest | 41.5% | 5,308 |
| XGBoost | 31.2% | 5,754 |

## How to use

1. Open the web app
2. Enter patient details (age, gender, BMI, etc.)
3. Click "Calculate Insurance Cost"
4. See the estimated cost in rupees

The app shows both the predicted cost and a range of possible values.

## Notes

- Costs are shown in Indian Rupees (₹)
- The model uses a medical insurance dataset with 159 records
- Predictions include confidence intervals
- Smoking and high BMI tend to increase costs
