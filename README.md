# Insurance Cost Calculator

A simple tool to estimate insurance costs using patient information.

## What it does

This tool takes patient details like age, BMI, smoking status, etc. and estimates insurance costs. It includes:

- Data cleaning and processing
- Several ML models (Linear Regression, Random Forest, XGBoost)
- A web interface for cost estimation
- Cost ranges with confidence

## Dataset description

Features included:

- **age**: Age of the primary beneficiary
- **sex**: Gender (male/female)
- **bmi**: Body mass index
- **children**: Number of dependents covered by health insurance
- **smoker**: Whether the individual is a smoker (yes/no)
- **region**: Residential area (northeast, northwest, southeast, southwest)
- **charges**: Individual medical costs billed by health insurance (target)

## Files

```
├── data/
│   └── medical_insurance.csv          # Insurance data
├── notebooks/
│   └── EDA.ipynb                      # Data analysis
├── src/
│   ├── preprocess.py                  # Data cleaning
│   ├── train.py                       # Model training
│   ├── mlflow_utils.py                # Experiment tracking
│   └── app.py                         # Web app
├── models/                            # Saved models
├── requirements.txt                   # Dependencies
├── run_complete_pipeline.py          # Main script
└── README.md                          # This file
```

## Quick start

Run this to set everything up:

```bash
python run_complete_pipeline.py
```

This does:
1. Install packages
2. Process the data
3. Train models
4. Start the web app

## Manual setup

If you want to do it step by step:

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

## Model results

| Model | Accuracy | Error |
|-------|----------|-------|
| Linear Regression | 58% | 4,467 |
| Ridge Regression | 56% | 4,610 |
| Random Forest | 42% | 5,308 |
| XGBoost | 31% | 5,754 |
| Lasso Regression | 57% | 5,900 |
| Gradient Boosting | 58% | 5,820 |

## How to use

1. Open the web app
2. Enter patient details (age, gender, BMI, etc.)
3. Click "Calculate Cost"
4. See the estimated cost in rupees

The app shows the predicted cost and a range of possible values.

## EDA questions

- Univariate: distribution of charges and age, smoker counts, average BMI, policyholders by region
- Bivariate: charges vs age, smokers vs non-smokers, BMI vs charges, gender-wise averages, children vs charges
- Multivariate: smoking with age vs charges, gender/region impact for smokers, age+BMI+smoking together, obese smokers vs others
- Outliers: top charges, extreme BMI values
- Correlation: numeric features vs charges

## Results

- Cleaned and processed dataset saved as `data/medical_insurance_clean.csv`
- Trained and validated regression models with metrics logged to MLflow
- Streamlit app to explore EDA and get cost estimates
- MLflow tracking for runs, metrics, and artifacts

## Evaluation metrics

- Data preprocessing completeness and accuracy
- Model performance: RMSE, MAE, R²
- Ease of use and reliability of the Streamlit app
- Clarity and value of EDA visualizations
- Effectiveness of MLflow logging and model management

## Technical tags

Python, Data Cleaning, EDA, Feature Engineering, Machine Learning, Regression, Streamlit, MLflow

## Deliverables

- Python scripts for cleaning, feature engineering, training, and MLflow integration
- Clean transformed dataset (CSV)
- Trained regression models, tracked with MLflow
- Streamlit app with prediction and visualizations
- Documentation covering methodology, analysis, and insights
## Notes

- Costs in Indian Rupees (₹)
- Uses insurance dataset with 159 records
- Shows confidence intervals
- Smoking and high BMI increase costs
