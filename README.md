# Equipment Sale Price Prediction

Predicting resale prices of heavy equipment using a machine learning pipeline evaluated on RMSLE (Root Mean Squared Logarithmic Error).

## Overview

This project builds an end-to-end regression pipeline to predict equipment sale prices from historical auction/transaction data. It covers data cleaning, feature engineering, model training, and prediction blending.

## Project Structure

root/
├── data/
│ ├── train.csv
│ ├── test.csv
│ ├── metadata.csv
│ └── sample_submission.csv
├── notebook.ipynb
└── README.md


## Workflow

1. **Data Cleaning** — duplicate handling, missing value treatment, fixing the `ManufactureYear = 1001` anomaly, parsing `TransactionDate`, winsorizing `OperationalHoursMeter`, target sanity checks.
2. **Exploratory Data Analysis** — target distribution (log1p transform), numeric feature distributions, correlation heatmap, categorical breakdowns (`FunctionalClassification`), price trends by sale year, equipment age vs price.
3. **Feature Engineering** — date-based features, machine age at sale, usage intensity (log hours).
4. **Preprocessing** — `ColumnTransformer` pipeline with median imputation for numeric features and appropriate encoding for categorical features.
5. **Modeling** — baseline comparison of LightGBM, XGBoost, and CatBoost.
6. **Tuning** — hyperparameter search on LightGBM (learning rate, etc.), selecting best validation RMSLE.
7. **Blending** — weighted blend of tuned LightGBM, native XGBoost, and native CatBoost predictions.
8. **Final Prediction** — refit on full training data, inverse log-transform, generate submission file.

## Key Insights

- Target variable (`TargetValue`) is heavily right-skewed → modeled in log1p space.
- Equipment age is negatively correlated with price (older machines sell for less).
- Usage intensity (operational hours) is a strong predictor — more usage, lower resale value.
- `FunctionalClassification` (~65 categories) shows clear price separation across equipment types.

## Tools & Libraries

- Python, Pandas, NumPy, Matplotlib
- Scikit-learn (pipelines, preprocessing)
- LightGBM, XGBoost, CatBoost

## Metric

Root Mean Squared Logarithmic Error (RMSLE)

## Author

Arun — BS Data Science and Applications, IIT Madras
