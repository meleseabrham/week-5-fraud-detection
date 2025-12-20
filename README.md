# Fraud Detection Project

## Overview
This project aims to detect fraudulent transactions using machine learning. It involves analyzing two distinct datasets:
1. **Fraud_Data.csv**: E-commerce transaction data including user behavior and IP addresses.
2. **creditcard.csv**: Anonymized credit card transactions with PCA features.

## Task 1: Data Analysis and Preprocessing (COMPLETED)

### Key Achievements:
- **Data Cleaning**: Handled duplicates and corrected data types (timestamps) for both datasets.
- **Exploratory Data Analysis (EDA)**:
    - Analyzed class imbalance (0.17% fraud in CreditCard, ~9% in Fraud_Data).
    - Visualized numerical distributions and categorical fraud rates.
    - Identified key countries with high fraud rates via IP geolocation.
- **Geolocation Integration**: Successfully mapped IP addresses to countries using range-based lookups.
- **Feature Engineering**:
    - Created transaction frequency/velocity features (per device and IP).
    - Derived temporal features: `hour_of_day`, `day_of_week`.
    - Calculated behavioral features: `time_since_signup` (signup to purchase duration).
- **Data Transformation**:
    - Numerical scaling (StandardScaler/RobustScaler).
    - Categorical encoding (One-Hot Encoding).
- **Class Imbalance Handling**:
    - Applied **SMOTE** to address the minority class representation.
    - Justified SMOTE over undersampling to preserve existing non-fraud data patterns.

### Processed Data Location:
- `data/processed/`: Contains scaled, encoded, and resampled training data (`.npy` and `.csv`).
- `models/preprocessor_fraud.pkl`: Saved preprocessing pipeline for the Fraud dataset.

## Project Structure
- `data/`: Raw and processed data (ignored by git).
- `notebooks/`: EDA and preprocessing notebooks.
- `src/`: Source code for reusable preprocessing logic.
- `models/`: Saved model artifacts.
- `scripts/`: Pipeline and utility scripts.
- `tests/`: Unit tests for the code base.

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Run the Task 1 pipeline: `python scripts/task1_pipeline.py`
3. Run tests: `pytest`
