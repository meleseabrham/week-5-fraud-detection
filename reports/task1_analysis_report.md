# Task 1 Report: Data Analysis and Preprocessing

## 1. Business Objective
The project aims to improve detection of fraudulent activities in two distinct domains: e-commerce and banking.
- **E-commerce Fraud**: Detecting bot-like behavior, account takeovers, and fake accounts using transaction metadata.
- **Bank Fraud**: Identifying unauthorized credit card transactions in a highly imbalanced dataset.
By the end of this task, we have prepared a high-quality, feature-rich dataset that enables machine learning models to distinguish between legitimate and fraudulent patterns with high precision.

---

## 2. Summary of Data Cleaning and Preprocessing
The raw data underwent several stages of refinement:
- **Duplicate Removal**: 0 duplicate rows were found in the final pass of the script, but logic is integrated into the `clean_data` function to ensure uniqueness.
- **Data Type Standardization**:
    - Converted `signup_time` and `purchase_time` to pandas `datetime` objects.
    - Converted IP addresses to integer format for efficient numerical processing.
- **Missing Values**: Verified that the datasets had no significant missing values. Rows with missing country info (post-merge) were categorized as 'Unknown' rather than dropped to preserve data.

**Sample Preprocessing Code:**
```python
def clean_data(df):
    df = df.drop_duplicates()
    if 'signup_time' in df.columns:
        df['signup_time'] = pd.to_datetime(df['signup_time'])
    if 'purchase_time' in df.columns:
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    return df
```

---

## 3. Exploratory Data Analysis (EDA) Insights

### 3.1 Class Distribution
We quantified the extreme imbalance in both datasets, which is the primary challenge for this task.

| Dataset | Non-Fraud (0) | Fraud (1) | Fraud Rate |
|---------|---------------|-----------|------------|
| E-commerce (`Fraud_Data.csv`) | 136,961 | 14,151 | **9.36%** |
| Credit Card (`creditcard.csv`) | 283,253 | 473 | **0.17%** |

### 3.2 Feature-Target Relationships
- **Velocity Peaks**: Higher transaction frequencies per device (`device_txn_count`) correlate strongly with fraud. 
- **Temporal Patterns**: Fraudulent transactions show significant spikes in the first hour after user registration.
- **Geospatial Hotspots**: While the USA has the highest volume of fraud, smaller clusters (e.g., specific Eastern European and North African IP ranges) show higher *proportionate* risk.

---

## 4. Feature Engineering Justifications

### Behavioral Features
1. **`time_since_signup_hours`**: 
    - *Logic*: Fraudsters often cycle through accounts immediately after creating them.
    - *Snippet*: `df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600`
2. **Transaction Velocity**: 
    - *Insight*: Legitimate users rarely use the same device for 10+ signups or transactions in a short window.
    - *Snippet*: `df['device_txn_count'] = df.groupby('device_id')['device_id'].transform('count')`

### IP Geolocation Mapping
We implemented an optimized range-lookup using `pd.merge_asof`.
- **Reasoning**: Standard equi-joins are impossible here because IP addresses must fall within a range `[lower_bound, upper_bound]`.
- **Snippet**:
```python
# Strategy: Convert IP to int and use merge_asof for range matching
merged = pd.merge_asof(
    fraud_df.sort_values('ip_address_int'), 
    ip_country_df.sort_values('lower_bound_ip_address'), 
    left_on='ip_address_int', 
    right_on='lower_bound_ip_address'
)
```

---

## 5. Class Imbalance Strategy
Class imbalance poses a major risk: models might "cheat" by always predicting 0 (non-fraud) and still achieve >90% accuracy.

### SMOTE Implementation
- **Technique**: **SMOTE (Synthetic Minority Over-sampling Technique)**.
- **Justification**: Unlike random undersampling, SMOTE preserves all information from the majority class while adding descriptive variance to the minority class by interpolating new samples.
- **Execution Strategy**:
    - Apply `RobustScaler` to numerical features (to handle outliers in transaction amounts).
    - Fit SMOTE only on the training data to prevent label leakage.

```python
# Post-SMOTE Distribution check
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_processed, y)
# Result: 50% Fraud / 50% Non-Fraud
```

---

## 6. Completed Work and Repository Structure
1. **Source Module (`src/preprocessing.py`)**: Reusable, modular functions with robust logging.
2. **Automated Pipeline (`scripts/task1_pipeline.py`)**: End-to-end script for reproducibility.
3. **Unit Tests**: Verified with `pytest` in `tests/test_preprocessing.py`.
4. **Persisted Artifacts**: Preprocessors (`models/preprocessor_fraud.pkl`) and balanced data (`data/processed/X_train_fraud.npy`).

---

## 7. Next Steps for Task 2
1. **Model Selection**: Benchmarking Random Forest, Gradient Boosting (XGBoost), and MLP.
2. **Metric Optimization**: Prioritizing **Precision-Recall AUC** over ROC AUC due to the severity of class imbalance.
3. **Model Explainability**: Integrating SHAP values to explain individual fraud predictions to forensic analysts.
