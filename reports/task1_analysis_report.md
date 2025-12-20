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

---

## 3. Exploratory Data Analysis (EDA) Insights
### Key Visualizations & Findings:
- **The Imbalance Gap**: 
    - E-commerce data: **9.36%** fraud rate.
    - Credit Card data: **0.17%** fraud rate.
    - *Action*: Heavy resampling needed for Credit Card; SMOTE suggested for both.
- **Country Risk Profiles**: High volume countries (USA, China) have high total fraud, but specific smaller countries showed higher *fraud rates* (fraud count / transaction count), indicating high-risk zones.
- **Demographic Insights**: Age and purchase value distributions were relatively similar between classes, prompting the need for more complex, engineered features.

---

## 4. Feature Engineering Choices
### Behavioral Features
1. **`time_since_signup_hours`**: Calculated as `purchase_time - signup_time`. 
    - *Insight*: Fraudulent transactions often happen almost immediately after signup (0-1 hour), whereas legitimate users have a longer lead time.
2. **Transaction Velocity**:
    - **`device_txn_count`**: Number of transactions per `device_id`.
    - **`ip_txn_count`**: Number of transactions per `ip_address`.
    - *Logic*: Fraudsters often use one high-performance device or one network to cycle through many stolen identities.

### IP Geolocation Mapping
We implemented an optimized range-lookup using `pd.merge_asof`.
- **Reasoning**: Standard joins cannot handle ranges (where `ip >= start` and `ip <= end`). 
- **Efficiency**: This vectorized approach reduced processing time from minutes (using loops) to seconds.

---

## 5. Class Imbalance Analysis & Strategy
Class imbalance poses a major risk: models might "cheat" by always predicting 0 (non-fraud) and still achieve >90% accuracy.
- **Strategy**: **SMOTE (Synthetic Minority Over-sampling Technique)**.
- **Implementation**: We generated synthetic samples for the minority class to balance the ratio exactly 50/50 in the training set.
- **Justification**: Unlike random undersampling, SMOTE preserves all information from the majority class while adding descriptive variance to the minority class.

---

## 6. Completed Work and Initial Analysis
We have successfully built:
1. **Source Module (`src/preprocessing.py`)**: Reusable, modular functions.
2. **Automated Pipeline (`scripts/task1_pipeline.py`)**: End-to-end script that cleans data, engineers features, scales variables, applies SMOTE, and saves the final preprocessor and data artifacts.
3. **Unit Tests**: Verified the cleaning and feature engineering logic with `pytest`.
4. **Persisted Artifacts**: Preprocessors are saved as `.pkl` and data as `.npy` to prevent data leakage in the modeling phase.

---

## 7. Next Steps and Key Focus Areas
1. **Task 2 (Modeling)**: Train Random Forest, XGBoost, and Logistic Regression.
2. **Evaluation Metrics**: Move beyond Accuracy to **F1-Score, Precision-Recall (PRC) curves, and AUROC**.
3. **Explainability**: Use SHAP to identify which features (like `device_txn_count`) contribute most to the fraud score.
4. **Real-time Serving**: Prepare the model for low-latency inference via a Flask API.
