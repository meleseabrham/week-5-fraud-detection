# Task 1 Report: Data Analysis and Preprocessing

## 1. Business Objective & Strategic Trade-offs
The primary goal of this project is to build a robust fraud detection system for e-commerce and banking transactions. 

### 1.1 The Security vs. User Experience Balance
A critical challenge in fraud detection is managing the **"Friction-Security Trade-off"**:
- **Stringent Security**: Minimizing False Negatives (missed fraud) prevents direct financial loss but increases False Positives (legitimate users flagged as fraud). High friction leads to "false declines," frustrating customers and causing churn.
- **Optimized Experience**: Minimizing friction ensures smooth UX but increases the vulnerability to account takeovers and chargebacks.
**Goal**: Our model targets **Precision-Recall optimization** to catch 95% of high-value fraud while keeping user friction in the bottom 5th percentile.

---

## 2. Comprehensive Data Cleaning & Transformation
We implemented a modular pipeline to ensure data integrity and model compatibility.

### 2.1 Applied Data Transformations
| Feature Type | Method Used | Justification |
| :--- | :--- | :--- |
| **Numerical** | `StandardScaler` / `RobustScaler` | `RobustScaler` was used for `Amount` to mitigate the influence of extreme transaction outliers common in fraud. |
| **Categorical** | `OneHotEncoder` | Used for `browser`, `source`, `sex`, and `country` to handle non-ordinal categorical data without implying a hierarchy. |
| **Temporal** | Cyclic Encoding / Extraction | Extracted components to capture seasonality of fraud. |

### 2.2 Engineered Features List
All features below were generated in `src/preprocessing.py` to capture behavioral patterns:
1.  **`hour_of_day`**: Captures activity during unusual "dark hours" (2 AM - 4 AM).
2.  **`day_of_week`**: Identifies weekend spikes in fraudulent account signups.
3.  **`time_since_signup_hours`**: The delta between account creation and first purchase.
4.  **`device_txn_count`**: Velocity check - frequency of different `user_id`s on one `device_id`.
5.  **`ip_txn_count`**: Network velocity - multiple accounts using the same IP address.
6.  **`user_txn_count`**: Transaction frequency per user to detect automated "bot-cycling."

---

## 3. Exploratory Data Analysis (EDA) Insights

### 3.1 Class Distribution Summary
The following tables quantify the extreme class imbalance identified during EDA:

**Table 1: E-commerce Dataset Distribution**
| Metric | Count | Percentage |
| :--- | :--- | :--- |
| Legitimate (0) | 136,961 | 90.64% |
| Fraudulent (1) | 14,151 | 9.36% |
| **Total** | **151,112** | **100%** |

**Table 2: Credit Card Dataset Distribution**
| Metric | Count | Percentage |
| :--- | :--- | :--- |
| Legitimate (0) | 283,253 | 99.83% |
| Fraudulent (1) | 473 | 0.17% |
| **Total** | **283,726** | **100%** |

### 3.2 Visual Analysis Findings (Reference to Notebooks)
*Detailed univariate and bivariate plots are available in `notebooks/eda-fraud-data.ipynb` and `eda-creditcard.ipynb`.*
- **Figure 1 (Velocity Correlation)**: Count plots show that as `device_txn_count` exceeds 2, the probability of fraud increases by **400%**.
- **Figure 2 (Temporal Decay)**: Histograms of `time_since_signup` reveal a massive spike at $t \approx 0$, where 90% of fraudsters transact within seconds of account creation.
- **Figure 3 (Geospatial Risk)**: While the US has the most fraud by volume, countries like Seychelles and Turkmenistan show a **100% fraud-to-transaction ratio** in our sample, marking them as high-risk regions.

---

## 4. Class Imbalance Strategy: SMOTE
To prevent the model from becoming biased toward the majority class, we applied **SMOTE (Synthetic Minority Over-sampling Technique)**.

- **Outcome**: The training data for both datasets was balanced to a **50/50 ratio**.
- **Observation**: This allows the model to learn the decision boundary of fraudulent transactions more effectively without losing data (as would happen in undersampling).

---

## 5. Next Steps and Key Focus Areas for Task 2

### 5.1 Modeling & Hyperparameter Tuning
We will use `XGBoost`, `Random Forest`, and `Logistic Regression`. 
- **Tuning**: We will implement **Bayesian Optimization** (using `Optuna`) or `RandomizedSearchCV` to tune `max_depth`, `learning_rate`, and `scale_pos_weight`.

### 5.2 Feature Importance & Comparison
We will compare feature importance through two lenses:
1.  **Model-Native Importance**: Gini Impurity (Random Forest) / Weight (XGBoost).
2.  **SHAP (SHapley Additive exPlanations)**: A local/global consistency approach to explain *why* a specific transaction was flagged (e.g., "The high `ip_txn_count` increased risk by 0.45").

### 5.3 Anticipated Challenges & Mitigation
| Challenge | Mitigation Strategy |
| :--- | :--- |
| **Overfitting on Synthetic Data** | Evaluate on a strictly **non-SMOTE** (original distribution) validation set to ensure real-world performance. |
| **Data Leakage** | Ensure scaling and SMOTE are fitted **only** on the training folds during cross-validation. |
| **Concept Drift** | Implement monitoring for changes in the distribution of `purchase_value` over time. |

---

## 6. Repository Status
- **Source**: `src/preprocessing.py` fully modularized.
- **Tests**: Unit tests covering 100% of feature logic in `tests/test_preprocessing.py`.
- **Pipeline**: `scripts/task1_pipeline.py` reproduces the entire analysis and outputs balanced artifacts.
