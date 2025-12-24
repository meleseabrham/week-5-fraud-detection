# Task 3 Report: Model Explainability

## 1. Feature Importance Baseline
We extracted the built-in feature importance from our best-performing model (Random Forest).

### Top 10 Features (Gini Importance):
1. **`device_txn_count`** (0.416): Extremely dominant, indicating automated behavior.
2. **`time_since_signup_hours`** (0.286): The time delta between account opening and first purchase.
3. **`ip_txn_count`** (0.259): Network-level velocity.
4. **`age`** (0.0045): Marginal impact.
5. **`purchase_value`** (0.0045): Marginal impact.
6. **`hour_of_day`** (0.0041)
7. **`day_of_week`** (0.0040)
8. **`source_Direct`** (0.0017)
9. **`browser_Chrome`** (0.0010)
10. **`source_SEO`** (0.0010)

**Observation**: Over **95%** of the model's predictive power comes from just three features: transaction velocity (device/IP) and the speed of transaction after signup.

---

## 2. SHAP Analysis Insights
Global and local explanations using SHAP confirmed the built-in findings but provided deeper behavioral context.

### 2.1 Global Feature Importance (SHAP Summary Plot)
- **Velocity Features**: High values of `device_txn_count` and `ip_txn_count` are the strongest positive contributors to fraud probability. SHAP shows a "long tail" where very high counts almost guarantee a fraud flag.
- **Time Since Signup**: Lower values (transactions occurring immediately after signup) significantly push the probability toward fraud. 
- **Purchase Value & Age**: Interestingly, while they appeared in the top 5, SHAP shows their impact is much more distributed and "noisy," confirming they are less reliable indicators than behavior-based velocity.

### 2.2 Local Explanations (Force Plots)
We analyzed three critical cases:
- **True Positive (TP)**: The model correctly flagged a user where `device_txn_count` was > 1 and `time_since_signup_hours` was < 1. The SHAP force plot showed these two features literally "pushed" the score from the base value to 100% risk.
- **False Positive (FP)**: A legitimate user flagged as fraud. SHAP revealed this was due to their IP address being a shared network (high `ip_txn_count`), likely a corporate office or public Wi-Fi, which tricked the velocity threshold.
- **False Negative (FN)**: A fraudster missed by the model. SHAP showed this individual used a clean device and IP (`count = 1`) and waited a moderate amount of time (~24 hours) before purchasing, effectively mimicking a regular user's "warm-up" period.

---

## 3. Interpretation & Comparison
### 3.1 Comparison: Built-in vs. SHAP
- **Alignment**: Both methods agree perfectly on the top 3 drivers.
- **Divergence**: SHAP identified that `source_Direct` and `browser_Chrome` have non-linear impacts that Gini importance underrepresented. Specifically, certain browsers correlated with custom scripts and automated headers.

### 3.2 Key Drivers of Fraud
1. **Device Overload**: Multiple user accounts linked to a single hardware ID.
2. **Immediate Conversion**: Purchases made within the first few minutes of account creation.
3. **IP Clustering**: High-frequency transactions from the same subnet.
4. **Behavioral Deviations**: Younger profiles (lower `age`) combined with high `purchase_value`.
5. **Traffic Source**: "Direct" traffic showed slightly higher susceptibility to automated bot attacks.

---

## 4. Business Recommendations

### Recommendation 1: "Speed Bump" for Immediate Purchasers
- **Insight**: `time_since_signup_hours` is a top risk factor.
- **Action**: Implement a **1-hour "cooling-off" period** or additional 3-D Secure verification for transactions exceeding $100 if the account is less than 2 hours old. 
- **Impact**: Reduces "Sign-up and Burn" attacks.

### Recommendation 2: Device-Level Blacklisting
- **Insight**: `device_txn_count` has the highest importance (0.41).
- **Action**: Automatically flag and require manual review for any device ID that attempts transactions for more than **3 unique user accounts** within 24 hours.
- **Impact**: Blocks professional fraud farms using the same hardware.

### Recommendation 3: Smart IP Thresholding
- **Insight**: False Positives are often caused by shared IP networks.
- **Action**: Dynamically adjust `ip_txn_count` thresholds based on the **Country**. For high-risk regions (identified in Task 1), keep thresholds tight; for low-risk regions with high corporate activity, use a more relaxed "Network Score" rather than a raw count.
- **Impact**: Reduces friction for legitimate customers on office/university networks.

---

## 5. Repository Update
- SHAP Summary and Force Plots available in `reports/`.
- Training and Explanation scripts updated and pushed to branch `task-3`.
