import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Add src to path
sys.path.append(os.path.abspath('src'))
from modeling import prepare_data, get_preprocessor, train_and_evaluate, cross_validate_model

def main():
    # Load data
    fraud_path = 'data/processed/fraud_data_engineered.csv'
    credit_path = 'data/processed/creditcard_processed.csv'
    
    if not os.path.exists(fraud_path) or not os.path.exists(credit_path):
        print("Processed data files not found. Please run preprocessing first.")
        return

    fraud_data = pd.read_csv(fraud_path)
    credit_data = pd.read_csv(credit_path)

    # Fraud Data Preparation
    fraud_drop = ['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address']
    X_fraud, y_fraud = prepare_data(fraud_data, 'class', fraud_drop)
    X_f_train, X_f_test, y_f_train, y_f_test = train_test_split(X_fraud, y_fraud, test_size=0.2, stratify=y_fraud, random_state=42)

    # Credit Data Preparation
    X_credit, y_credit = prepare_data(credit_data, 'Class')
    X_c_train, X_c_test, y_c_train, y_c_test = train_test_split(X_credit, y_credit, test_size=0.2, stratify=y_credit, random_state=42)

    # 1. Baseline - Logistic Regression (Already in modeling.py logic, but let's run it here)
    from sklearn.linear_model import LogisticRegression
    f_p = get_preprocessor(X_f_train)
    lr_f = Pipeline([('prep', f_p), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))])
    train_and_evaluate(lr_f, X_f_train, X_f_test, y_f_train, y_f_test, "Logistic Regression (Fraud)", plot=False)

    # 2. Ensemble - Random Forest with Tuning
    print("\n--- Tuning Random Forest ---")
    rf_pipe = Pipeline([('prep', f_p), ('clf', RandomForestClassifier(random_state=42, class_weight='balanced'))])
    param_grid = {
        'clf__n_estimators': [50, 100],
        'clf__max_depth': [5, 10]
    }
    grid_search = GridSearchCV(rf_pipe, param_grid, cv=3, scoring='average_precision', n_jobs=-1)
    grid_search.fit(X_f_train, y_f_train)
    
    print(f"Best Parameters: {grid_search.best_params_}")
    best_rf = grid_search.best_estimator_
    train_and_evaluate(best_rf, X_f_train, X_f_test, y_f_train, y_f_test, "Best Random Forest (Fraud)", plot=False)

    # 3. Cross-Validation
    cross_validate_model(best_rf, X_fraud, y_fraud)

if __name__ == "__main__":
    main()
