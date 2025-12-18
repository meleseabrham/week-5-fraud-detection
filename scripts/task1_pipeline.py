
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from preprocessing import load_data, clean_data, merge_ip_data, feature_engineering

# Configuration
DATA_RAW = os.path.join('data', 'raw')
DATA_PROCESSED = os.path.join('data', 'processed')
MODELS_DIR = os.path.join('models')

os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def main():
    print("Starting Task 1 Pipeline...")
    
    # ---------------------------
    # 1. Process Fraud_Data.csv
    # ---------------------------
    fraud_path = os.path.join(DATA_RAW, 'Fraud_Data.csv')
    ip_path = os.path.join(DATA_RAW, 'IpAddress_to_Country.csv')
    
    if os.path.exists(fraud_path) and os.path.exists(ip_path):
        print("Loading Fraud Data...")
        fraud_df = load_data(fraud_path)
        ip_df = load_data(ip_path)
        
        # Clean
        print("Cleaning Data...")
        fraud_df = clean_data(fraud_df)
        
        # Geolocation
        print("Merging Geolocation Data...")
        fraud_df = merge_ip_data(fraud_df, ip_df)
        
        # Feature Engineering
        print("Engineering Features...")
        fraud_df = feature_engineering(fraud_df)
        
        # Save intermediate
        intermediate_path = os.path.join(DATA_PROCESSED, 'fraud_data_engineered.csv')
        fraud_df.to_csv(intermediate_path, index=False)
        print(f"Engineered data saved to {intermediate_path}")
        
        # Preparation for Modeling (Encoding/Scaling)
        target = 'class'
        drop_cols = ['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address', 'source', 'browser', 'sex', 'country'] 
        # Note: We drop categorical columns here ONLY IF using OneHotEncoder in pipeline, 
        # BUT usually we want to keep them in the dataframe if we render the preprocessor separately.
        # Let's define X and y
        
        y = fraud_df[target]
        X = fraud_df.drop(columns=[target, 'user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address'])
        
        # Define categorical and numerical
        # Remaining Categorical: source, browser, sex, country
        # Remaining Numerical: purchase_value, age, hour_of_day, day_of_week, time_since_signup_hours, device_txn_count, ip_txn_count...
        
        cat_features = ['source', 'browser', 'sex', 'country']
        num_features = [c for c in X.columns if c not in cat_features]
        
        print(f"Categorical Features: {cat_features}")
        print(f"Numerical Features: {num_features}")
        
        # Create Preprocessing Pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
            ]
        )
        
        print("Fitting Preprocessor...")
        X_processed = preprocessor.fit_transform(X)
        
        # Handle Imbalance (SMOTE)
        print("Applying SMOTE...")
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_processed, y)
        
        print(f"Original shape: {X_processed.shape}, Resampled shape: {X_res.shape}")
        
        # Save processed arrays? Or DataFrame?
        # Saving sparse matrix or array is efficient, but for visibility lets save a small sample or pickling objects.
        # Ideally we save the preprocessor and the data.
        
        joblib.dump(preprocessor, os.path.join(MODELS_DIR, 'preprocessor_fraud.pkl'))
        
        # For the sake of the assignment structure, we might want to save a CSV of the transformed data, 
        # but transformed data (one-hot) is high dim. 
        # Let's save the resampled data as numpy arrays for now.
        np.save(os.path.join(DATA_PROCESSED, 'X_train_fraud.npy'), X_res)
        np.save(os.path.join(DATA_PROCESSED, 'y_train_fraud.npy'), y_res)
        
        print("Fraud Data Processing Complete.")
        
    else:
        print("Fraud_Data.csv or IpAddress_to_Country.csv not found. Skipping...")

    # ---------------------------
    # 2. Process creditcard.csv
    # ---------------------------
    credit_path = os.path.join(DATA_RAW, 'creditcard.csv')
    if os.path.exists(credit_path):
        print("\nProcessing Credit Card Data...")
        cc_df = load_data(credit_path)
        
        # Clean
        cc_df = clean_data(cc_df)
        
        # Scaling (Amount and Time)
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        
        cc_df[['scaled_amount', 'scaled_time']] = scaler.fit_transform(cc_df[['Amount', 'Time']])
        cc_df = cc_df.drop(columns=['Amount', 'Time'])
        
        # SMOTE?
        # Usually split first provided in instructions, but we'll apply resampling to the whole text for "Data Analysis and Preprocessing" output generation logic if requested.
        # Actually standard practice is ONLY resample TRAIN data.
        # Since this is "Task 1", we are preparing the data.
        
        print("Saving Processed Credit Card Data...")
        cc_df.to_csv(os.path.join(DATA_PROCESSED, 'creditcard_processed.csv'), index=False)
        
    else:
        print("creditcard.csv not found. Skipping...")
        
    print("\nTask 1 Pipeline Finished.")

if __name__ == "__main__":
    main()
