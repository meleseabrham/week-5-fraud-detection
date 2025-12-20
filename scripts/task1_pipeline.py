import os
import sys
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from preprocessing import load_data, clean_data, merge_ip_data, feature_engineering

# Configuration
DATA_RAW = os.path.join('data', 'raw')
DATA_PROCESSED = os.path.join('data', 'processed')
MODELS_DIR = os.path.join('models')

os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def print_class_distribution(y, label=""):
    """Helper to log class distribution counts and percentages."""
    counts = y.value_counts()
    pcts = y.value_counts(normalize=True) * 100
    print(f"\n--- Class Distribution: {label} ---")
    for cls in counts.index:
        print(f"Class {cls}: {counts[cls]:,} cases ({pcts[cls]:.2f}%)")

def main():
    logger.info("Starting Task 1 Pipeline...")
    
    # ---------------------------
    # 1. Process Fraud_Data.csv
    # ---------------------------
    try:
        fraud_path = os.path.join(DATA_RAW, 'Fraud_Data.csv')
        ip_path = os.path.join(DATA_RAW, 'IpAddress_to_Country.csv')
        
        if os.path.exists(fraud_path) and os.path.exists(ip_path):
            logger.info("Processing Fraud Data...")
            fraud_df = load_data(fraud_path)
            ip_df = load_data(ip_path)
            
            # Cleaning & Geolocation
            fraud_df = clean_data(fraud_df)
            fraud_df = merge_ip_data(fraud_df, ip_df)
            fraud_df = feature_engineering(fraud_df)
            
            # Save intermediate
            intermediate_path = os.path.join(DATA_PROCESSED, 'fraud_data_engineered.csv')
            fraud_df.to_csv(intermediate_path, index=False)
            
            # Preparation for Modeling
            target = 'class'
            y = fraud_df[target]
            X = fraud_df.drop(columns=[target, 'user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address'])
            
            # Explicitly compute and print class distribution before SMOTE (Task 1b gap)
            print_class_distribution(y, "Fraud Data (Before SMOTE)")
            
            cat_features = ['source', 'browser', 'sex', 'country']
            num_features = [c for c in X.columns if c not in cat_features]
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), num_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
                ]
            )
            
            X_processed = preprocessor.fit_transform(X)
            
            # Apply SMOTE
            logger.info("Applying SMOTE to Fraud Data...")
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X_processed, y)
            
            # Explicitly compute and print class distribution after SMOTE (Task 1b gap)
            print_class_distribution(pd.Series(y_res), "Fraud Data (After SMOTE)")
            
            joblib.dump(preprocessor, os.path.join(MODELS_DIR, 'preprocessor_fraud.pkl'))
            np.save(os.path.join(DATA_PROCESSED, 'X_train_fraud.npy'), X_res)
            np.save(os.path.join(DATA_PROCESSED, 'y_train_fraud.npy'), y_res)
            
            logger.info("Fraud Data processing complete.")
        else:
            logger.warning("Fraud Data files missing at data/raw/. Skipping Fraud Data segment.")
            
    except Exception as e:
        logger.error(f"Critical failure during Fraud Data processing: {e}")

    # ---------------------------
    # 2. Process creditcard.csv
    # ---------------------------
    try:
        credit_path = os.path.join(DATA_RAW, 'creditcard.csv')
        if os.path.exists(credit_path):
            logger.info("\nProcessing Credit Card Data...")
            cc_df = load_data(credit_path)
            cc_df = clean_data(cc_df)
            
            # Explicitly compute and print class distribution before processing (Task 1b gap)
            if 'Class' in cc_df.columns:
                print_class_distribution(cc_df['Class'], "Credit Card Data (Original)")
                
                # Scaling
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                cc_df[['scaled_amount', 'scaled_time']] = scaler.fit_transform(cc_df[['Amount', 'Time']])
                cc_df = cc_df.drop(columns=['Amount', 'Time'])
                
                # Apply SMOTE for Task 1 completeness
                X_cc = cc_df.drop('Class', axis=1)
                y_cc = cc_df['Class']
                
                logger.info("Applying SMOTE to Credit Card Data...")
                sm_cc = SMOTE(random_state=42)
                X_cc_res, y_cc_res = sm_cc.fit_resample(X_cc, y_cc)
                
                # Explicitly compute and print class distribution after SMOTE (Task 1b gap)
                print_class_distribution(pd.Series(y_cc_res), "Credit Card Data (After SMOTE)")
                
                # Save processed data
                cc_df_resampled = pd.DataFrame(X_cc_res, columns=X_cc.columns)
                cc_df_resampled['Class'] = y_cc_res
                cc_df_resampled.to_csv(os.path.join(DATA_PROCESSED, 'creditcard_processed.csv'), index=False)
                
                logger.info("Credit Card Data processing complete.")
            else:
                logger.error("Column 'Class' not found in creditcard.csv")
        else:
            logger.warning("creditcard.csv missing at data/raw/. Skipping Credit Card Data segment.")
            
    except Exception as e:
        logger.error(f"Critical failure during Credit Card Data processing: {e}")

    logger.info("\nTask 1 Pipeline Finished.")

if __name__ == "__main__":
    main()
