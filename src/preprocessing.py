import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    """Load data from a CSV file with error handling."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data from {file_path}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        raise

def clean_data(df):
    """
    Handle missing values, duplicates, and data type corrections with validation.
    """
    if df is None or df.empty:
        logger.error("Input DataFrame is empty or None")
        return df

    df = df.copy()

    # Remove duplicates
    initial_len = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_len:
        logger.info(f"Removed {initial_len - len(df)} duplicate rows")
    
    # Correct data types if necessary (e.g. timestamps)
    time_cols = ['signup_time', 'purchase_time']
    for col in time_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
                logger.info(f"Converted {col} to datetime")
            except Exception as e:
                logger.warning(f"Failed to convert {col} to datetime: {e}")
        
    return df

def merge_ip_data(fraud_df, ip_country_df):
    """
    Merge fraud data with IP country data with validation.
    """
    required_fraud = ['ip_address']
    required_ip = ['lower_bound_ip_address', 'upper_bound_ip_address', 'country']
    
    for col in required_fraud:
        if col not in fraud_df.columns:
            logger.error(f"Missing required column in fraud data: {col}")
            raise KeyError(f"Missing required column in fraud data: {col}")
    for col in required_ip:
        if col not in ip_country_df.columns:
            logger.error(f"Missing required column in IP country data: {col}")
            raise KeyError(f"Missing required column in IP country data: {col}")

    try:
        # Ensure IPs are integers.
        fraud_df = fraud_df.copy()
        ip_country_df = ip_country_df.copy()
        
        fraud_df['ip_address_int'] = fraud_df['ip_address'].astype(float).astype(np.int64)
        ip_country_df['lower_bound_ip_address'] = ip_country_df['lower_bound_ip_address'].astype(float).astype(np.int64)
        ip_country_df['upper_bound_ip_address'] = ip_country_df['upper_bound_ip_address'].astype(float).astype(np.int64)
        
        # Sort for merge_asof
        ip_country_df = ip_country_df.sort_values('lower_bound_ip_address')
        fraud_df = fraud_df.sort_values('ip_address_int')
        
        merged = pd.merge_asof(
            fraud_df, 
            ip_country_df, 
            left_on='ip_address_int', 
            right_on='lower_bound_ip_address'
        )
        
        merged['country'] = np.where(merged['ip_address_int'] <= merged['upper_bound_ip_address'], merged['country'], 'Unknown')
        
        cols_to_drop = ['lower_bound_ip_address', 'upper_bound_ip_address', 'ip_address_int']
        merged = merged.drop(columns=cols_to_drop)
        
        logger.info("IP-Country merge completed successfully")
        return merged
    except Exception as e:
        logger.error(f"Error during IP-Country merge: {e}")
        raise

def feature_engineering(df):
    """
    Add temporal and velocity features with validation.
    """
    df = df.copy()
    
    try:
        # Time-based features
        if 'purchase_time' in df.columns:
            df['hour_of_day'] = df['purchase_time'].dt.hour
            df['day_of_week'] = df['purchase_time'].dt.dayofweek
        
        # Duration
        if 'purchase_time' in df.columns and 'signup_time' in df.columns:
            df['time_since_signup_hours'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
        
        # Velocity Features
        if 'device_id' in df.columns:
            df['device_txn_count'] = df.groupby('device_id')['device_id'].transform('count')
        if 'ip_address' in df.columns:
            df['ip_txn_count'] = df.groupby('ip_address')['ip_address'].transform('count')
        if 'user_id' in df.columns:
            df['user_txn_count'] = df.groupby('user_id')['user_id'].transform('count')
            
        logger.info("Feature engineering completed")
        return df
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        raise
