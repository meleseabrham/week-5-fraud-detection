import pandas as pd
import numpy as np
import ipaddress

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def clean_data(df):
    """
    Handle missing values, duplicates, and data type corrections.
    """
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Correct data types if necessary (e.g. timestamps)
    if 'signup_time' in df.columns:
        df['signup_time'] = pd.to_datetime(df['signup_time'])
    if 'purchase_time' in df.columns:
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        
    return df

def merge_ip_data(fraud_df, ip_country_df):
    """
    Merge fraud data with IP country data using vectorization (merge_asof).
    """
    # Ensure IPs are integers. In the csv they are often floats or strings.
    # We force conversion to float then int to handle generic '1.23e9' or '12345.0' formats safely
    fraud_df['ip_address_int'] = fraud_df['ip_address'].astype(float).astype(int)
    
    ip_country_df['lower_bound_ip_address'] = ip_country_df['lower_bound_ip_address'].astype(float).astype(int)
    ip_country_df['upper_bound_ip_address'] = ip_country_df['upper_bound_ip_address'].astype(float).astype(int)
    
    # Sort for merge_asof
    ip_country_df = ip_country_df.sort_values('lower_bound_ip_address')
    fraud_df = fraud_df.sort_values('ip_address_int')
    
    merged = pd.merge_asof(
        fraud_df, 
        ip_country_df, 
        left_on='ip_address_int', 
        right_on='lower_bound_ip_address'
    )
    
    # Filter out mismatch (where IP > upper_bound)
    # If match is valid: ip >= lower (guaranteed by merge_asof) AND ip <= upper
    merged['country'] = np.where(merged['ip_address_int'] <= merged['upper_bound_ip_address'], merged['country'], 'Unknown')
    
    # Drop temp columns usually not needed for final model, keeping original structure + country
    # But usually we drop the bounds.
    cols_to_drop = ['lower_bound_ip_address', 'upper_bound_ip_address', 'ip_address_int']
    merged = merged.drop(columns=cols_to_drop)
    
    return merged

def feature_engineering(df):
    """
    Add temporal and velocity features.
    """
    df = df.copy()
    
    # Time-based features
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    
    # Duration
    df['time_since_signup_hours'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
    
    # Velocity Features
    # Note: Transformation should nominally be done based on training fit to avoid leakage, 
    # but for simple row-wise counts in a static historical dataset, we often calculate globally or using rolling windows.
    # Instruction says: "number of transactions per user in time windows" or "frequency".
    # Here we do simple counts.
    
    # Transactions per device
    df['device_txn_count'] = df.groupby('device_id')['device_id'].transform('count')
    
    # Transactions per IP
    df['ip_txn_count'] = df.groupby('ip_address')['ip_address'].transform('count')
    
    # Transactions per user (often 1 in this dataset, but good to have)
    df['user_txn_count'] = df.groupby('user_id')['user_id'].transform('count')
    
    return df
