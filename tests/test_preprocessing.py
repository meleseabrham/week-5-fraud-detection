import pytest
import pandas as pd
import numpy as np
from src.preprocessing import clean_data, feature_engineering

def test_clean_data():
    # Create sample data
    data = {
        'signup_time': ['2015-01-01 00:00:00', '2015-01-01 00:00:00'], # Duplicate
        'purchase_time': ['2015-01-01 01:00:00', '2015-01-01 01:00:00'],
        'user_id': [1, 1]
    }
    df = pd.DataFrame(data)
    
    cleaned_df = clean_data(df)
    
    # Check duplicates removed
    assert len(cleaned_df) == 1
    # Check datetime conversion
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df['signup_time'])
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df['purchase_time'])

def test_feature_engineering():
    # Create sample data
    data = {
        'signup_time': pd.to_datetime(['2015-01-01 00:00:00']),
        'purchase_time': pd.to_datetime(['2015-01-01 10:00:00']),
        'device_id': ['D1'],
        'ip_address': [123.0],
        'user_id': [1]
    }
    df = pd.DataFrame(data)
    
    fe_df = feature_engineering(df)
    
    # Check new columns exist
    assert 'hour_of_day' in fe_df.columns
    assert 'day_of_week' in fe_df.columns
    assert 'time_since_signup_hours' in fe_df.columns
    assert 'device_txn_count' in fe_df.columns
    
    # Check calculations
    assert fe_df['hour_of_day'].iloc[0] == 10
    assert fe_df['time_since_signup_hours'].iloc[0] == 10.0
