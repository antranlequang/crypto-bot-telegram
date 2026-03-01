#!/usr/bin/env python3
"""Test MacroDataProvider"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.DEBUG)

FRED_API_KEY = os.getenv("FRED_API_KEY")

if not FRED_API_KEY:
    print("❌ FRED_API_KEY not set")
    sys.exit(1)

# Manual implementation to test
from fredapi import Fred
import pandas as pd
from datetime import datetime, timedelta

def get_macro_data_test(api_key, days=30):
    """Test version of get_macro_data"""
    fred = Fred(api_key=api_key)
    
    # Calculate start_date
    start_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    print(f"Start date: {start_date}")
    
    macro_series = {
        'sp500': 'SP500',
        'vix': 'VIXCLS',
        'dxy': 'DTWEXBGS',
        'goldprice': 'GOLDAMGBD228NLBM',
        'brentoil': 'DCOILBRENTEU',
        'dowjones': 'DJIA',
    }
    
    df_list = []
    fetched_series = []
    
    for name, series_id in macro_series.items():
        try:
            print(f"\nFetching {name} ({series_id})...")
            data = fred.get_series(series_id, observation_start=start_date)
            
            print(f"  data type: {type(data)}")
            print(f"  data is not None: {data is not None}")
            print(f"  data.empty: {data.empty if hasattr(data, 'empty') else 'N/A'}")
            print(f"  len(data): {len(data)}")
            
            if data is not None:
                print(f"  ✓ Got {len(data)} records for {name}")
                df_list.append(pd.DataFrame({name: data}))
                fetched_series.append(name)
            else:
                print(f"  ✗ data is None for {name}")
                
        except ValueError as e:
            print(f"  ValueError: {e}")
        except Exception as e:
            print(f"  Exception: {e}")
    
    print(f"\nFetched series: {fetched_series}")
    
    if not df_list:
        print("No data retrieved!")
        return pd.DataFrame()
    
    macro_df = pd.concat(df_list, axis=1)
    macro_df.index.name = 'Date'
    macro_df = macro_df.ffill()
    
    print(f"\nFinal macro_df shape: {macro_df.shape}")
    print(f"Columns: {list(macro_df.columns)}")
    print(f"\nFirst few rows:")
    print(macro_df.head())
    print(f"\nLast few rows:")
    print(macro_df.tail())
    
    return macro_df

# Test it
df = get_macro_data_test(FRED_API_KEY, days=365)
print(f"\n✓ Test complete. DataFrame shape: {df.shape}")
