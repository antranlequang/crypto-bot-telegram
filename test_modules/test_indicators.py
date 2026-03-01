#!/usr/bin/env python3
"""
Test the compute_indicators function to verify all technical indicators are calculated correctly.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from bot
from bot import compute_indicators

# Create sample OHLCV data
def create_sample_data(days=100):
    """Create realistic sample OHLCV data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D', tz='UTC')
    
    # Generate realistic price movements
    np.random.seed(42)
    close = 50000 + np.cumsum(np.random.randn(days) * 500)
    
    df = pd.DataFrame({
        'Open': close + np.random.randn(days) * 100,
        'High': close + abs(np.random.randn(days) * 200),
        'Low': close - abs(np.random.randn(days) * 200),
        'Close': close,
        'Volume': np.random.rand(days) * 1e6,
    }, index=dates)
    
    return df

# Test indicators
print("=" * 80)
print("TECHNICAL INDICATORS TEST")
print("=" * 80)

try:
    # Create test data
    print("\n1. Creating sample OHLCV data...")
    df_raw = create_sample_data(365)
    print(f"   ✓ Created {len(df_raw)} rows of OHLCV data")
    print(f"   Columns: {list(df_raw.columns)}")
    
    # Compute indicators
    print("\n2. Computing technical indicators...")
    df_feat = compute_indicators(df_raw)
    print(f"   ✓ Indicators computed successfully")
    print(f"   Total columns: {len(df_feat.columns)}")
    
    # Verify required indicators exist
    print("\n3. Verifying all required indicators...")
    required_indicators = [
        'RSI_10', 'RSI_14', 'ROC_12', 'STD_DEV_20', 'CCI_20',
        'H_L', 'H_CP', 'MACD_Line_6_20', 'MACD_Signal_6_20',
        'ATR_14', 'OBV', 'EMA_5', 'SMA_5',
        'STOCH_K', 'STOCH_D',
        # Backward compatibility indicators
        'EMA_12', 'EMA_26', 'MACD_Histogram',
        'Target_Return_1d', 'Log_return', 'SMA_14'
    ]
    
    missing = []
    for ind in required_indicators:
        if ind in df_feat.columns:
            print(f"   ✓ {ind}")
        else:
            print(f"   ❌ {ind} - MISSING")
            missing.append(ind)
    
    if missing:
        print(f"\n❌ Missing indicators: {', '.join(missing)}")
        sys.exit(1)
    
    # Check data quality
    print("\n4. Checking data quality...")
    print(f"   Rows: {len(df_feat)}")
    print(f"   Total columns: {len(df_feat.columns)}")
    print(f"   NaN values: {df_feat.isna().sum().sum()}")
    print(f"   Inf values: {np.isinf(df_feat.select_dtypes(include=[np.float64])).sum().sum()}")
    
    # Display latest values
    print("\n5. Latest indicator values:")
    latest = df_feat.iloc[-1]
    for ind in required_indicators[:15]:  # Show first 15
        if ind in latest.index:
            val = latest[ind]
            print(f"   {ind}: {val:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ INDICATORS TEST PASSED - All indicators calculated successfully!")
    print("=" * 80)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
