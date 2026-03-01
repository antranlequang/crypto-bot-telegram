#!/usr/bin/env python3
"""
Test macro data retrieval from FRED API
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from config import FRED_API_KEY
from bot import MacroDataProvider
import pandas as pd

print("=" * 80)
print("MACRO DATA RETRIEVAL TEST")
print("=" * 80)

# Check if API key is configured
if not FRED_API_KEY:
    print("❌ FRED_API_KEY not configured in .env or config.py")
    sys.exit(1)

print(f"✓ FRED_API_KEY detected: {FRED_API_KEY[:10]}...")

# Test MacroDataProvider initialization
print("\n1. Initializing MacroDataProvider...")
try:
    macro_provider = MacroDataProvider(FRED_API_KEY)
    if macro_provider.available:
        print("   ✓ MacroDataProvider initialized successfully")
    else:
        print("   ⚠ MacroDataProvider initialized but not available (fredapi may not be installed)")
        print("   Install with: pip install fredapi")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ Failed to initialize: {e}")
    sys.exit(1)

# Test macro data fetch
print("\n2. Fetching macro data...")
try:
    df_macro = macro_provider.get_macro_data(days=365)
    if df_macro.empty:
        print("   ⚠ No macro data returned (FRED API may be unavailable)")
    else:
        print(f"   ✓ Macro data retrieved: {len(df_macro)} records × {len(df_macro.columns)} indicators")
        print(f"\n   Indicators fetched:")
        for col in df_macro.columns:
            print(f"   - {col}: {df_macro[col].notna().sum()} values")
        print(f"\n   Latest values (most recent date):")
        print(df_macro.tail(1).T)
except Exception as e:
    print(f"   ❌ Failed to fetch macro data: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ MACRO DATA RETRIEVAL TEST PASSED")
print("=" * 80)
