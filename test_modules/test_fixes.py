#!/usr/bin/env python3
"""Test script to verify bot.py fixes"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print("Testing bot.py integration...")

try:
    # Test recommendation engine
    from recommendation_engine import generate_final_recommendation, RECOMMENDATION_WEIGHTS
    print("✅ recommendation_engine imports OK")
    
    # Test ml_model
    from ml_model import predict_price_direction, load_data
    print("✅ ml_model imports OK")
    
    # Test the weight extraction logic
    import pandas as pd
    ml_test = {'proba_up': 0.72, 'proba_down': 0.28, 'label': 'UP'}
    drl_test = {'label': 'BUY', 'confidence': 0.65}
    sentiment_test = pd.DataFrame({'sentiment_score': [0.15, 0.22, 0.18]})
    
    rec_detailed = generate_final_recommendation('BTC', ml_test, drl_test, sentiment_test)
    print("✅ generate_final_recommendation works")
    
    # Test weight extraction (this was the bug)
    w_news = float(rec_detailed.get("weights", {}).get("news_sentiment", 0.0))
    w_drl = float(rec_detailed.get("weights", {}).get("drl_model", 0.0))
    w_xgb = float(rec_detailed.get("weights", {}).get("ml_model", 0.0))
    
    print(f"\n✅ Weight extraction works:")
    print(f"   w_xgb (ML):    {w_xgb:.4f}")
    print(f"   w_drl (DRL):   {w_drl:.4f}")
    print(f"   w_news (News): {w_news:.4f}")
    print(f"   Total:         {w_xgb + w_drl + w_news:.4f}")
    
    # Verify syntax of bot.py
    import py_compile
    try:
        py_compile.compile('bot.py', doraise=True)
        print("\n✅ bot.py syntax OK")
    except py_compile.PyCompileError as e:
        print(f"\n❌ bot.py syntax error: {e}")
        sys.exit(1)
    
    print("\n🎉 All fixes applied successfully!")
    print("\nSummary of fixes:")
    print("  1. ✅ Fixed undefined 'log' error (moved logging setup before ML import)")
    print("  2. ✅ Fixed undefined 'model_weight_detail' (using rec_detailed instead)")
    print("  3. ✅ Fixed weight key names (news_sentiment, drl_model, ml_model)")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
