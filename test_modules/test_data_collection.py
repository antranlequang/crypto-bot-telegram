#!/usr/bin/env python3
"""
Test script to verify that the chatbot collects all required data for analysis.
Simulates the full /analyze command flow.
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

# Setup logging to see detailed information
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Import bot components
from bot import (
    build_feature_table,
    data_provider,
    sentiment_provider,
    macro_provider,
    get_analysis_lookback_days,
    technical_score
)

def test_data_collection(symbol: str = "BTC"):
    """
    Test complete data collection for analysis.
    
    Args:
        symbol: Cryptocurrency symbol to test (default: BTC)
    
    Returns:
        Dictionary with all collected data and statistics
    """
    
    print("\n" + "=" * 90)
    print(f"CHATBOT DATA COLLECTION TEST - {symbol}")
    print("=" * 90)
    
    try:
        # Calculate lookback period
        lookback_days = get_analysis_lookback_days()
        approx_years = lookback_days / 365
        print(f"\n📅 Lookback period: {lookback_days} days (~{approx_years:.1f} years)")
        
        # Build feature table (this collects all data)
        print(f"\n📊 Building feature table for {symbol}...")
        print("-" * 90)
        
        df_feat, feature_info = build_feature_table(
            symbol=symbol,
            lookback_days=lookback_days,
            include_onchain=True,
            include_macro=True
        )
        
        print("-" * 90)
        
        if df_feat.empty:
            print(f"❌ ERROR: Feature table is empty!")
            return None
        
        # Display collected data summary
        print("\n📈 DATA COLLECTION SUMMARY:")
        print("-" * 90)
        
        # 1. OHLCV Data
        print(f"\n✓ OHLCV Data (Price):")
        print(f"  • Records: {len(df_feat)}")
        print(f"  • Date range: {df_feat.index.min().date()} to {df_feat.index.max().date()}")
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in ohlcv_cols:
            if col in df_feat.columns:
                print(f"  • {col}: {df_feat[col].notna().sum()} values, "
                      f"range [{df_feat[col].min():.2f}, {df_feat[col].max():.2f}]")
        
        # 2. On-chain Data (if available)
        onchain_cols = [col for col in df_feat.columns if col.startswith('onchain_')]
        if onchain_cols:
            print(f"\n✓ On-chain Data (CoinDesk):")
            for col in onchain_cols[:5]:  # Show first 5
                non_null = df_feat[col].notna().sum()
                print(f"  • {col}: {non_null} values")
            if len(onchain_cols) > 5:
                print(f"  ... and {len(onchain_cols) - 5} more on-chain metrics")
        else:
            print(f"\n⚠ On-chain Data: Not available for {symbol}")
        
        # 3. Sentiment Data
        if 'sentiment_score' in df_feat.columns:
            print(f"\n✓ Sentiment Data (Alpha Vantage):")
            non_null = df_feat['sentiment_score'].notna().sum()
            print(f"  • sentiment_score: {non_null} values, "
                  f"range [{df_feat['sentiment_score'].min():.4f}, {df_feat['sentiment_score'].max():.4f}]")
            if 'articles_count' in df_feat.columns:
                print(f"  • articles_count: {df_feat['articles_count'].notna().sum()} values")
        else:
            print(f"\n⚠ Sentiment Data: Not available")
        
        # 4. Macro Data
        macro_cols = [col for col in df_feat.columns if col in 
                      ['sp500', 'vix', 'dxy', 'brentoil', 'dowjones', 'goldprice']]
        if macro_cols:
            print(f"\n✓ Macro Data (FRED API):")
            for col in macro_cols:
                non_null = df_feat[col].notna().sum()
                print(f"  • {col}: {non_null} values")
        else:
            print(f"\n⚠ Macro Data: Not available")
        
        # 5. Technical Indicators
        tech_indicators = [
            'RSI_10', 'RSI_14', 'ROC_12', 'STD_DEV_20', 'CCI_20',
            'H_L', 'H_CP', 'MACD_Line_6_20', 'MACD_Signal_6_20',
            'ATR_14', 'OBV', 'EMA_5', 'SMA_5',
            'STOCH_K', 'STOCH_D', 'EMA_12', 'EMA_26', 'MACD_Histogram'
        ]
        
        available_indicators = [ind for ind in tech_indicators if ind in df_feat.columns]
        if available_indicators:
            print(f"\n✓ Technical Indicators:")
            print(f"  • Total: {len(available_indicators)} indicators")
            for ind in available_indicators[:10]:  # Show first 10
                print(f"  • {ind}: {df_feat[ind].notna().sum()} values")
            if len(available_indicators) > 10:
                print(f"  ... and {len(available_indicators) - 10} more indicators")
        else:
            print(f"\n❌ Technical Indicators: None found!")
        
        # 6. Summary statistics
        print(f"\n📊 DATASET SUMMARY:")
        print(f"  • Total rows: {len(df_feat)}")
        print(f"  • Total columns: {len(df_feat.columns)}")
        print(f"  • Data completeness: {(1 - df_feat.isna().sum().sum() / (len(df_feat) * len(df_feat.columns))) * 100:.1f}%")
        print(f"  • NaN count: {df_feat.isna().sum().sum()}")
        print(f"  • Inf count: {np.isinf(df_feat.select_dtypes(include=[np.float64])).sum().sum()}")
        
        # 7. Data sources used
        print(f"\n🔗 DATA SOURCES USED:")
        sources = feature_info.get('sources', {})
        for source, used in sources.items():
            status = "✓" if used else "✗"
            print(f"  {status} {source.upper()}: {'Included' if used else 'Not used'}")
        
        # 8. Latest values (for model input)
        print(f"\n🎯 LATEST VALUES (for model):")
        latest = feature_info.get('latest', {})
        print(f"  Close Price: {latest.get('Close', 'N/A')}")
        print(f"  RSI_14: {latest.get('RSI_14', 'N/A')}")
        print(f"  MACD_Line: {latest.get('MACD_Line_6_20', 'N/A')}")
        print(f"  EMA_5: {latest.get('EMA_5', 'N/A')}")
        if 'sentiment_score' in latest:
            print(f"  Sentiment: {latest.get('sentiment_score', 'N/A')}")
        if 'sp500' in latest:
            print(f"  S&P 500: {latest.get('sp500', 'N/A')}")
        
        # 9. Technical score calculation test
        print(f"\n⚙️ CALCULATING TECHNICAL SCORE:")
        try:
            tech_score = technical_score(df_feat)
            print(f"  ✓ Technical Score: {tech_score['score']:.4f}")
            print(f"  ✓ Trend: {tech_score['trend_note']}")
        except Exception as e:
            print(f"  ❌ Error calculating technical score: {e}")
        
        # 10. Column listing
        print(f"\n📋 ALL COLUMNS IN FEATURE TABLE:")
        print("-" * 90)
        columns = list(df_feat.columns)
        for i, col in enumerate(columns, 1):
            print(f"  {i:2d}. {col}")
        
        print("\n" + "=" * 90)
        print("✅ DATA COLLECTION TEST COMPLETED SUCCESSFULLY")
        print("=" * 90)
        
        return {
            'df_feat': df_feat,
            'feature_info': feature_info,
            'columns': columns,
            'sources': sources,
            'latest': latest
        }
        
    except Exception as e:
        print(f"\n❌ ERROR DURING DATA COLLECTION:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        return None


def test_multiple_symbols():
    """Test data collection for multiple cryptocurrencies"""
    
    symbols = ["BTC", "ETH", "SOL"]
    results = {}
    
    print("\n" + "=" * 90)
    print("TESTING MULTIPLE CRYPTOCURRENCIES")
    print("=" * 90)
    
    for symbol in symbols:
        print(f"\n\nTesting {symbol}...")
        result = test_data_collection(symbol)
        results[symbol] = result
        
        if result:
            print(f"✓ {symbol}: {len(result['df_feat'])} rows × {len(result['columns'])} columns")
        else:
            print(f"✗ {symbol}: FAILED")
    
    return results


def test_sentiment_fetch(symbol: str = "BTC", days: int = 7, show_items: int = 5):
    """
    Test sentiment news fetching and daily sentiment scoring.

    Args:
        symbol: Cryptocurrency symbol (default: BTC)
        days: Number of days to look back
        show_items: Number of news items to display
    """
    print("\n" + "=" * 90)
    print(f"SENTIMENT FETCH TEST - {symbol}")
    print("=" * 90)

    if not sentiment_provider or not sentiment_provider.available:
        print("⚠ Sentiment provider not available or missing API key.")
        return None

    logging.getLogger("bot").setLevel(logging.INFO)
    logger.info("🔍 Fetching sentiment news for %s (%s days)...", symbol, days)

    articles = sentiment_provider.fetch_news_with_sentiment(symbol, days=days)
    print(f"\n📰 Articles fetched: {len(articles)}")

    if articles:
        print("\n📌 Sample articles:")
        for idx, item in enumerate(articles[:show_items], 1):
            title = item.get("title") or "N/A"
            score = item.get("sentiment_score")
            published = item.get("published_date") or item.get("published")
            source = item.get("source") or "N/A"
            print(f"  {idx}. {title}")
            print(f"     • score: {score:+.4f}" if score is not None else "     • score: N/A")
            print(f"     • published: {published}")
            print(f"     • source: {source}")
    else:
        print("❌ No articles returned.")

    logger.info("📊 Fetching daily sentiment aggregation...")
    df_daily = sentiment_provider.fetch_daily_sentiment(symbol, days=days)

    if df_daily.empty:
        print("\n❌ No daily sentiment data returned.")
    else:
        print("\n✅ Daily sentiment (latest rows):")
        print(df_daily.tail(min(len(df_daily), 5)))

    print("\n" + "=" * 90)
    print("✅ SENTIMENT FETCH TEST COMPLETED")
    print("=" * 90)

    return {
        "articles": articles,
        "daily": df_daily
    }


def verify_data_quality(result):
    """Verify data quality metrics"""
    
    if not result:
        return False
    
    df_feat = result['df_feat']
    
    print("\n" + "=" * 90)
    print("DATA QUALITY VERIFICATION")
    print("=" * 90)
    
    checks = {
        "Minimum rows": (len(df_feat) >= 30, f"✓ {len(df_feat)} rows"),
        "Minimum columns": (len(df_feat.columns) >= 25, f"✓ {len(df_feat.columns)} columns"),
        "No NaN values": (df_feat.isna().sum().sum() == 0, f"✓ All values present"),
        "No Inf values": (np.isinf(df_feat.select_dtypes(include=[np.float64])).sum().sum() == 0, 
                         f"✓ No infinite values"),
        "Price data valid": (df_feat['Close'].min() > 0, f"✓ Close price > 0"),
        "Volume data valid": (df_feat['Volume'].min() >= 0, f"✓ Volume >= 0"),
    }
    
    all_passed = True
    for check_name, (passed, message) in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}: {message}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✅ All data quality checks passed!")
    else:
        print("\n⚠️ Some data quality checks failed!")
    
    return all_passed


def export_data_to_csv(result, symbol="BTC"):
    """
    Export collected data to a single CSV file.
    
    Args:
        result: Dictionary with collected data
        symbol: Cryptocurrency symbol
    
    Returns:
        CSV file path or None
    """
    
    if not result:
        print("❌ No data to export!")
        return None
    
    df_feat = result['df_feat']
    
    # Create data_exports directory if it doesn't exist
    export_dir = os.path.join(os.getcwd(), 'data_exports')
    os.makedirs(export_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "=" * 90)
    print("EXPORTING DATA FILES")
    print("=" * 90)
    
    # Export main data to CSV
    csv_file = os.path.join(export_dir, f"{symbol}_feature_table_{timestamp}.csv")
    df_feat.to_csv(csv_file)
    print(f"\n✓ CSV Export:")
    print(f"  📄 {csv_file}")
    print(f"  📊 Records: {len(df_feat)} | Columns: {len(df_feat.columns)}")
    print(f"  📦 Size: {os.path.getsize(csv_file) / 1024:.1f} KB")
    
    print("\n" + "=" * 90)
    print("✅ DATA EXPORT COMPLETED")
    print("=" * 90)
    
    return csv_file


if __name__ == "__main__":
    print("\n🚀 STARTING CHATBOT DATA COLLECTION VERIFICATION\n")

    # Prompt user to input coin symbol
    user_symbol = input("🔎 Enter cryptocurrency symbol to test (e.g. BTC, ETH, SOL): ").strip().upper()

    if not user_symbol:
        print("⚠️ No symbol entered. Defaulting to BTC.")
        user_symbol = "BTC"

    result = test_data_collection(user_symbol)

    # Verify data quality
    if result:
        verify_data_quality(result)

        # Export data to files
        export_path = export_data_to_csv(result, user_symbol)

        # Files are exported but not auto-opened.
        if export_path:
            print("\n📁 Files exported to: data_exports/")
            print("\n💡 You can now:")
            print("   • View the CSV file in any spreadsheet application")

        # Optional: Test multiple symbols
        print("\n\n" + "=" * 90)
        user_input = input("Do you want to test multiple symbols? (y/n): ").strip().lower()
        if user_input == 'y':
            test_multiple_symbols()

    print("\n✨ Test completed!")
