#!/usr/bin/env python3
"""
ML Model Training and Prediction Script (XGBoost Only)
======================================================
Trains XGBoost classifier for cryptocurrency price direction prediction.
- Uses XGBoost as the sole model (best performance, minimal complexity)
- Returns predictions and confidence scores for use in /analyze command
- Does NOT save models to disk - re-runs fresh each time (as per requirements)
- Supports different cryptocurrencies via symbol parameter
- Fetches historical data from CoinGecko API (free, no authentication needed)
- Calculates 30+ technical indicators for feature engineering
- Can be called directly from bot.py via predict_price_direction()
- Standalone terminal tool for testing: `python ml_model.py [--symbol SOL]`

Integration Flow:
    1. /analyze command is triggered in bot.py
    2. run_full_analysis() collects features data
    3. get_ml_direction_prediction(df_feat) calls ml_model.predict_price_direction()
    4. XGBoost model is trained on the feature data
    5. Final prediction with confidence is returned to recommendation engine
    6. Weighted recommendation is generated using ML + DRL + News signals

Features Calculated:
    - Moving Averages: SMA (5, 10, 20, 50), EMA (12, 26)
    - MACD: MACD line, Signal line, Histogram
    - RSI: Relative Strength Index (14)
    - Bollinger Bands: Upper, Middle, Lower, Width, Position
    - ATR: Average True Range (14)
    - Momentum: Rate of Change, Price Momentum
    - Stochastic: %K, %D
    - CCI: Commodity Channel Index
    - Price/Range metrics

Supported Symbols:
    BTC, ETH, SOL, ADA, BNB, XRP, DOGE, DOT, MATIC, LINK

Usage (Terminal Testing):
    python ml_model.py                          # Test with default ETH
    python ml_model.py --symbol BTC             # Test with BTC
    python ml_model.py --symbol SOL --verbose   # Verbose output

Usage (Bot Integration):
    from ml_model import predict_price_direction
    result = predict_price_direction(None, symbol='SOL')
    # Returns: {
    #     "label": "UP" | "DOWN",
    #     "score": float (0-1),
    #     "proba_up": float (0-1),
    #     "proba_down": float (0-1)
    # }
"""

import logging
import argparse
import warnings
import os
import sys
from typing import Tuple, Dict, Optional
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Cryptocurrency API endpoints
COINGECKO_API_BASE = 'https://api.coingecko.com/api/v3'
SYMBOL_TO_COINGECKO_ID = {
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'SOL': 'solana',
    'ADA': 'cardano',
    'BNB': 'binancecoin',
    'XRP': 'ripple',
    'DOGE': 'dogecoin',
    'DOT': 'polkadot',
    'MATIC': 'matic-network',
    'LINK': 'chainlink'
}

TEMP_DOWNLOAD_NAME = 'ml_features_temp.csv'

TRADING_CONFIG = {
    'transaction_fee': 0.001,        # 0.1% per transaction
    'stop_loss_pct': 0.02,           # 2% stop loss
    'take_profit_pct': 0.06,         # 6% take profit
    'probability_threshold': 0.60,   # Buy threshold for ML probability
    'annualized_trading_days': 252,
    'risk_free_rate': 0.0
}

SPLIT_RATIO = 0.8  # 80% train, 20% test


# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_data(symbol: str = 'BTC', use_local: bool = True) -> pd.DataFrame:
    """
    Load feature data for the cryptocurrency from local file or API.
    
    Args:
        symbol: Cryptocurrency symbol (BTC, ETH, SOL, etc.)
        use_local: If True, try local file first; fall back to API
        
    Returns:
        DataFrame with features
    """
    logger.info(f"Loading data for {symbol}...")
    
    # Construct local path based on symbol
    local_data_path = f'{symbol.lower()}_features.csv'
    
    # Try local file first
    if use_local and os.path.exists(local_data_path):
        logger.info(f"Loading from local file: {local_data_path}")
        df = pd.read_csv(local_data_path)
        logger.info(f"Data shape: {df.shape}")
        return df
    
    # Fetch from API
    logger.info(f"Fetching {symbol} data from CoinGecko API...")
    df = _fetch_data_from_api(symbol)
    
    if df is not None:
        # Save to local file for future use
        df.to_csv(local_data_path, index=False)
        logger.info(f"Data saved to local file: {local_data_path}")
        logger.info(f"Data shape: {df.shape}")
        return df
    else:
        logger.error(f"Failed to fetch data for {symbol}")
        raise ValueError(f"Could not fetch data for {symbol} from API or local storage")


def _fetch_data_from_api(symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
    """
    Fetch historical OHLCV data from CoinGecko API and calculate technical indicators.
    
    Args:
        symbol: Cryptocurrency symbol (BTC, ETH, SOL, etc.)
        days: Number of days of historical data to fetch (default: 365)
        
    Returns:
        DataFrame with date, OHLCV, and technical indicators
    """
    try:
        # Map symbol to CoinGecko ID
        coin_id = SYMBOL_TO_COINGECKO_ID.get(symbol.upper())
        if not coin_id:
            logger.error(f"Symbol {symbol} not supported. Supported: {list(SYMBOL_TO_COINGECKO_ID.keys())}")
            return None
        
        logger.info(f"Fetching {days} days of {symbol} data from CoinGecko...")
        
        # Fetch OHLCV data from CoinGecko
        url = f"{COINGECKO_API_BASE}/coins/{coin_id}/ohlc"
        params = {
            'vs_currency': 'usd',
            'days': min(days, 365)  # CoinGecko free API limit is 365 days
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        ohlcv_data = response.json()
        
        if not ohlcv_data:
            logger.error(f"No data returned from CoinGecko for {symbol}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[['date', 'open', 'high', 'low', 'close']]
        
        logger.info(f"✓ Fetched {len(df)} candles from API")
        
        # Calculate technical indicators
        df = _calculate_technical_indicators(df)
        
        return df
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Error fetching data from API: {e}")
        return None


def _calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for ML features.
    
    Args:
        df: DataFrame with OHLCV data (date, open, high, low, close)
        
    Returns:
        DataFrame with additional technical indicator columns
    """
    logger.info("Calculating technical indicators...")
    
    df = df.copy()
    
    # Simple Moving Averages
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # MACD (Moving Average Convergence Divergence)
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    sma_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    df['bb_upper'] = sma_20 + (std_20 * 2)
    df['bb_lower'] = sma_20 - (std_20 * 2)
    df['bb_middle'] = sma_20
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    
    # Price Position in Bollinger Bands
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # ATR (Average True Range)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift()),
            abs(df['low'] - df['close'].shift())
        )
    )
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    # Momentum and Rate of Change
    df['momentum'] = df['close'] - df['close'].shift(10)
    df['roc'] = (df['close'] - df['close'].shift(12)) / df['close'].shift(12) * 100
    
    # Volume (using price range as proxy for volume since we don't have volume data)
    df['price_range'] = df['high'] - df['low']
    df['price_change'] = df['close'].pct_change() * 100
    
    # Stochastic Oscillator
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # ADX (Average Directional Index) - Simplified
    df['up_move'] = df['high'].diff()
    df['down_move'] = -df['low'].diff()
    df['true_range'] = df['tr']
    df['plus_dm'] = df['up_move'].where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), 0)
    df['minus_dm'] = df['down_move'].where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), 0)
    
    # CCI (Commodity Channel Index)
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = tp.rolling(window=20).mean()
    df['cci'] = (tp - sma_tp) / (0.015 * tp.rolling(window=20).std())
    
    # Drop intermediate columns
    df = df.drop(columns=['tr', 'up_move', 'down_move', 'true_range', 'plus_dm', 'minus_dm'], errors='ignore')
    
    # Fill NaN values with forward fill then backward fill
    df = df.ffill().bfill()
    
    logger.info(f"✓ Calculated {len(df.columns) - 5} technical indicators (5 OHLCV columns)")
    
    return df


def prepare_data(df: pd.DataFrame, verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, RobustScaler]:
    """
    Prepare data for ML training.
    
    Args:
        df: Raw feature DataFrame
        verbose: If True, log preparation details; if False, suppress logs
        
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    if verbose:
        logger.info("Preparing data...")
    
    # Parse date and create target
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    
    # Target: 1 if tomorrow's close > today's close, else 0
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(subset=['target'], inplace=True)
    
    y = df['target']
    X = df.drop(columns=['date', 'close', 'target', 'avg_sentiment_score'], errors='ignore')
    
    # Handle NaN values
    X = X.fillna(X.mean())
    
    # Chronological split
    split_index = int(len(df) * SPLIT_RATIO)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    # Scale features using RobustScaler (handles outliers better for financial data)
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    if verbose:
        logger.info(f"Training set: {X_train_scaled.shape}")
        logger.info(f"Test set: {X_test_scaled.shape}")
        logger.info(f"Target distribution (train): {y_train.value_counts().to_dict()}")
        logger.info(f"Target distribution (test): {y_test.value_counts().to_dict()}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series, verbose: bool = False):
    """
    Train XGBoost classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        verbose: If True, log training messages; if False, suppress logs
        
    Returns:
        Trained XGBoost model
    """
    if verbose:
        logger.info("Training XGBoost Classifier...")
    
    model = xgb.XGBClassifier(
        random_state=42,
        verbosity=0,
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1
    )
    model.fit(X_train, y_train)
    
    if verbose:
        logger.info("✓ XGBoost model trained successfully")
    
    return model


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    show_metrics: bool = False
) -> dict:
    """
    Evaluate XGBoost model on test set.
    
    Args:
        model: Trained XGBoost model
        X_test: Test features
        y_test: Test target
        show_metrics: If True, log detailed metrics
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating XGBoost model...")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    if show_metrics:
        logger.info(f"\nXGBoost Model Evaluation Results:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1-Score:  {f1:.4f}")
        logger.info(f"  AUC-ROC:   {auc_score:.4f}")
    
    return results





def predict_price_direction(df_feat: Optional[pd.DataFrame] = None, symbol: str = 'BTC', use_local: bool = True) -> Dict:
    """
    Predict price direction for cryptocurrency using XGBoost.
    
    This is the main function called by bot.py's get_ml_direction_prediction().
    Can accept pre-computed features (from bot) OR collect data itself (standalone).
    DOES NOT SAVE DATA - runs fresh each time as per requirements.
    
    Args:
        df_feat: Pre-computed feature DataFrame (from bot.py). If None, will collect data.
        symbol: Cryptocurrency symbol (BTC, ETH, etc.)
        use_local: Use local data if available (when df_feat is None)
        
    Returns:
        Dictionary with keys:
        - label: "UP" or "DOWN"
        - score: confidence score (0-1)
        - proba_up: P(up) for recommendation engine
        - proba_down: P(down) for recommendation engine
        
    Example (from bot.py):
        >>> ml_info = predict_price_direction(df_feat, symbol='BTC')
        >>> print(ml_info['label'])  # 'UP' or 'DOWN'
        >>> print(ml_info['proba_up'])  # 0.72
        
    Example (standalone):
        >>> ml_info = predict_price_direction(symbol='ETH')
        >>> print(ml_info)
    """
    logger.info(f"Predicting price direction for {symbol}...")
    
    try:
        # If features provided (called from bot.py), use them directly
        if df_feat is not None:
            X_train, X_test, y_train, y_test, scaler = _prepare_features_for_model(df_feat, verbose=False)
        else:
            # Standalone mode: collect data and prepare features
            df = load_data(symbol, use_local)
            X_train, X_test, y_train, y_test, scaler = prepare_data(df, verbose=False)
        
        # Train XGBoost model
        xgb_model = train_xgboost_model(X_train, y_train, verbose=False)
        
        # Get predictions on test set
        y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]  # P(UP)
        
        # Use latest prediction
        latest_proba_up = float(y_pred_proba[-1])
        latest_proba_down = 1.0 - latest_proba_up
        
        # Determine direction
        direction = "UP" if latest_proba_up > 0.5 else "DOWN"
        score = max(latest_proba_up, latest_proba_down)
        
        result = {
            "label": direction,
            "score": score,
            "proba_up": latest_proba_up,
            "proba_down": latest_proba_down
        }
        
        logger.info(f"✓ Prediction: {direction} (confidence: {score:.2%})")
        return result
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return {
            "label": "UNKNOWN",
            "score": 0.5,
            "proba_up": 0.5,
            "proba_down": 0.5
        }


def _prepare_features_for_model(df_feat: pd.DataFrame, verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, RobustScaler]:
    """
    Prepare pre-computed features (from bot.py) for XGBoost training.
    
    Args:
        df_feat: Feature DataFrame from build_feature_table()
        verbose: If True, log details
        
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    if verbose:
        logger.info("Preparing features for model training...")
    
    df = df_feat.copy()
    
    # Create target: 1 if tomorrow's close > today's close, else 0
    if 'Close' in df.columns:
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    else:
        # Fallback if column name is different
        close_col = [c for c in df.columns if 'close' in c.lower()][0] if any('close' in c.lower() for c in df.columns) else None
        if close_col:
            df['target'] = (df[close_col].shift(-1) > df[close_col]).astype(int)
        else:
            raise ValueError("No Close/close column found in features")
    
    df.dropna(subset=['target'], inplace=True)
    
    y = df['target']
    
    # Select only numeric columns and drop non-feature columns
    X = df.select_dtypes(include=[np.number]).drop(columns=['target'], errors='ignore')
    
    # Handle NaN values
    X = X.fillna(X.mean())
    
    # Chronological split: 80/20
    split_index = int(len(df) * SPLIT_RATIO)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    # Scale features using RobustScaler
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    if verbose:
        logger.info(f"Training set: {X_train_scaled.shape}")
        logger.info(f"Test set: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ============================================================================
# STANDALONE TERMINAL TESTING & DISPLAY
# ============================================================================

def format_recommendation(symbol: str, ml_info: Dict) -> str:
    """Format ML recommendation for display."""
    action_emoji = "🟢" if ml_info['label'] == "UP" else "🔴"
    confidence = ml_info['score'] * 100
    
    lines = [
        "",
        "=" * 80,
        f"  📊 ML MODEL RECOMMENDATION - {symbol}",
        "=" * 80,
        f"  {action_emoji} Action: {ml_info['label']}",
        f"  📈 Probability UP:   {ml_info['proba_up']:.2%}",
        f"  📉 Probability DOWN: {ml_info['proba_down']:.2%}",
        f"  💪 Confidence:       {confidence:.1f}%",
        "=" * 80,
        ""
    ]
    return "\n".join(lines)


def run_standalone_analysis(symbol: str = 'ETH', verbose: bool = False):
    """
    Run complete ML analysis as standalone terminal tool.
    Mimics what bot.py does in run_full_analysis() for the ML component.
    
    Args:
        symbol: Cryptocurrency symbol (default: ETH for testing)
        verbose: Show detailed information during analysis
    """
    print("\n" + "=" * 80)
    print("  🚀 ML MODEL STANDALONE TEST")
    print("=" * 80)
    print(f"  Symbol: {symbol}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 80 + "\n")
    
    try:
        # Step 1: Load data
        print(f"[1/4] Loading feature data for {symbol}...")
        df = load_data(symbol, use_local=True)
        print(f"      ✓ Loaded {len(df)} rows of data")
        
        # Step 2: Prepare data
        print(f"\n[2/4] Preparing features...")
        X_train, X_test, y_train, y_test, scaler = prepare_data(df, verbose=verbose)
        print(f"      ✓ Training samples: {len(X_train)}")
        print(f"      ✓ Test samples: {len(X_test)}")
        print(f"      ✓ Features: {X_train.shape[1]}")
        
        # Step 3: Train model
        print(f"\n[3/4] Training XGBoost model...")
        xgb_model = train_xgboost_model(X_train, y_train, verbose=False)
        print(f"      ✓ Model trained successfully")
        
        # Step 4: Evaluate and predict
        print(f"\n[4/4] Evaluating model and making prediction...")
        results = evaluate_model(xgb_model, X_test, y_test, show_metrics=verbose)
        
        # Get latest prediction
        y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
        ml_info = {
            "label": "UP" if y_pred_proba[-1] > 0.5 else "DOWN",
            "score": max(y_pred_proba[-1], 1.0 - y_pred_proba[-1]),
            "proba_up": float(y_pred_proba[-1]),
            "proba_down": float(1.0 - y_pred_proba[-1])
        }
        
        # Display results
        print(f"      ✓ Prediction generated")
        print(format_recommendation(symbol, ml_info))
        
        # Display model metrics summary
        print(f"\n  📋 Model Performance Summary:")
        print(f"      Accuracy:  {results['accuracy']:.2%}")
        print(f"      Precision: {results['precision']:.2%}")
        print(f"      Recall:    {results['recall']:.2%}")
        print(f"      F1-Score:  {results['f1']:.2%}")
        print(f"      AUC-ROC:   {results['auc']:.2%}")
        print(f"\n  ✅ Analysis Complete!\n")
        
        return ml_info
        
    except Exception as e:
        print(f"\n  ❌ Error during analysis: {e}")
        print(f"      Make sure you have internet access or the local feature file exists\n")
        return None


# ============================================================================
# COMMAND-LINE INTERFACE (FOR TESTING)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='XGBoost ML Model - Standalone Testing Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ml_model.py                    # Test with default ETH
  python ml_model.py --symbol BTC       # Test with BTC
  python ml_model.py --symbol ETH -v    # Verbose output
  python ml_model.py --symbol SOL --no-local  # Force download from Google Drive
        """
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='ETH',
        help='Cryptocurrency symbol to analyze (default: ETH)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed information during analysis'
    )
    parser.add_argument(
        '--no-local',
        action='store_true',
        help='Force download from Google Drive (ignore local file)'
    )
    
    args = parser.parse_args()
    
    # Override use_local in load_data if --no-local is specified
    if args.no_local:
        global load_data_use_local
        load_data_use_local = False
    
    # Run standalone analysis
    ml_info = run_standalone_analysis(symbol=args.symbol, verbose=args.verbose)
    
    if ml_info is None:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
