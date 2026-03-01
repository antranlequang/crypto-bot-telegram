"""
Weighted Recommendation Engine
================================
Generates final investment signals using weighted combination of:
- Machine Learning (XGBoost): 0.6794 weight
- Deep Reinforcement Learning (DRL): 0.1 weight
- News Sentiment: 0.2206 weight

The news sentiment is calculated as the average of the last 3 days.
All scores are normalized to [-1, 1] range before weighting.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


# ============================================================================
# WEIGHTS CONFIGURATION
# ============================================================================

RECOMMENDATION_WEIGHTS = {
    "ml_model": 0.6794,          # XGBoost direction prediction
    "drl_model": 0.1,            # DRL trading agent (BTC)
    "news_sentiment": 0.2206,    # Average news sentiment (last 3 days)
}

NEWS_LOOKBACK_DAYS = 3  # Use last 3 days of news for sentiment average


# ============================================================================
# SCORE NORMALIZATION
# ============================================================================

def normalize_to_range(score: float, min_val: float, max_val: float) -> float:
    """
    Normalize a score to [-1, 1] range.
    
    Args:
        score: Raw score value
        min_val: Minimum possible value
        max_val: Maximum possible value
        
    Returns:
        Normalized score in [-1, 1]
    """
    if max_val == min_val:
        return 0.0
    
    normalized = 2.0 * (score - min_val) / (max_val - min_val) - 1.0
    return float(np.clip(normalized, -1.0, 1.0))


def ml_score_from_prediction(ml_info: Dict) -> float:
    """
    Convert ML model prediction to [-1, 1] score.
    
    Args:
        ml_info: ML model output with 'proba_up' and 'proba_down'
        
    Returns:
        Score in [-1, 1] where 1 = strong UP, -1 = strong DOWN
    """
    if not ml_info:
        return 0.0
    
    proba_up = float(ml_info.get("proba_up", 0.5))
    proba_down = float(ml_info.get("proba_down", 0.5))
    
    # Normalize: 1.0 = 100% UP, -1.0 = 100% DOWN, 0.0 = 50/50
    score = proba_up - proba_down
    return float(np.clip(score, -1.0, 1.0))


def drl_score_from_action(drl_info: Dict) -> float:
    """
    Convert DRL action to [-1, 1] score.
    
    Args:
        drl_info: DRL model output with 'label' (BUY/SELL/HOLD)
        
    Returns:
        Score where 1 = BUY, -1 = SELL, 0 = HOLD
    """
    if not drl_info:
        return 0.0
    
    label = drl_info.get("label", "HOLD").upper()
    if label == "BUY":
        return 1.0
    elif label == "SELL":
        return -1.0
    else:  # HOLD or other
        return 0.0


def news_score_from_sentiment(sentiment_daily: pd.DataFrame, lookback_days: int = 3) -> float:
    """
    Calculate average news sentiment score from last N days.
    
    Args:
        sentiment_daily: DataFrame with daily sentiment scores (indexed by date)
        lookback_days: Number of days to average (default: 3)
        
    Returns:
        Score in [-1, 1] representing average sentiment
    """
    if sentiment_daily is None or sentiment_daily.empty:
        return 0.0
    
    if "sentiment_score" not in sentiment_daily.columns:
        return 0.0
    
    # Get last N days
    recent_scores = sentiment_daily["sentiment_score"].tail(lookback_days)
    
    if recent_scores.empty:
        return 0.0
    
    # Calculate average and clip to [-1, 1]
    avg_score = float(recent_scores.mean())
    return float(np.clip(avg_score, -1.0, 1.0))


# ============================================================================
# WEIGHTED COMBINATION
# ============================================================================

def calculate_weighted_recommendation(
    ml_info: Optional[Dict],
    drl_info: Optional[Dict],
    sentiment_daily: Optional[pd.DataFrame],
    weights: Optional[Dict] = None
) -> Tuple[float, Dict]:
    """
    Calculate final recommendation using weighted combination of all signals.
    
    Args:
        ml_info: ML model prediction with proba_up/proba_down
        drl_info: DRL agent action (BUY/SELL/HOLD)
        sentiment_daily: Daily sentiment scores DataFrame
        weights: Custom weights dict (uses RECOMMENDATION_WEIGHTS if None)
        
    Returns:
        (final_score, details_dict) where:
        - final_score: [-1, 1] combined signal (-1=SELL, 0=HOLD, 1=BUY)
        - details_dict: Component scores, weights, and breakdown
    """
    if weights is None:
        weights = RECOMMENDATION_WEIGHTS
    
    # Normalize all components to [-1, 1]
    ml_score = ml_score_from_prediction(ml_info)
    drl_score = drl_score_from_action(drl_info)
    news_score = news_score_from_sentiment(sentiment_daily, NEWS_LOOKBACK_DAYS)
    
    # Apply weights
    total_weight = sum(weights.values())
    if total_weight == 0:
        total_weight = 1.0  # Avoid division by zero
    
    weighted_sum = (
        weights["ml_model"] * ml_score +
        weights["drl_model"] * drl_score +
        weights["news_sentiment"] * news_score
    )
    
    # Normalize by total weight to keep in [-1, 1]
    normalized_total = weighted_sum / total_weight
    final_score = float(np.clip(normalized_total, -1.0, 1.0))
    
    # Build detail breakdown
    details = {
        "component_scores": {
            "ml_model": ml_score,
            "drl_model": drl_score,
            "news_sentiment": news_score,
        },
        "weights": {
            "ml_model": weights["ml_model"],
            "drl_model": weights["drl_model"],
            "news_sentiment": weights["news_sentiment"],
        },
        "weighted_components": {
            "ml_model": weights["ml_model"] * ml_score,
            "drl_model": weights["drl_model"] * drl_score,
            "news_sentiment": weights["news_sentiment"] * news_score,
        },
        "total_weight": total_weight,
        "weighted_sum": weighted_sum,
        "final_score": final_score,
    }
    
    return final_score, details


# ============================================================================
# RECOMMENDATION LOGIC
# ============================================================================

def score_to_recommendation(
    score: float,
    threshold_buy: float = 0.25,
    threshold_sell: float = -0.25
) -> Dict:
    """
    Convert final score to trading recommendation.
    
    Args:
        score: Final weighted score in [-1, 1]
        threshold_buy: Score threshold for BUY signal (default: 0.25)
        threshold_sell: Score threshold for SELL signal (default: -0.25)
        
    Returns:
        Recommendation dict with action, confidence, and description
    """
    if score >= threshold_buy:
        action = "BUY"
        confidence = float(np.clip(score, 0.0, 1.0))
        description = f"Tín hiệu mua với độ tin cậy {confidence:.0%}"
    elif score <= threshold_sell:
        action = "SELL"
        confidence = float(np.clip(abs(score), 0.0, 1.0))
        description = f"Tín hiệu bán với độ tin cậy {confidence:.0%}"
    else:
        action = "HOLD"
        confidence = float(1.0 - abs(score))  # Higher when close to 0
        description = f"Giữ vị thế (không rõ tín hiệu)"
    
    return {
        "action": action,
        "score": score,
        "confidence": confidence,
        "confidence_percent": confidence * 100.0,
        "description": description,
    }


def generate_final_recommendation(
    symbol: str,
    ml_info: Optional[Dict],
    drl_info: Optional[Dict],
    sentiment_daily: Optional[pd.DataFrame],
    weights: Optional[Dict] = None,
    threshold_buy: float = 0.25,
    threshold_sell: float = -0.25,
) -> Dict:
    """
    Generate complete final recommendation with all details.
    
    Args:
        symbol: Cryptocurrency symbol (BTC, ETH, etc.)
        ml_info: ML model prediction
        drl_info: DRL model action
        sentiment_daily: Daily sentiment DataFrame
        weights: Custom weights
        threshold_buy: BUY signal threshold
        threshold_sell: SELL signal threshold
        
    Returns:
        Complete recommendation dict
    """
    # Calculate weighted score
    final_score, score_details = calculate_weighted_recommendation(
        ml_info, drl_info, sentiment_daily, weights
    )
    
    # Convert to recommendation
    rec = score_to_recommendation(final_score, threshold_buy, threshold_sell)
    
    # Build detailed breakdown
    details = score_details.copy()
    details.update({
        "symbol": symbol,
        "action": rec["action"],
        "confidence": rec["confidence"],
        "confidence_percent": rec["confidence_percent"],
    })
    
    return details


# ============================================================================
# FORMATTING FOR BOT OUTPUT
# ============================================================================

def format_recommendation_for_display(recommendation: Dict) -> str:
    """
    Format recommendation details for bot message display.
    
    Args:
        recommendation: Output from generate_final_recommendation()
        
    Returns:
        Formatted string for Telegram message
    """
    symbol = recommendation.get("symbol", "N/A")
    action = recommendation.get("action", "HOLD")
    confidence = recommendation.get("confidence_percent", 0.0)
    
    scores = recommendation.get("component_scores", {})
    weights = recommendation.get("weights", {})
    
    # Emoji mapping
    action_emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "🟡"
    
    lines = [
        f"{action_emoji} **Khuyến nghị cuối:** {action} ({confidence:.1f}% tin cậy)",
        "",
        "**📊 Chi tiết từng mô hình:**",
    ]
    
    # ML component
    ml_score = scores.get("ml_model", 0.0)
    ml_weight = weights.get("ml_model", 0.0)
    ml_emoji = "📈" if ml_score > 0.2 else "📉" if ml_score < -0.2 else "➡️"
    lines.append(f"  {ml_emoji} XGBoost (ML): {ml_score:+.2f} ({ml_weight:.1%} weight)")
    
    # DRL component
    drl_score = scores.get("drl_model", 0.0)
    drl_weight = weights.get("drl_model", 0.0)
    drl_emoji = "🤖" if drl_score > 0.5 else "⚠️" if drl_score < -0.5 else "⏸️"
    drl_label = "BUY" if drl_score > 0.5 else "SELL" if drl_score < -0.5 else "HOLD"
    lines.append(f"  {drl_emoji} DRL (BTC): {drl_label} ({drl_weight:.1%} weight)")
    
    # News component
    news_score = scores.get("news_sentiment", 0.0)
    news_weight = weights.get("news_sentiment", 0.0)
    news_emoji = "😊" if news_score > 0.1 else "😔" if news_score < -0.1 else "😐"
    news_label = "Tích cực" if news_score > 0.1 else "Tiêu cực" if news_score < -0.1 else "Trung lập"
    lines.append(f"  {news_emoji} Tin tức: {news_label} {news_score:+.2f} ({news_weight:.1%} weight)")
    
    return "\n".join(lines)


def format_recommendation_summary(recommendation: Dict) -> str:
    """
    Format concise recommendation summary.
    
    Args:
        recommendation: Output from generate_final_recommendation()
        
    Returns:
        Concise formatted string
    """
    symbol = recommendation.get("symbol", "?")
    action = recommendation.get("action", "HOLD")
    confidence = recommendation.get("confidence_percent", 0.0)
    final_score = recommendation.get("final_score", 0.0)
    
    action_emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "🟡"
    
    return (
        f"{action_emoji} **{symbol}:** {action} "
        f"(Score: {final_score:+.3f}, Confidence: {confidence:.1f}%)"
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Test example
    ml_test = {
        "proba_up": 0.78,
        "proba_down": 0.22,
        "label": "UP"
    }
    
    drl_test = {
        "label": "BUY",
        "confidence": 0.65
    }
    
    # Create sample sentiment daily data
    sentiment_test = pd.DataFrame({
        "date": pd.date_range(start="2026-02-25", periods=3),
        "sentiment_score": [0.15, 0.22, 0.18]
    })
    
    # Calculate recommendation
    rec = generate_final_recommendation(
        symbol="BTC",
        ml_info=ml_test,
        drl_info=drl_test,
        sentiment_daily=sentiment_test
    )
    
    print("Recommendation Details:")
    print(rec)
    print("\nFormatted for Display:")
    print(format_recommendation_for_display(rec))
    print("\nSummary:")
    print(format_recommendation_summary(rec))
