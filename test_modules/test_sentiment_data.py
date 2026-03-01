#!/usr/bin/env python3
"""
Standalone test to verify sentiment data fetching works in bot.py.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
from datetime import datetime

from bot import sentiment_provider


def test_sentiment_data(symbol: str = "BTC", days: int = 7, show_items: int = 5) -> int:
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    print("\n" + "=" * 90)
    print(f"SENTIMENT DATA TEST - {symbol}")
    print("=" * 90)

    provider_name = getattr(sentiment_provider, "provider_name", "unknown") if sentiment_provider else "none"
    print(f"Provider: {provider_name}")

    if not sentiment_provider or not sentiment_provider.available:
        init_error = getattr(sentiment_provider, "init_error", None) if sentiment_provider else None
        if init_error:
            print(f"⚠ Sentiment provider init error: {init_error}")
        print("⚠ Sentiment provider not available.")
        return 1

    logger.info("Fetching news with sentiment for %s (%s days)...", symbol, days)
    articles = sentiment_provider.fetch_news_with_sentiment(symbol, days=days)

    print(f"\n📰 Articles fetched: {len(articles)}")
    if articles:
        print("\n📌 Sample articles:")
        for idx, item in enumerate(articles[:show_items], 1):
            title = item.get("title") or "N/A"
            score = item.get("sentiment_score")
            published = item.get("published_date") or item.get("published")
            source = item.get("source") or "N/A"
            if isinstance(published, datetime):
                published = published.isoformat()
            print(f"  {idx}. {title}")
            print(f"     • score: {score:+.4f}" if score is not None else "     • score: N/A")
            print(f"     • published: {published}")
            print(f"     • source: {source}")
    else:
        print("❌ No articles returned.")

    logger.info("Fetching daily sentiment aggregation...")
    df_daily = sentiment_provider.fetch_daily_sentiment(symbol, days=days)
    if df_daily.empty:
        print("\n❌ No daily sentiment data returned.")
    else:
        print("\n✅ Daily sentiment (latest rows):")
        print(df_daily.tail(min(len(df_daily), 5)))

    print("\n" + "=" * 90)
    print("✅ SENTIMENT DATA TEST COMPLETED")
    print("=" * 90)

    return 0


if __name__ == "__main__":
    raise SystemExit(test_sentiment_data())
