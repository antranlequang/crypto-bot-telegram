#!/usr/bin/env python3
"""
Integrate sentiment data fetch into build_feature_table
"""

with open('bot.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find where to insert sentiment fetch
# We want to add it after df_raw = df_crypto.copy() and before the indicator computation

insert_text = '''    # Fetch and merge sentiment data from GNews
    if sentiment_provider:
        try:
            logger.info(f"📰 Retrieving sentiment data for {symbol}...")
            df_sentiment = sentiment_provider.fetch_daily_sentiment(symbol, days=lookback_days)
            if not df_sentiment.empty:
                df_raw = df_raw.join(df_sentiment, how='left')
                df_raw['sentiment_score'].fillna(method='ffill', inplace=True)
                df_raw['sentiment_score'].fillna(0, inplace=True)  # Default neutral sentiment
                sources_used["sentiment"] = True
                logger.info(f"✓ Sentiment data merged successfully for {symbol}")
            else:
                logger.info(f"ℹ No sentiment data found for {symbol}, proceeding without sentiment")
                sources_used["sentiment"] = False
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"⚠ Sentiment data fetch failed for {symbol}: {exc}")
            sources_used["sentiment"] = False
    else:
        logger.info("⚠ Sentiment provider not available, skipping sentiment fetch")
        sources_used["sentiment"] = False

'''

# Find the line "    df_raw = df_crypto.copy()" and insert after it
for i, line in enumerate(lines):
    if '    df_raw = df_crypto.copy()' in line:
        # Insert the sentiment code after this line
        lines.insert(i + 1, '\n')
        # Need to insert in reverse order since each insert shifts indices
        insert_lines = insert_text.split('\n')
        for j, insert_line in enumerate(reversed(insert_lines)):
            lines.insert(i + 2, (insert_line + '\n') if insert_line else '\n')
        print(f"✓ Inserted sentiment fetch at line {i + 2}")
        break

# Also need to update sources_used initialization to include sentiment
for i, line in enumerate(lines):
    if 'sources_used = {"ohlcv": True, "onchain": False, "macro": False}' in line:
        lines[i] = '    sources_used = {"ohlcv": True, "onchain": False, "sentiment": False}\n'
        print("✓ Updated sources_used to track sentiment")
        break

with open('bot.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("✅ Sentiment integration complete!")
