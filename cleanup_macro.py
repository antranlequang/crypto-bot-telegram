#!/usr/bin/env python3
"""
Clean up macro references and add sentiment_provider initialization
"""

with open('bot.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

# 1. Remove include_macro parameter from build_feature_table signature
old_sig = r'def build_feature_table\(\s*symbol: str,\s*lookback_days: int,\s*include_macro: bool = True,\s*include_onchain: bool = True\s*\)'
new_sig = 'def build_feature_table(\n    symbol: str,\n    lookback_days: int,\n    include_onchain: bool = True\n)'
content = re.sub(old_sig, new_sig, content)
print("✓ Removed include_macro parameter from build_feature_table signature")

# 2. Remove macro data fetch block from build_feature_table
old_macro_block = r'    # Try to fetch macro data if enabled\s*if include_macro and macro_provider:.*?logger\.warning\(f"⚠ Macro data fetch failed: \{exc\}"\)'
content = re.sub(old_macro_block, '', content, flags=re.DOTALL)
print("✓ Removed macro data fetch block from build_feature_table")

# 3. Remove "macro": False initialization (since we're removing macro completely)
# Let's just leave sources_used as is - it will still be there but not used

# 4. Remove macro_provider initialization block
old_macro_init = r'# Initialize macro data provider.*?logger\.info\("Macro data disabled via USE_MACRO_DATA=0 \(skip macro fetch\)"\)'
content = re.sub(old_macro_init, '', content, flags=re.DOTALL)
print("✓ Removed macro_provider initialization block")

# 5. Add sentiment_provider initialization after data_provider init
# Find where data_provider is initialized
sentiment_init = '''# Initialize sentiment data provider (GNews-based)
sentiment_provider = None
try:
    sentiment_provider = SentimentDataProvider()
    logger.info("✓ Sentiment data provider initialized (GNews-based)")
except Exception as e:
    logger.warning(f"Failed to initialize SentimentDataProvider: {e}")
    sentiment_provider = None

'''

# Find the line after coindesk_fetcher initialization
pattern = r'(coindesk_fetcher = None\n)'
content = re.sub(pattern, r'\1\n' + sentiment_init, content)
print("✓ Added sentiment_provider initialization")

# Write back
with open('bot.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\n✅ All macro cleanup and sentiment initialization complete!")
