# bot.py
"""
Crypto Analysis Bot - Unified AI-Powered Trading Recommendation System
========================================================================

DATA COLLECTION STRATEGY (Updated March 2026):
- OHLCV: 5 years of price data (1,825 days) from CoinGecko/Binance/CMC
- On-Chain: 5 years of blockchain metrics (Glassnode/CryptoCompare)
- Macro: 5 years of economic indicators (FRED API - VIX, DXY, Gold, etc.)
- Sentiment: 3-day rolling average of news articles with sentiment scores

MODEL PREDICTION PIPELINE:
1. ML Model (XGBoost): Trained fresh on 5-year feature set for price direction
2. DRL Model (PPO Agent): Regime detection on Bitcoin market (BTC focus)
3. Sentiment Analysis: News sentiment aggregation (3-day rolling average)
4. Weighted Recommendation: Combines 3 models with calibrated weights
   - ML Direction: 67.94%
   - DRL Regime: 10.00%
   - News Sentiment: 22.06%

DATA EXPORT (/data command):
- Exports COMPLETE unfiltered 5-year dataset
- Includes OHLCV, on-chain, macro, technical indicators, and news
- No time period filtering applied to downloads
"""

import os
import io
import json
import logging
import re
import time
import zipfile
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from html import escape as html_escape
from urllib.parse import urlparse

import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dateutil import parser as dateparser
from dateutil.relativedelta import relativedelta
# from keep_alive import keep_alive

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputFile,
    BotCommand
)
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
    Application,
)
from telegram.helpers import escape_markdown

from stable_baselines3 import PPO

# DRL training script execution

# scheduling helpers (run daily at 8:00 Asia/Ho_Chi_Minh)
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz
import shutil

from config import (
    TELEGRAM_BOT_TOKEN,
    SECONDARY_BOT_TOKEN,
    MONITOR_CHAT_ID,
    CMC_API_KEY, CMC_BASE_URL,
    COINDESK_API_KEY, COINDESK_BASE_URL,
    COINMARKETCAL_API_KEY,
    TWITTER_BEARER_TOKEN, REDDIT_USER_AGENT,
    FRED_API_KEY,
    ALPHA_VANTAGE_API_KEY,
    SENTIMENT_PROVIDER,
    GEMINI_API_KEY, GOOGLE_PROJECT_ID, GOOGLE_LOCATION,
    PPO_BTC_MODEL_PATH, XGB_DIRECTION_MODEL_PATH,
    XGB_DIRECTION_MODEL_WITH_ONCHAIN_PATH, XGB_DIRECTION_MODEL_NO_ONCHAIN_PATH,
    WINDOW_SIZE, DATA_LOOKBACK_DAYS, DISPLAY_NEWS_DAYS, NEWS_LOOKBACK_DAYS, CHART_FOLDER, DATA_EXPORT_FOLDER,
    RAG_SYSTEM_PROMPT, ANALYSIS_QUESTION_KEYWORDS,
    EXPLANATION_REPLY_TEMPLATE, EXPLANATION_PARSE_MODE,
    USE_RISK_SCORE
)

# Import weighted recommendation engine
from recommendation_engine import (
    generate_final_recommendation,
    format_recommendation_for_display,
    format_recommendation_summary,
    RECOMMENDATION_WEIGHTS
)

# ---------- Logging ----------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)
for handler in logging.getLogger().handlers:
    handler.addFilter(lambda record: record.levelno != logging.WARNING)

# Import ML model for price direction prediction
try:
    from ml_model import (
        predict_price_direction,
        load_data as ml_load_data,
        prepare_data as ml_prepare_data,
        train_xgboost_model,
        evaluate_model
    )
    ML_MODEL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ML model import failed: {e}. ML predictions will be unavailable.")
    ML_MODEL_AVAILABLE = False
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)

# ---------- Utils ----------
def normalize_symbol(symbol: str) -> str:
    return symbol.upper().replace("USDT", "").replace("USD", "")


def get_ml_direction_prediction(df_feat: pd.DataFrame) -> dict:
    """
    Get ML price direction prediction from the pre-configured ml_model.py.
    
    This function is called automatically by run_full_analysis when /analyze is triggered.
    It uses the predict_price_direction() function from ml_model.py which has been
    already trained and configured.
    
    Data Flow:
        1. User clicks /analyze BTC
        2. collect_data() gathers OHLCV, on-chain, sentiment, macro data
        3. build_feature_table() creates feature dataframe
        4. This function calls ml_model.predict_price_direction(df_feat)
        5. Prediction is fed to recommendation_engine as ml_info
        6. Final recommendation combines ML + DRL + News with weights (0.6794 + 0.1 + 0.2206)
    
    Args:
        df_feat: Feature dataframe prepared from collect_data with 40+ indicators
        
    Returns:
        dict with keys: label, score, proba_up, proba_down
        - label: "UP" or "DOWN"
        - score: confidence score (0-1)
        - proba_up: P(up) for recommendation engine
        - proba_down: P(down) for recommendation engine
    """
    if not ML_MODEL_AVAILABLE:
        logger.warning("ML model not available - returning neutral prediction")
        return {
            "label": "UNKNOWN",
            "score": 0.5,
            "proba_up": 0.5,
            "proba_down": 0.5
        }
    
    try:
        # Use the pre-configured ml_model.py predict_price_direction function
        prediction = predict_price_direction(df_feat)
        
        if prediction is None:
            logger.warning("ML model returned None prediction")
            return {
                "label": "UNKNOWN",
                "score": 0.5,
                "proba_up": 0.5,
                "proba_down": 0.5
            }
        
        # Extract prediction components
        label = prediction.get("label", "UNKNOWN")
        proba_up = float(prediction.get("proba_up", 0.5))
        proba_down = float(prediction.get("proba_down", 0.5))
        score = float(prediction.get("score", max(proba_up, proba_down)))
        
        logger.info("ML prediction: %s (confidence: %.2f%%)", label, score * 100)
        
        return {
            "label": label,
            "score": score,
            "proba_up": proba_up,
            "proba_down": proba_down
        }
    except Exception as exc:
        logger.warning("ML direction prediction error: %s", exc)
        return {
            "label": "UNKNOWN",
            "score": 0.5,
            "proba_up": 0.5,
            "proba_down": 0.5
        }


MAX_CONVERSATION_HISTORY = 50
MAX_RAG_HISTORY = 10
VIETNAM_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
AUTO_REFRESH_INTERVAL = timedelta(hours=1)
UNCLEAR_REPLY = "Vui lòng làm rõ lại câu hỏi, tôi vẫn chưa hiểu rõ ý bạn."
NEWS_FETCH_LIMIT = 10
MAX_NEWS_PER_SENTIMENT = 5
ANALYSIS_LOOKBACK_YEARS = 5
USE_DRL = True
DEFAULT_NEWS_DAYS = 3  # Default number of days to fetch news data
CACHE_TTL_SECONDS = 15 * 60
_FEATURE_TABLE_CACHE: dict[tuple, dict] = {}
_REALTIME_PRICE_CACHE: dict[str, dict] = {}
_SENTIMENT_NEWS_CACHE: dict[tuple, dict] = {}
REALTIME_PRICE_TTL_SECONDS = 5 * 60
SENTIMENT_NEWS_TTL_SECONDS = 30 * 60


# ---------- Monitoring/Logging to Secondary Bot ----------
async def send_to_monitor_bot(message: str) -> bool:
    """
    Send a monitoring log message to the secondary bot.
    
    Args:
        message: The message to send to the monitoring chat
        
    Returns:
        True if successful, False otherwise
    """
    if not SECONDARY_BOT_TOKEN or not MONITOR_CHAT_ID:
        logger.debug("Monitoring bot not configured (SECONDARY_BOT_TOKEN or MONITOR_CHAT_ID missing)")
        return False
    
    try:
        import httpx
        
        # Format timestamp in Vietnam time
        now = datetime.now(VIETNAM_TZ)
        timestamp = now.strftime("%H:%M:%S %d/%m/%Y")
        
        # Create formatted message with timestamp
        formatted_message = f"⏰ {timestamp}\n{message}"
        
        # Send to secondary bot
        api_url = f"https://api.telegram.org/bot{SECONDARY_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": MONITOR_CHAT_ID,
            "text": formatted_message,
            "parse_mode": "HTML"
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(api_url, json=payload)
            if response.status_code == 200:
                logger.debug("Monitor log sent successfully")
                return True
            else:
                logger.warning(f"Failed to send monitor log (status {response.status_code})")
                return False
    except Exception as exc:
        logger.warning(f"Error sending monitor log: {exc}")
        return False


def _version_tuple(value: str) -> tuple[int, ...]:

    parts = []
    for chunk in value.split("."):
        match = re.match(r"(\\d+)", chunk)
        if not match:
            break
        parts.append(int(match.group(1)))
    return tuple(parts)


def _ensure_httpx_version(min_version: str = "0.28.0") -> None:
    try:
        import httpx
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Thiếu thư viện httpx. Hãy chạy: pip install -U httpx") from exc

    current = getattr(httpx, "__version__", "0")
    if _version_tuple(current) < _version_tuple(min_version):
        raise RuntimeError(
            f"httpx {current} quá cũ; cần >= {min_version}. "
            "Hãy chạy: pip install -U httpx"
        )


def _append_history_entry(user_data: dict, role: str, text: str | None, symbol: str | None = None):
    if not text:
        return
    history = user_data.setdefault("conversation_history", [])
    history.append({
        "role": role,
        "text": text.strip(),
        "symbol": symbol,
        "timestamp": datetime.now().isoformat()
    })
    if len(history) > MAX_CONVERSATION_HISTORY:
        history.pop(0)


def log_user_message(user_data: dict, text: str, symbol: str | None = None):
    _append_history_entry(user_data, "user", text, symbol)


def log_bot_message(user_data: dict, text: str, symbol: str | None = None):
    _append_history_entry(user_data, "assistant", text, symbol)


def get_recent_history(user_data: dict, limit: int = MAX_RAG_HISTORY) -> list[dict]:
    history = user_data.get("conversation_history", [])
    if not history:
        return []
    if limit <= 0 or len(history) <= limit:
        return history.copy()
    return history[-limit:].copy()


def to_vietnam_time(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        dt = value.to_pydatetime()
    elif isinstance(value, datetime):
        dt = value
    else:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(VIETNAM_TZ)


# ---------- DRL training helpers ----------

def run_drl_script() -> bool:
    """Execute drl_model.py to retrain the DRL model.

    Returns True if execution succeeded, False otherwise.
    """
    script_path = os.path.join(os.path.dirname(__file__), "drl_model.py")
    if not os.path.exists(script_path):
        logger.error("DRL script not found at %s", script_path)
        return False
    try:
        cmd = [
            sys.executable,
            script_path,
            "--no-plot",
            "--source",
            "local",
            "--model-path",
            PPO_BTC_MODEL_PATH,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            logger.error("DRL script failed (code %s)", result.returncode)
            if result.stdout:
                logger.error("DRL script stdout:\n%s", result.stdout)
            if result.stderr:
                logger.error("DRL script stderr:\n%s", result.stderr)
            return False
        logger.info("DRL script executed successfully")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Error executing DRL script: %s", exc, exc_info=True)
        return False


def get_dated_model_path(dt: datetime | None = None) -> str:
    """Return a path for today's (UTC) DRL model zip file."""
    if dt is None:
        dt = datetime.utcnow()
    day = dt.strftime("%Y%m%d")
    base_dir = os.path.dirname(PPO_BTC_MODEL_PATH)
    return os.path.join(base_dir, f"drl_ppo_btc_{day}.zip")


def select_drl_model_path() -> str:
    """Choose which DRL model file to load for analysis.

    Prefer today's dated file; fall back to the generic path. If neither exist,
    log and trigger an immediate update to generate one.
    """
    today_path = get_dated_model_path()
    if os.path.exists(today_path):
        return today_path

    logger.warning("No DRL model found for today (%s).", today_path)
    # fall back to base path
    if os.path.exists(PPO_BTC_MODEL_PATH):
        logger.info("Using fallback model %s", PPO_BTC_MODEL_PATH)
        return PPO_BTC_MODEL_PATH

    logger.warning("Fallback model %s missing as well; running daily update now.", PPO_BTC_MODEL_PATH)
    daily_btc_update()
    if os.path.exists(today_path):
        return today_path
    return PPO_BTC_MODEL_PATH


def daily_btc_update() -> None:
    """Scheduled task run each morning to refresh BTC data and retrain DRL model.

    This will rebuild the feature table, export it to `btc_features.csv` and
    then execute the DRL training script to produce an updated model zip file. The
    resulting zip is also copied to a date‑stamped filename for that UTC day.
    """
    logger.info("🔄 starting scheduled BTC update and DRL training")
    
    # Send monitoring log for DRL training start
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        monitor_msg = "🤖 <b>Scheduled DRL Training Started:</b>\n⏰ Daily BTC data refresh and model retraining at 00:30 UTC"
        loop.run_until_complete(send_to_monitor_bot(monitor_msg))
    except Exception as exc:
        logger.warning(f"Failed to send DRL start log: {exc}")
    
    try:
        df_feat, _ = build_feature_table("BTC", DATA_LOOKBACK_DAYS)
        csv_path = "btc_features.csv"
        df_feat.to_csv(csv_path, index=True, index_label="Date")
        logger.info("Exported BTC feature CSV to %s", csv_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to refresh BTC data: %s", exc)

    success = run_drl_script()
    if success:
        # copy produced model to dated file
        try:
            dated_path = get_dated_model_path()
            shutil.copy2(PPO_BTC_MODEL_PATH, dated_path)
            logger.info("Copied model to dated file %s", dated_path)
            
            # Send monitoring log for DRL training completion
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                completed_at_utc = datetime.now(timezone.utc)
                completed_at_vn = format_vietnam_time(completed_at_utc)
                monitor_msg = (
                    "✅ <b>Scheduled DRL Training Completed:</b>\n"
                    f"- Time (UTC): {completed_at_utc.isoformat(timespec='seconds')}\n"
                    f"- Time (VN): {completed_at_vn}\n"
                    "- BTC data updated\n"
                    f"- Model retrained and saved to {dated_path}"
                )
                loop.run_until_complete(send_to_monitor_bot(monitor_msg))
            except Exception as exc:
                logger.warning(f"Failed to send DRL completion log: {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to copy model to dated filename: %s", exc)
    else:
        # Send monitoring log for DRL training failure
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            monitor_msg = "❌ <b>Scheduled DRL Training Failed:</b>\n- DRL training script failed - check logs for details"
            loop.run_until_complete(send_to_monitor_bot(monitor_msg))
        except Exception as exc:
            logger.warning(f"Failed to send DRL failure log: {exc}")

def format_vietnam_time(value) -> str:
    dt = to_vietnam_time(value)
    if not dt:
        return "N/A"
    return dt.strftime("%H:%M %d/%m")

def _last_reset_time(user_data: dict) -> datetime | None:
    ts = user_data.get("history_last_reset")
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


async def refresh_history_if_needed(message, context: ContextTypes.DEFAULT_TYPE):
    now = datetime.now(timezone.utc)
    last_reset = _last_reset_time(context.user_data)
    if not last_reset:
        context.user_data["history_last_reset"] = now.isoformat()
        return

    if now - last_reset >= AUTO_REFRESH_INTERVAL:
        context.user_data.clear()
        context.user_data["history_last_reset"] = now.isoformat()
        text = (
            "🔄 **Chat đã được làm mới tự động**\n\n"
            "Lịch sử phân tích đã được xóa vì đã quá 1 giờ.\n"
            "Vui lòng chạy lại lệnh `/analyze <mã_coin>` để bắt đầu phiên mới."
        )
        await message.reply_text(text, parse_mode="Markdown")
        log_bot_message(context.user_data, text)


def _is_unclear_query(text: str) -> bool:
    stripped = (text or "").strip()
    if len(stripped) < 6:
        return True
    tokens = re.findall(r"[A-Za-z0-9]+", stripped)
    if len(tokens) < 2:
        return True
    return False


def get_analysis_lookback_days(years: int = ANALYSIS_LOOKBACK_YEARS) -> int:
    today = datetime.now(timezone.utc).date()
    start = today - relativedelta(years=years)
    requested = (today - start).days + 1  # count inclusive span
    return min(requested, DATA_LOOKBACK_DAYS)


def get_realtime_price_coingecko(symbol: str, vs_currency: str = "USD") -> float | None:
    cache_key = f"{symbol.upper()}:{vs_currency.upper()}"
    cached = _REALTIME_PRICE_CACHE.get(cache_key)
    if cached:
        cached_at = cached.get("fetched_at")
        if cached_at and datetime.now(timezone.utc) - cached_at < timedelta(seconds=REALTIME_PRICE_TTL_SECONDS):
            return cached.get("price")

    coin_map = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "USDT": "tether",
        "BNB": "binancecoin",
        "SOL": "solana",
        "USDC": "usd-coin",
        "XRP": "ripple",
        "DOGE": "dogecoin",
        "TON": "the-open-network",
        "ADA": "cardano",
        "AVAX": "avalanche-2",
        "TRX": "tron",
        "SHIB": "shiba-inu",
        "DOT": "polkadot",
        "LINK": "chainlink",
        "MATIC": "matic-network",
        "BCH": "bitcoin-cash",
        "ICP": "internet-computer",
        "NEAR": "near",
        "LTC": "litecoin",
        "UNI": "uniswap",
        "APT": "aptos",
        "LEO": "leo-token",
        "STX": "blockstack",
        "ETC": "ethereum-classic",
        "FIL": "filecoin",
        "HBAR": "hedera-hashgraph",
        "XLM": "stellar",
        "IMX": "immutable-x",
        "INJ": "injective-protocol",
        "OP": "optimism",
        "ATOM": "cosmos",
        "ARB": "arbitrum",
        "MKR": "maker",
        "VET": "vechain",
        "GRT": "the-graph",
        "RNDR": "render-token",
        "ALGO": "algorand",
        "AAVE": "aave",
        "SUI": "sui",
        "FTM": "fantom",
        "EGLD": "elrond-erd-2",
        "FLOW": "flow",
        "KAS": "kaspa",
        "THETA": "theta-token",
        "AXS": "axie-infinity",
        "NEO": "neo",
        "XTZ": "tezos"
    }
    coin_id = coin_map.get(symbol.upper(), symbol.lower())
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": coin_id, "vs_currencies": vs_currency.lower()}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        price = data.get(coin_id, {}).get(vs_currency.lower())
        price_val = float(price) if price is not None else None
        _REALTIME_PRICE_CACHE[cache_key] = {
            "fetched_at": datetime.now(timezone.utc),
            "price": price_val,
        }
        return price_val
    except Exception as exc:  # noqa: BLE001
        logger.error("Realtime price fetch failed for %s: %s", symbol, exc)
        return None


def select_direction_model_path(df_feat: pd.DataFrame) -> str:
    def _prefer_ubj(path: str) -> str:
        base, _ = os.path.splitext(path)
        ubj_path = base + ".ubj"
        return ubj_path if os.path.exists(ubj_path) else path

    has_onchain = any(col.startswith("onchain_") for col in df_feat.columns)
    if has_onchain:
        path = _prefer_ubj(XGB_DIRECTION_MODEL_WITH_ONCHAIN_PATH)
        if os.path.exists(path):
            logger.info("XGBoost model selected: with_onchain (%s)", path)
            return path
    else:
        path = _prefer_ubj(XGB_DIRECTION_MODEL_NO_ONCHAIN_PATH)
        if os.path.exists(path):
            logger.info("XGBoost model selected: no_onchain (%s)", path)
            return path

    default_path = _prefer_ubj(XGB_DIRECTION_MODEL_PATH)
    logger.info("XGBoost model selected: default (%s)", default_path)
    return default_path


# ---------- Data Provider (OHLCV daily) ----------

class CryptoDataProvider:
    """
    Fetch OHLCV daily data from CryptoCompare (primary) with CoinDesk as backup.
    Clean and minimal implementation.
    """

    API_KEY = "537456fe97c8d896875c910fbe86a4882134a6bf5597202c5495d35d9d720cf5"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or self.API_KEY

    def get_daily_ohlcv(self, symbol: str, vs_currency: str = "USD", days: int = 365) -> pd.DataFrame:
        """
        Fetch OHLCV data from CryptoCompare.
        Returns DataFrame with columns: ['Open', 'High', 'Low', 'Close', 'Volume']
        """
        try:
            return self._fetch_cryptocompare_daily(symbol, vs_currency, days)
        except Exception as e:
            logger.warning(f"CryptoCompare failed for {symbol}: {e} — trying CoinDesk backup")
            try:
                return coindesk_fetcher.get_ohlcv_data(symbol, vs_currency, days) if coindesk_fetcher else None
            except Exception as e2:
                logger.error(f"Both CryptoCompare and CoinDesk failed for {symbol}: {e2}")
                raise

    def _fetch_cryptocompare_daily(self, symbol: str, vs_currency: str, days: int) -> pd.DataFrame:
        """
        Fallback khi yfinance lỗi: dùng CryptoCompare histoday (không có adjusted).
        """
        url = "https://min-api.cryptocompare.com/data/v2/histoday"
        target_days = max(1, days)
        api_key = self.api_key or API_KEY

        try:
            chunks: list[pd.DataFrame] = []
            fetched = 0
            to_ts = int(datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp())

            while fetched < target_days:
                batch_limit = min(2000, target_days - fetched)
                params = {
                    "fsym": symbol.upper(),
                    "tsym": vs_currency.upper(),
                    "limit": batch_limit,
                    "toTs": to_ts,
                }
                if api_key:
                    params["api_key"] = api_key

                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                payload = resp.json()
                data = payload.get("Data", {}).get("Data", [])
                if not data:
                    break

                df_batch = pd.DataFrame(data)
                chunks.append(df_batch)
                fetched += len(df_batch)

                oldest_ts = df_batch["time"].min()
                to_ts = int(oldest_ts) - 86400  # lùi tiếp về quá khứ

                if len(df_batch) < batch_limit:
                    # server trả ít hơn yêu cầu => hết dữ liệu
                    break

            if not chunks:
                raise ValueError(f"Empty response from CryptoCompare for {symbol}")

            df = pd.concat(chunks, axis=0, ignore_index=True)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.drop_duplicates(subset="time", keep="last")
            df = df.set_index("time")
            df = df.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volumeto": "Volume",
                }
            )
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            df.sort_index(inplace=True)
            if len(df) > target_days:
                df = df.tail(target_days)
            logger.info("✓ CryptoCompare fallback: Fetched %d daily records for %s", len(df), symbol)
            return df
        except Exception as exc:  # noqa: BLE001
            logger.error("CryptoCompare fallback failed for %s: %s", symbol, exc)
            raise ValueError(f"Failed to fetch data for {symbol} from all sources: {exc}")

# ---------- CoinDesk API Data Fetcher ----------
class CoinDeskDataFetcher:
    """
    Fetch OHLCV and on-chain data from CoinDesk API.
    CoinDesk provides comprehensive crypto market data and blockchain metrics.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or COINDESK_API_KEY
        self.base_url = COINDESK_BASE_URL

    def get_ohlcv_data(
        self,
        symbol: str,
        vs_currency: str = "USD",
        days: int = 365
    ) -> pd.DataFrame | None:
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) data from CoinDesk.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            vs_currency: Currency to quote prices in (default: 'USD')
            days: Number of days of historical data to retrieve
            
        Returns:
            DataFrame with OHLCV data indexed by date, or None if fetch fails
        """
        try:
            # CoinDesk API endpoint for price data
            url = f"{self.base_url}/bpi/historical/close"
            
            end_date = datetime.now(timezone.utc).date()
            start_date = end_date - timedelta(days=days)
            
            params = {
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d"),
                "currency": vs_currency.upper()
            }
            
            if self.api_key:
                params["api_key"] = self.api_key
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if "bpi" not in data:
                logger.warning(f"❌ CoinDesk API returned unexpected format for {symbol}")
                return None
            
            # Convert to DataFrame
            bpi_data = data["bpi"]
            dates = []
            prices = []
            
            for date_str, price in bpi_data.items():
                dates.append(pd.to_datetime(date_str, utc=True))
                prices.append(price)
            
            df = pd.DataFrame({
                "Close": prices,
                "Date": dates
            }).set_index("Date").sort_index()
            
            # For OHLCV, use close price as a proxy for all OHLC
            # This is a limitation of CoinDesk's basic API tier
            df["Open"] = df["Close"]
            df["High"] = df["Close"]
            df["Low"] = df["Close"]
            df["Volume"] = 0  # CoinDesk basic tier doesn't provide volume
            
            # Reorder columns to standard OHLCV format
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            
            logger.info(
                f"✅ CoinDesk: Successfully retrieved {len(df)} OHLCV records "
                f"for {symbol} ({start_date} to {end_date})"
            )
            
            return df
            
        except requests.exceptions.Timeout:
            logger.warning(f"❌ CoinDesk API timeout for {symbol}")
            return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"❌ CoinDesk API request error for {symbol}: {e}")
            return None
        except Exception as e:
            logger.warning(f"❌ CoinDesk OHLCV fetch error for {symbol}: {e}")
            return None

    def get_onchain_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame | None:
        """
        Fetch on-chain blockchain metrics from CoinDesk.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            start_date: Start date in format 'YYYY-MM-DD'
            end_date: End date in format 'YYYY-MM-DD'
            
        Returns:
            DataFrame with on-chain metrics indexed by date, or None if unavailable
        """
        try:
            # CoinDesk on-chain data endpoint (if available)
            # Note: CoinDesk's public API may have limited on-chain endpoints
            # For full on-chain data, you may need a premium subscription
            
            url = f"{self.base_url}/data/chain/addresses"
            
            params = {
                "asset": symbol.lower(),
                "start": start_date,
                "end": end_date
            }
            
            if self.api_key:
                params["api_key"] = self.api_key
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if not data or "data" not in data:
                logger.info(f"❌ No on-chain data available from CoinDesk for {symbol}")
                return None
            
            # Process on-chain data
            records = data.get("data", [])
            df = pd.DataFrame(records)
            
            if "date" in df.columns:
                df["Date"] = pd.to_datetime(df["date"], utc=True)
                df = df.set_index("Date").drop(columns=["date"], errors="ignore")
                df = df.sort_index()
                
                logger.info(
                    f"✅ CoinDesk: Successfully retrieved {len(df)} on-chain metrics "
                    f"for {symbol} ({start_date} to {end_date})"
                )
                
                return df
            
            return None
            
        except requests.exceptions.Timeout:
            logger.warning(f"❌ CoinDesk on-chain API timeout for {symbol}")
            return None
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.info(f"❌ CoinDesk on-chain data not available for {symbol}")
            else:
                logger.warning(f"❌ CoinDesk API HTTP error for {symbol}: {e}")
            return None
        except Exception as e:
            logger.warning(f"❌ CoinDesk on-chain fetch error for {symbol}: {e}")
            return None

# ---------- Sentiment Data Provider (Alpha Vantage / GNews + FinBERT) ----------

import re
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone
import logging

logger = logging.getLogger(__name__)


class AlphaVantageSentimentDataProvider:
    """
    Fetch historical crypto news sentiment from Alpha Vantage NEWS_SENTIMENT API
    using yearly backward pagination logic (Jupyter-style).
    """

    BASE_URL = "https://www.alphavantage.co/query"
    MAX_LIMIT = 1000
    CHUNK_DAYS = 90
    MAX_RETRIES = 3

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self.available = bool(api_key)
        self._article_cache = {}
        self.provider_name = "alpha_vantage"
        self.init_error = None

        if not self.available:
            self.init_error = "ALPHA_VANTAGE_API_KEY not configured"
            logger.warning("ALPHA_VANTAGE_API_KEY not configured. Sentiment will be skipped.")

    def _fetch_news_dataframe(
        self,
        symbol: str,
        total_years: int = 1
    ) -> pd.DataFrame:
        """
        Fetch raw news sentiment data year-by-year.
        Logic mirrors Jupyter implementation exactly.
        """

        if not self.available:
            logger.warning("API key not available. Returning empty DataFrame.")
            return pd.DataFrame()

        all_records = []
        logger.info("System time now: %s (UTC: %s)", datetime.now(), datetime.now(timezone.utc))
        global_end_date = datetime.now(timezone.utc)

        logger.info("Start fetching %s years of sentiment for %s", total_years, symbol)

        for year_idx in range(total_years):

            year_start_dt = global_end_date - timedelta(days=(year_idx + 1) * 365)
            year_end_dt = global_end_date - timedelta(days=year_idx * 365)

            logger.info("Year window %s: %s -> %s", year_idx, year_start_dt.date(), year_end_dt.date())
            year_records = 0

            chunk_end = year_end_dt
            while chunk_end > year_start_dt:
                chunk_start = max(year_start_dt, chunk_end - timedelta(days=self.CHUNK_DAYS))
                chunk_records = 0

                time_from = chunk_start.strftime("%Y%m%dT%H%M")
                current_time_to = chunk_end.strftime("%Y%m%dT%H%M")

                logger.info("Chunk window: %s -> %s", chunk_start.date(), chunk_end.date())

                while True:
                    params = {
                        "function": "NEWS_SENTIMENT",
                        "tickers": f"CRYPTO:{symbol.upper()}",
                        "time_from": time_from,
                        "time_to": current_time_to,
                        "limit": self.MAX_LIMIT,
                        "apikey": self.api_key
                    }

                    data = None
                    for attempt in range(1, self.MAX_RETRIES + 1):
                        try:
                            response = requests.get(self.BASE_URL, params=params, timeout=10)
                            response.raise_for_status()
                            data = response.json()
                            break
                        except requests.exceptions.RequestException as e:
                            logger.warning(
                                "Request failed (attempt %s/%s). Sleeping 30s. Error: %s",
                                attempt,
                                self.MAX_RETRIES,
                                e
                            )
                            time.sleep(30)
                    if data is None:
                        logger.warning(
                            "Request failed after %s attempts. Skipping window %s -> %s",
                            self.MAX_RETRIES,
                            time_from,
                            current_time_to
                        )
                        break

                    # Rate limit
                    if "Note" in data:
                        logger.warning(
                            "Alpha Vantage rate limit hit for %s. Detail=%s. Sleeping 30s.",
                            symbol,
                            data.get("Note")
                        )
                        time.sleep(30)
                        continue
                    if "Error Message" in data or "Information" in data:
                        logger.warning(
                            "Alpha Vantage response issue for %s. time_from=%s time_to=%s keys=%s detail=%s",
                            symbol,
                            time_from,
                            current_time_to,
                            list(data.keys()),
                            data.get("Error Message") or data.get("Information")
                        )
                        break

                    feed = data.get("feed", [])
                    if not feed:
                        logger.info(
                            "No articles returned. time_from=%s time_to=%s keys=%s",
                            time_from,
                            current_time_to,
                            list(data.keys())
                        )
                        logger.debug("No more articles in this window.")
                        break

                    batch_count = 0

                    for item in feed:
                        try:
                            published_dt = datetime.strptime(
                                item["time_published"], "%Y%m%dT%H%M%S"
                            ).replace(tzinfo=timezone.utc)

                            ticker_sentiment = [
                                t for t in item.get("ticker_sentiment", [])
                                if t.get("ticker") == f"CRYPTO:{symbol.upper()}"
                            ]

                            if not ticker_sentiment:
                                continue

                            sentiment_score = float(
                                ticker_sentiment[0].get("ticker_sentiment_score", 0)
                            )
                            if abs(sentiment_score) > 1:
                                if abs(sentiment_score) <= 100:
                                    sentiment_score = sentiment_score / 100.0
                                sentiment_score = max(min(sentiment_score, 1.0), -1.0)

                            summary_text = item.get("summary") or item.get("description") or ""

                            all_records.append({
                                "Ngày": published_dt,
                                "Sentiment_Score": sentiment_score,
                                "Tiêu đề": item.get("title", "").strip(),
                                "Nguồn": item.get("source", ""),
                                "URL": item.get("url", ""),
                                "Mô tả": summary_text.strip()
                            })

                            batch_count += 1
                            chunk_records += 1
                            year_records += 1

                        except Exception:
                            continue

                    last_item_dt = datetime.strptime(
                        feed[-1]["time_published"], "%Y%m%dT%H%M%S"
                    ).replace(tzinfo=timezone.utc)

                    logger.info("Fetched %s articles. Current pointer %s", batch_count, last_item_dt)

                    # Điều kiện dừng giống notebook
                    if last_item_dt <= chunk_start or len(feed) < 10:
                        break

                    current_time_to = (last_item_dt - timedelta(seconds=1)).strftime("%Y%m%dT%H%M")
                    time.sleep(1)

                chunk_end = chunk_start - timedelta(seconds=1)
                logger.info(
                    "Chunk summary: %s -> %s | chunk=%s year=%s total=%s",
                    chunk_start.date(),
                    chunk_end.date(),
                    chunk_records,
                    year_records,
                    len(all_records)
                )
                time.sleep(30)

            if year_records == 0:
                logger.info(
                    "No articles fetched in year window %s: %s -> %s",
                    year_idx,
                    year_start_dt.date(),
                    year_end_dt.date()
                )
            else:
                logger.info(
                    "Year summary %s: %s -> %s | year=%s total=%s",
                    year_idx,
                    year_start_dt.date(),
                    year_end_dt.date(),
                    year_records,
                    len(all_records)
                )

        df = pd.DataFrame(all_records)

        if not df.empty:
            df = (
                df.drop_duplicates(subset=["Tiêu đề", "Ngày"])
                  .sort_values("Ngày", ascending=False)
                  .reset_index(drop=True)
            )

        self._article_cache[symbol] = df
        logger.info("Total articles collected: %s", len(df))

        return df

    def _fetch_news_with_time_window(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> list[dict]:
        """
        Efficiently fetch news for a specific time window WITHOUT fetching months/years of data.
        This avoids the year-long fetch issue by only hitting the API for the requested period.
        """
        if not self.available:
            logger.warning("API key not available. Returning empty news list.")
            return []

        all_records = []
        logger.info("Fetching sentiment for %s from %s to %s", symbol, start_date.date(), end_date.date())

        chunk_end = end_date
        while chunk_end > start_date:
            chunk_start = max(start_date, chunk_end - timedelta(days=self.CHUNK_DAYS))
            chunk_records = 0

            time_from = chunk_start.strftime("%Y%m%dT%H%M")
            current_time_to = chunk_end.strftime("%Y%m%dT%H%M")

            logger.info("Chunk window: %s -> %s", chunk_start.date(), chunk_end.date())

            while True:
                params = {
                    "function": "NEWS_SENTIMENT",
                    "tickers": f"CRYPTO:{symbol.upper()}",
                    "time_from": time_from,
                    "time_to": current_time_to,
                    "limit": self.MAX_LIMIT,
                    "apikey": self.api_key
                }

                data = None
                for attempt in range(1, self.MAX_RETRIES + 1):
                    try:
                        response = requests.get(self.BASE_URL, params=params, timeout=10)
                        response.raise_for_status()
                        data = response.json()
                        break
                    except requests.exceptions.RequestException as e:
                        logger.warning(
                            "Request failed (attempt %s/%s). Sleeping 30s. Error: %s",
                            attempt,
                            self.MAX_RETRIES,
                            e
                        )
                        time.sleep(30)
                
                if data is None:
                    logger.warning(
                        "Request failed after %s attempts. Skipping window %s -> %s",
                        self.MAX_RETRIES,
                        time_from,
                        current_time_to
                    )
                    break

                # Rate limit
                if "Note" in data:
                    logger.warning(
                        "Alpha Vantage rate limit hit for %s. Detail=%s. Sleeping 30s.",
                        symbol,
                        data.get("Note")
                    )
                    time.sleep(30)
                    continue
                if "Error Message" in data or "Information" in data:
                    logger.warning(
                        "Alpha Vantage response issue for %s. time_from=%s time_to=%s keys=%s detail=%s",
                        symbol,
                        time_from,
                        current_time_to,
                        list(data.keys()),
                        data.get("Error Message") or data.get("Information")
                    )
                    break

                feed = data.get("feed", [])
                if not feed:
                    logger.info(
                        "No articles returned. time_from=%s time_to=%s",
                        time_from,
                        current_time_to
                    )
                    break

                batch_count = 0

                for item in feed:
                    try:
                        published_dt = datetime.strptime(
                            item["time_published"], "%Y%m%dT%H%M%S"
                        ).replace(tzinfo=timezone.utc)

                        # Only include articles within the requested time window
                        if published_dt < start_date or published_dt > end_date:
                            continue

                        ticker_sentiment = [
                            t for t in item.get("ticker_sentiment", [])
                            if t.get("ticker") == f"CRYPTO:{symbol.upper()}"
                        ]

                        if not ticker_sentiment:
                            continue

                        sentiment_score = float(
                            ticker_sentiment[0].get("ticker_sentiment_score", 0)
                        )
                        if abs(sentiment_score) > 1:
                            if abs(sentiment_score) <= 100:
                                sentiment_score = sentiment_score / 100.0
                            sentiment_score = max(min(sentiment_score, 1.0), -1.0)

                        summary_text = item.get("summary") or item.get("description") or ""

                        all_records.append({
                            "Ngày": published_dt,
                            "Sentiment_Score": sentiment_score,
                            "Tiêu đề": item.get("title", "").strip(),
                            "Nguồn": item.get("source", ""),
                            "URL": item.get("url", ""),
                            "Mô tả": summary_text.strip()
                        })

                        batch_count += 1
                        chunk_records += 1

                    except Exception:
                        continue

                last_item_dt = datetime.strptime(
                    feed[-1]["time_published"], "%Y%m%dT%H%M%S"
                ).replace(tzinfo=timezone.utc)

                logger.info("Fetched %s articles. Current pointer %s", batch_count, last_item_dt)

                # Stop conditions
                if last_item_dt <= chunk_start or len(feed) < 10:
                    break

                current_time_to = (last_item_dt - timedelta(seconds=1)).strftime("%Y%m%dT%H%M")
                time.sleep(1)

            chunk_end = chunk_start - timedelta(seconds=1)
            logger.info(
                "Chunk summary: %s -> %s | chunk=%s total=%s",
                chunk_start.date(),
                chunk_end.date(),
                chunk_records,
                len(all_records)
            )
            time.sleep(30)

        # Convert to articles format
        def _label_from_score(score: float) -> str:
            if score > 0.15:
                return "positive"
            if score < -0.15:
                return "negative"
            return "neutral"

        articles = []
        for record in all_records:
            sentiment_score = float(record.get("Sentiment_Score", 0))
            articles.append({
                "title": record.get("Tiêu đề", ""),
                "url": record.get("URL", ""),
                "source": record.get("Nguồn", ""),
                "published": record.get("Ngày"),
                "published_date": record.get("Ngày"),
                "sentiment_score": sentiment_score,
                "sentiment_label": _label_from_score(sentiment_score),
                "summary": record.get("Mô tả", ""),
                "description": record.get("Mô tả", "")
            })

        logger.info("Total articles for %s days: %s", 
                   (end_date - start_date).days, len(articles))
        
        return articles

    def fetch_news_with_sentiment(
        self,
        symbol: str,
        days: int | None = None,
        total_years: int | None = None
    ) -> list[dict]:
        """
        Fetch sentiment news articles and return a list of dicts for display/analysis.
        If days is provided, fetches only the last N days (not months/years).
        """

        if not self.available:
            logger.warning("API key not available. Returning empty news list.")
            return []

        # FIX: When days is specified, fetch ONLY that many days, not convert to years
        if days is not None and total_years is None:
            # Fetch data for the requested days only, not a full year
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            logger.info("Fetching last %s days of sentiment for %s (%s to %s)", 
                       days, symbol, start_date.date(), end_date.date())
            
            return self._fetch_news_with_time_window(symbol, start_date, end_date)

        # Legacy behavior: if total_years specified, use old method
        if days is None and total_years is None:
            total_years = 1
        
        df = self._fetch_news_dataframe(symbol, total_years=total_years or 1)
        if df.empty:
            return []

        def _label_from_score(score: float) -> str:
            if score > 0.15:
                return "positive"
            if score < -0.15:
                return "negative"
            return "neutral"

        articles = []
        for _, row in df.iterrows():
            sentiment_score = float(row.get("Sentiment_Score", 0))
            articles.append({
                "title": row.get("Tiêu đề", ""),
                "url": row.get("URL", ""),
                "source": row.get("Nguồn", ""),
                "published": row.get("Ngày"),
                "published_date": row.get("Ngày"),
                "sentiment_score": sentiment_score,
                "sentiment_label": _label_from_score(sentiment_score),
                "summary": row.get("Mô tả", ""),
                "description": row.get("Mô tả", "")
            })

        return articles

    def summarize_daily_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate daily sentiment with threshold-based labeling.
        Exact logic from Jupyter notebook.
        """

        if df.empty:
            logger.warning("Empty DataFrame. No sentiment to summarize.")
            return pd.DataFrame()

        df = df.copy()
        df["date"] = pd.to_datetime(df["Ngày"]).dt.date

        def count_positive(scores):
            return (scores > 0.15).sum()

        def count_negative(scores):
            return (scores < -0.15).sum()

        daily = (
            df.groupby("date")
              .agg(
                  num_positive=("Sentiment_Score", count_positive),
                  num_negative=("Sentiment_Score", count_negative),
                  avg_sentiment_score=("Sentiment_Score", "mean"),
                  total_news=("Sentiment_Score", "count")
              )
              .reset_index()
        )

        daily["num_neutral"] = (
            daily["total_news"]
            - daily["num_positive"]
            - daily["num_negative"]
        )

        conditions = [
            daily["avg_sentiment_score"] > 0.05,
            daily["avg_sentiment_score"] < -0.05
        ]
        choices = ["Positive", "Negative"]

        daily["status"] = np.select(conditions, choices, default="Neutral")

        daily["avg_sentiment_score"] = daily["avg_sentiment_score"].round(4)

        daily = daily.sort_values("date", ascending=False).reset_index(drop=True)

        return daily

    def fetch_daily_sentiment(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """
        Fetch daily sentiment scores for last N days.
        Returns DataFrame indexed by date with sentiment_score and articles_count.
        """
        if not self.available:
            logger.warning("API key not available. Returning empty daily sentiment.")
            return pd.DataFrame()

        df_articles = self._article_cache.get(symbol)
        if df_articles is None or df_articles.empty:
            df_articles = self._fetch_news_dataframe(symbol, total_years=max(1, int(np.ceil(days / 365))))

        if df_articles.empty:
            return pd.DataFrame()

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        df_filtered = df_articles[df_articles["Ngày"] >= cutoff]
        if df_filtered.empty:
            return pd.DataFrame()

        daily = self.summarize_daily_sentiment(df_filtered)
        if daily.empty:
            return pd.DataFrame()

        daily = daily.rename(
            columns={
                "avg_sentiment_score": "sentiment_score",
                "total_news": "articles_count"
            }
        )
        daily["sentiment_score"] = daily["sentiment_score"].clip(-1.0, 1.0)
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.set_index("date").sort_index()

        return daily[["sentiment_score", "articles_count", "num_positive", "num_negative", "num_neutral"]]

    def get_cached_articles(self, symbol: str) -> pd.DataFrame:
        return self._article_cache.get(symbol, pd.DataFrame())


def create_sentiment_provider(provider_name: str | None) -> object | None:
    if not provider_name:
        return None
    normalized = provider_name.strip().lower()
    if normalized in {"none", "off", "disable", "disabled"}:
        return None
    if normalized in {"alpha", "alpha_vantage", "alphavantage"}:
        return AlphaVantageSentimentDataProvider(api_key=ALPHA_VANTAGE_API_KEY)
    if normalized in {"gnews", "finbert"}:
        return GNewsSentimentDataProvider(days_default=7)
    logger.warning("Unknown SENTIMENT_PROVIDER '%s'. Expected 'gnews', 'alpha', or 'off'.", provider_name)
    return None


class GNewsSentimentDataProvider:
    """
    Fetch crypto news sentiment using GNews + FinBERT.
    Only pulls recent news (default 7 days).
    """

    def __init__(self, days_default: int = 7):
        self.days_default = days_default
        self.available = True
        self._article_cache = {}
        self.provider_name = "gnews"
        self.init_error = None
        self._torch = None
        self._Article = None
        self._BeautifulSoup = None

        try:
            from gnews import GNews
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            from newspaper import Article
            from bs4 import BeautifulSoup
        except ImportError as exc:
            self.init_error = f"Missing dependency: {exc}"
            logger.warning("GNews/FinBERT dependencies missing: %s", exc)
            self.available = False
            return

        self._torch = torch
        self._Article = Article
        self._BeautifulSoup = BeautifulSoup

        try:
            self.gnews = GNews(language="en", country="US", max_results=50)
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.model.eval()
        except Exception as exc:  # noqa: BLE001
            self.init_error = f"Init failure: {exc}"
            logger.warning("Failed to initialize FinBERT or GNews: %s", exc)
            self.available = False

    @staticmethod
    def _coin_name(symbol: str) -> str | None:
        name_map = {
            "BTC": "Bitcoin",
            "ETH": "Ethereum",
            "BNB": "Binance",
            "SOL": "Solana",
            "XRP": "XRP",
            "ADA": "Cardano",
            "DOGE": "Dogecoin",
            "DOT": "Polkadot",
            "AVAX": "Avalanche",
            "MATIC": "Polygon",
            "LTC": "Litecoin",
            "LINK": "Chainlink",
            "ATOM": "Cosmos",
            "TRX": "Tron",
            "FIL": "Filecoin",
            "TON": "Toncoin",
            "ARB": "Arbitrum",
            "OP": "Optimism",
            "NEAR": "Near Protocol",
            "UNI": "Uniswap",
        }
        return name_map.get(symbol.upper())

    @staticmethod
    def _build_query(symbol: str) -> str:
        name = GNewsSentimentDataProvider._coin_name(symbol)
        if name:
            return f"{symbol} {name} crypto"
        return f"{symbol} crypto"

    @staticmethod
    def _is_relevant_item(symbol: str, item: dict) -> bool:
        name = GNewsSentimentDataProvider._coin_name(symbol)
        title = (item.get("title") or "").lower()
        desc = (item.get("description") or "").lower()
        text = f"{title} {desc}"
        symbol_lc = symbol.lower()
        keywords = [symbol_lc]
        if name:
            keywords.append(name.lower())
        match = any(k in text for k in keywords)
        if not match:
            return False
        if name is None and len(symbol) <= 3:
            return "crypto" in text or "bitcoin" in text or "ethereum" in text
        return True

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text or "")
        text = re.sub(r"http\S+", "", text)
        return text.strip()

    @staticmethod
    def _is_valid_article(text: str, min_words: int = 50) -> bool:
        if not text:
            return False
        return len(text.split()) >= min_words

    def _extract_full_article(self, url: str) -> str | None:
        try:
            article = self._Article(url)
            article.download()
            article.parse()
            text = self._clean_text(article.text)
            if self._is_valid_article(text):
                return text
        except Exception:
            pass

        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            soup = self._BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")
            text = self._clean_text(" ".join(p.get_text() for p in paragraphs))
            if self._is_valid_article(text):
                return text
        except Exception:
            pass

        return None

    def _finbert_sentiment(self, text: str, max_length: int = 512) -> tuple[str, float]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        )

        with self._torch.no_grad():
            outputs = self.model(**inputs)

        scores = self._torch.nn.functional.softmax(outputs.logits, dim=1).numpy()[0]
        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        sentiment_label = sentiment_map[int(np.argmax(scores))]
        sentiment_score = float(scores[2] - scores[0])
        return sentiment_label, sentiment_score

    def fetch_news_with_sentiment(self, symbol: str, days: int | None = None) -> list[dict]:
        if not self.available:
            logger.warning("Sentiment provider not available. Returning empty news list.")
            return []

        days = days or self.days_default
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)

        self.gnews.start_date = (start_date.year, start_date.month, start_date.day)
        self.gnews.end_date = (end_date.year, end_date.month, end_date.day)

        query = self._build_query(symbol)
        logger.info("Fetching GNews sentiment for %s (%s days) query='%s'...", symbol, days, query)
        news_items = self.gnews.get_news(query) or []
        logger.info("GNews returned %s raw items for %s", len(news_items), symbol)
        if news_items:
            before = len(news_items)
            news_items = [item for item in news_items if self._is_relevant_item(symbol, item)]
            logger.info("GNews filtered to %s relevant items for %s", len(news_items), symbol)

        records = []
        for item in news_items:
            description = item.get("description") or ""
            sentiment_label = None
            sentiment_score = None
            text_for_sentiment = ""

            if len(description.split()) >= 3:
                text_for_sentiment = description
            else:
                full_text = self._extract_full_article(item.get("url", ""))
                if full_text:
                    text_for_sentiment = full_text

            if text_for_sentiment:
                try:
                    sentiment_label, sentiment_score = self._finbert_sentiment(text_for_sentiment)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("FinBERT failed for '%s': %s", item.get("title", ""), exc)

            published_raw = item.get("published date") or item.get("published_date")
            published_dt = pd.to_datetime(published_raw, errors="coerce")
            if pd.isna(published_dt):
                published_dt = datetime.now()

            source = item.get("publisher", {})
            if isinstance(source, dict):
                source = source.get("title", "")

            records.append({
                "Ngày": published_dt,
                "Sentiment_Score": sentiment_score,
                "Tiêu đề": item.get("title", ""),
                "Nguồn": source or "",
                "URL": item.get("url", ""),
                "Mô tả": description
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df = (
                df.drop_duplicates(subset=["Tiêu đề", "Ngày"])
                  .sort_values("Ngày", ascending=False)
                  .reset_index(drop=True)
            )

        self._article_cache[symbol] = df
        logger.info("Total GNews articles collected: %s", len(df))

        def _label_from_score(score: float | None) -> str:
            if score is None:
                return "neutral"
            if score > 0.15:
                return "positive"
            if score < -0.15:
                return "negative"
            return "neutral"

        articles = []
        for _, row in df.iterrows():
            sentiment_score = row.get("Sentiment_Score")
            sentiment_score = None if pd.isna(sentiment_score) else float(sentiment_score)
            articles.append({
                "title": row.get("Tiêu đề", ""),
                "url": row.get("URL", ""),
                "source": row.get("Nguồn", ""),
                "published": row.get("Ngày"),
                "published_date": row.get("Ngày"),
                "sentiment_score": sentiment_score,
                "sentiment_label": _label_from_score(sentiment_score),
                "summary": row.get("Mô tả", ""),
                "description": row.get("Mô tả", "")
            })

        return articles

    # ======================================================
    # 2. DAILY AGGREGATION + LABELING (GIỐNG NOTEBOOK)
    # ======================================================
    def summarize_daily_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate daily sentiment with threshold-based labeling.
        Exact logic from Jupyter notebook.
        """

        if df.empty:
            logger.warning("Empty DataFrame. No sentiment to summarize.")
            return pd.DataFrame()

        df = df.copy()
        df["date"] = pd.to_datetime(df["Ngày"]).dt.date

        def count_positive(scores):
            return (scores > 0.15).sum()

        def count_negative(scores):
            return (scores < -0.15).sum()

        daily = (
            df.groupby("date")
              .agg(
                  num_positive=("Sentiment_Score", count_positive),
                  num_negative=("Sentiment_Score", count_negative),
                  avg_sentiment_score=("Sentiment_Score", "mean"),
                  total_news=("Sentiment_Score", "count")
              )
              .reset_index()
        )

        daily["num_neutral"] = (
            daily["total_news"]
            - daily["num_positive"]
            - daily["num_negative"]
        )

        conditions = [
            daily["avg_sentiment_score"] > 0.05,
            daily["avg_sentiment_score"] < -0.05
        ]
        choices = ["Positive", "Negative"]

        daily["status"] = np.select(conditions, choices, default="Neutral")

        daily["avg_sentiment_score"] = daily["avg_sentiment_score"].round(4)

        daily = daily.sort_values("date", ascending=False).reset_index(drop=True)

        return daily

    def fetch_daily_sentiment(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """
        Fetch daily sentiment scores for last N days.
        Returns DataFrame indexed by date with sentiment_score and articles_count.
        """
        if not self.available:
            logger.warning("Sentiment provider not available. Returning empty daily sentiment.")
            return pd.DataFrame()

        days = days or self.days_default
        articles = self.fetch_news_with_sentiment(symbol, days=days)
        if not articles:
            return pd.DataFrame()

        df_articles = pd.DataFrame({
            "Ngày": [item.get("published_date") for item in articles],
            "Sentiment_Score": [item.get("sentiment_score") for item in articles]
        })
        df_articles = df_articles.dropna(subset=["Sentiment_Score"])
        if df_articles.empty:
            return pd.DataFrame()

        daily = self.summarize_daily_sentiment(df_articles)
        if daily.empty:
            return pd.DataFrame()

        daily = daily.rename(
            columns={
                "avg_sentiment_score": "sentiment_score",
                "total_news": "articles_count"
            }
        )
        daily["sentiment_score"] = daily["sentiment_score"].clip(-1.0, 1.0)
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.set_index("date").sort_index()

        return daily[["sentiment_score", "articles_count", "num_positive", "num_negative", "num_neutral"]]

    # ======================================================
    # 3. CACHE ACCESS
    # ======================================================
    def get_cached_articles(self, symbol: str) -> pd.DataFrame:
        return self._article_cache.get(symbol, pd.DataFrame())


# ---------- Macro Data Provider (FRED API) ----------

class MacroDataProvider:
    """
    Fetch macroeconomic data from FRED (Federal Reserve Economic Data) API.
    """

    def __init__(self, api_key: str):
        """
        Initialize FRED API client.
        
        Args:
            api_key: FRED API key from config
        """
        try:
            from fredapi import Fred
            self.fred = Fred(api_key=api_key)
            self.available = True
        except ImportError:
            logger.warning("fredapi library not installed. Macro data will be skipped.")
            self.fred = None
            self.available = False
        except Exception as e:
            logger.warning(f"Failed to initialize FRED client: {e}")
            self.fred = None
            self.available = False

    def get_macro_data(self, start_date: str = None, days: int = 30) -> pd.DataFrame:
        """
        Fetch macroeconomic indicators from FRED.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            days: Number of days to look back if start_date not provided
            
        Returns:
            DataFrame with macro indicators indexed by date
        """
        if not self.available or not self.fred:
            logger.warning("FRED provider not available. Returning empty macro data.")
            return pd.DataFrame()

        # Calculate start_date if not provided
        if start_date is None:
            start_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

        # FRED series to fetch
        macro_series = {
            'sp500': 'SP500',              # S&P 500
            'vix': 'VIXCLS',               # VIX (Volatility Index)
            'dxy': 'DTWEXBGS',             # US Dollar Index
            'goldprice': 'GOLDAMGBD228NLBM',  # Gold Price
            'brentoil': 'DCOILBRENTEU',    # Brent Crude Oil
            'dowjones': 'DJIA',            # Dow Jones Industrial Average
        }

        df_list = []
        fetched_series = []

        for name, series_id in macro_series.items():
            try:
                logger.debug(f"Fetching FRED series: {name} ({series_id})")
                data = self.fred.get_series(series_id, observation_start=start_date)
                
                if data is not None:
                    df_list.append(pd.DataFrame({name: data}))
                    fetched_series.append(name)
                else:
                    logger.debug(f"⚠ Warning: No data returned for series ID '{series_id}' (Name: {name}). It will be skipped.")
            except ValueError as e:
                logger.debug(f"ValueError fetching series ID '{series_id}' (Name: {name}): {e}. It will be skipped.")
            except Exception as e:
                logger.debug(f"An unexpected error occurred while fetching series ID '{series_id}' (Name: {name}): {e}. It will be skipped.")

        if not df_list:
            logger.warning("No macro data can be fetched. Returning an empty DataFrame.")
            return pd.DataFrame()

        # Combine all series based on date
        macro_df = pd.concat(df_list, axis=1)
        macro_df.index.name = 'Date'

        # FRED returns data daily but may be missing weekends (Market closed)
        # Forward fill to handle holidays/weekends
        macro_df = macro_df.ffill()

        logger.info(f"✓ Macro data: Fetched {len(fetched_series)} indicators ({', '.join(fetched_series)})")
        return macro_df


def merge_crypto_macro(df_crypto: pd.DataFrame, df_macro: pd.DataFrame) -> pd.DataFrame:
    """
    Merge crypto OHLCV data with macro indicators by date.
    Uses left join to preserve crypto data, fills missing macro values forward.
    """
    if df_macro.empty:
        return df_crypto
    
    # Handle timezone mismatch: convert both to timezone-naive (UTC) before joining
    df_crypto_tz = df_crypto.copy()
    df_macro_tz = df_macro.copy()
    
    # Convert timezone-aware indices to timezone-naive
    if hasattr(df_crypto_tz.index, 'tz') and df_crypto_tz.index.tz is not None:
        df_crypto_tz.index = df_crypto_tz.index.tz_localize(None)
    if hasattr(df_macro_tz.index, 'tz') and df_macro_tz.index.tz is not None:
        df_macro_tz.index = df_macro_tz.index.tz_localize(None)
    
    merged = df_crypto_tz.join(df_macro_tz, how='left')
    merged.ffill(inplace=True)
    return merged


def compute_indicators(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators for the given OHLCV data.
    
    Indicators calculated:
    - RSI_10, RSI_14: Relative Strength Index
    - ROC_12: Rate of Change
    - STD_DEV_20: Standard Deviation
    - CCI_20: Commodity Channel Index
    - H_L: High - Low
    - H_CP: High - Close Previous
    - MACD_Line_6_20, MACD_Signal_6_20: MACD
    - ATR_14: Average True Range
    - OBV: On-Balance Volume
    - EMA_5: Exponential Moving Average
    - SMA_5: Simple Moving Average
    - STOCH_K, STOCH_D: Stochastic Oscillator
    """
    df = df_raw.copy()
    df = df.ffill().bfill()
    
    # Ensure required columns exist
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # 1. RSI (Relative Strength Index)
    def calculate_rsi(series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['RSI_10'] = calculate_rsi(df['Close'], 10)
    df['RSI_14'] = calculate_rsi(df['Close'], 14)
    
    # 2. ROC (Rate of Change)
    df['ROC_12'] = (df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12) * 100
    
    # 3. Standard Deviation
    df['STD_DEV_20'] = df['Close'].rolling(window=20).std()
    
    # 4. CCI (Commodity Channel Index)
    def calculate_cci(high, low, close, period=20):
        tp = (high + low + close) / 3  # Typical Price
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (tp - sma_tp) / (0.015 * mad)
        return cci
    
    df['CCI_20'] = calculate_cci(df['High'], df['Low'], df['Close'], 20)
    
    # 5. H_L and H_CP
    df['H_L'] = df['High'] - df['Low']
    df['H_CP'] = df['High'] - df['Close'].shift(1)
    
    # 6. MACD (Moving Average Convergence Divergence)
    exp1 = df['Close'].ewm(span=6, adjust=False).mean()
    exp2 = df['Close'].ewm(span=20, adjust=False).mean()
    df['MACD_Line_6_20'] = exp1 - exp2
    df['MACD_Signal_6_20'] = df['MACD_Line_6_20'].ewm(span=9, adjust=False).mean()
    
    # 7. ATR (Average True Range)
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    df['ATR_14'] = calculate_atr(df['High'], df['Low'], df['Close'], 14)
    
    # 8. OBV (On-Balance Volume)
    obv = np.where(df['Close'] > df['Close'].shift(1), df['Volume'],
                   np.where(df['Close'] < df['Close'].shift(1), -df['Volume'], 0))
    df['OBV'] = np.cumsum(obv)
    
    # 9. EMA (Exponential Moving Average)
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    
    # 10. SMA (Simple Moving Average)
    df['SMA_5'] = df['Close'].rolling(window=5).mean()

    # 12. Bollinger Bands (20)
    if 'BB_Mid_20' not in df.columns:
        bb_mid = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Mid_20'] = bb_mid
        df['BB_Upper_20'] = bb_mid + (2 * bb_std)
        df['BB_Lower_20'] = bb_mid - (2 * bb_std)

    # 11. Stochastic Oscillator
    def calculate_stochastic(high, low, close, period=14, smooth_k=3, smooth_d=3):
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        k_raw = 100 * (close - lowest_low) / (highest_high - lowest_low)
        k = k_raw.rolling(window=smooth_k).mean()
        d = k.rolling(window=smooth_d).mean()
        
        return k, d
    
    stoch_k, stoch_d = calculate_stochastic(df['High'], df['Low'], df['Close'], 14, 3, 3)
    df['STOCH_K'] = stoch_k
    df['STOCH_D'] = stoch_d
    
    # Keep original OHLCV and add derived columns for backward compatibility
    if 'EMA_12' not in df.columns:
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    if 'EMA_26' not in df.columns:
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    if 'MACD_Histogram' not in df.columns:
        df['MACD_Histogram'] = df['MACD_Line_6_20'] - df['MACD_Signal_6_20']
    if 'Target_Return_1d' not in df.columns:
        df['Target_Return_1d'] = df['Close'].pct_change().shift(-1)
    if 'Log_return' not in df.columns:
        df['Log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    if 'SMA_14' not in df.columns:
        df['SMA_14'] = df['Close'].rolling(window=14).mean()
    
    # Fill NaN values
    df = df.ffill().bfill().fillna(0)
    
    return df


def build_feature_table(
    symbol: str,
    lookback_days: int,
    include_onchain: bool = True,
    include_macro: bool = True
) -> tuple[pd.DataFrame, dict]:
    """
    Gom toàn bộ dữ liệu (OHLCV, on-chain, macro, indicators, target)
    thành một bảng duy nhất để phục vụ cả DRL và ML.
    """
    cache_key = (symbol.upper(), lookback_days, include_onchain, include_macro)
    cached = _FEATURE_TABLE_CACHE.get(cache_key)
    if cached:
        cached_at = cached.get("fetched_at")
        if cached_at and datetime.now(timezone.utc) - cached_at < timedelta(seconds=CACHE_TTL_SECONDS):
            df_cached = cached.get("df_feat")
            info_cached = cached.get("feature_table_info")
            if isinstance(df_cached, pd.DataFrame):
                logger.info("✓ Using cached feature table for %s", symbol)
                return df_cached.copy(), dict(info_cached or {})
    if not data_provider:
        raise ValueError("Data provider chưa sẵn sàng")

    # Fetch OHLCV data
    logger.info(f"# Retrieving OHLCV data for {symbol}...")
    df_crypto = data_provider.get_daily_ohlcv(symbol, days=lookback_days)
    logger.info(f"✓ OHLCV data successfully retrieved: {len(df_crypto)} records for {symbol}")

    sources_used = {"ohlcv": True, "onchain": False, "sentiment": False, "macro": False}

    # Try to fetch on-chain data for supported coins
    if include_onchain and symbol.upper() in {"BTC", "ETH"}:
        try:
            logger.info(f"# Retrieving on-chain metrics for {symbol}...")
            df_onchain = get_onchain_history(
                symbol=symbol,
                start_date=df_crypto.index.min().strftime("%Y-%m-%d"),
                end_date=df_crypto.index.max().strftime("%Y-%m-%d")
            )
            df_crypto = merge_crypto_onchain(df_crypto, df_onchain)
            sources_used["onchain"] = True
            logger.info(f"✓ On-chain metrics merged successfully for {symbol}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"⚠ On-chain data fetch failed for {symbol}: {exc}")

    df_raw = df_crypto.copy()

    # Sentiment data is handled separately (not merged into the feature table)
    if sentiment_provider and sentiment_provider.available:
        logger.info("ℹ Sentiment data handled separately.")
    else:
        logger.info("⚠ Sentiment provider not available; skipping sentiment fetch.")
    sources_used["sentiment"] = False

    # Fetch and merge macro data from FRED
    if include_macro and macro_provider:
        try:
            logger.info("# Retrieving macroeconomic data...")
            df_macro = macro_provider.get_macro_data(days=lookback_days)
            if not df_macro.empty:
                df_raw = merge_crypto_macro(df_raw, df_macro)
                sources_used["macro"] = True
                logger.info(f"✓ Macro data merged successfully ({len(df_macro.columns)} indicators)")
            else:
                logger.info("ℹ No macro data available, proceeding without macro indicators")
                sources_used["macro"] = False
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"⚠ Macro data fetch failed: {exc}")
            sources_used["macro"] = False
    else:
        if not include_macro:
            logger.info("Macro data disabled via include_macro=False")
        else:
            logger.info("⚠ Macro provider not available, skipping macro fetch")
        sources_used["macro"] = False


    def _to_native(value):
        if isinstance(value, np.generic):
            return value.item()
        return value

    # Compute technical indicators
    logger.info(f"📊 Computing technical indicators for {symbol}...")
    df_feat = compute_indicators(df_raw)
    logger.info(f"✓ Technical indicators computed: {len(df_feat.columns)} features extracted")

    # Drop sentiment count columns before feeding into models
    drop_cols = ["articles_count", "num_positive", "num_negative", "num_neutral"]
    existing_drop_cols = [col for col in drop_cols if col in df_feat.columns]
    if existing_drop_cols:
        df_feat = df_feat.drop(columns=existing_drop_cols)

    latest_row = {}
    if not df_feat.empty:
        raw_latest = df_feat.tail(1).to_dict(orient="records")[0]
        latest_row = {k: _to_native(v) for k, v in raw_latest.items()}

    feature_table_info = {
        "rows": int(len(df_feat)),
        "cols": list(df_feat.columns),
        "latest": latest_row,
        "sources": sources_used
    }

    # Log final summary
    sources_summary = []
    if sources_used["ohlcv"]:
        sources_summary.append("OHLCV")
    if sources_used["onchain"]:
        sources_summary.append("On-chain")
    if sources_used["sentiment"]:
        sources_summary.append("Sentiment")
    if sources_used["macro"]:
        sources_summary.append("Macro")
    
    logger.info(
        f"✅ Complete feature table built for {symbol}: "
        f"{len(df_feat)} records × {len(df_feat.columns)} features "
        f"(Sources: {', '.join(sources_summary)})"
    )

    _FEATURE_TABLE_CACHE[cache_key] = {
        "fetched_at": datetime.now(timezone.utc),
        "df_feat": df_feat.copy(),
        "feature_table_info": dict(feature_table_info),
    }
    return df_feat, feature_table_info


# ---------- Data Export Function ----------
def export_feature_table_to_csv(
    df_feat: pd.DataFrame,
    symbol: str,
    feature_table_info: dict
) -> str | None:
    """
    Export the complete feature table to a CSV file for data verification.
    
    Args:
        df_feat: Feature DataFrame with OHLCV, indicators, on-chain, and macro data
        symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        feature_table_info: Metadata about the feature table
        
    Returns:
        Path to the exported CSV file, or None if export fails
    """
    try:
        # Create data_exports directory if it doesn't exist
        export_dir = DATA_EXPORT_FOLDER
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
            logger.info(f"📁 Created data export directory: {export_dir}")
        
        # Generate filename with timestamp
        timestamp = datetime.now(timezone.utc).strftime("%Y_%m_%d")
        sources = feature_table_info.get("sources", {})
        sources_str = "_".join([
            key.replace("-", "_") for key, enabled in sources.items() if enabled
        ]).upper()
        
        filename = f"{symbol}_{timestamp}_{sources_str}.csv"
        filepath = os.path.join(export_dir, filename)
        
        # Export to CSV with formatting
        df_feat.to_csv(filepath, index=True, index_label="Date")
        
        file_size_kb = os.path.getsize(filepath) / 1024
        
        logger.info(
            f"✅ Data export successful: {filepath} "
            f"({len(df_feat)} rows × {len(df_feat.columns)} columns, {file_size_kb:.2f} KB)"
        )
        
        return filepath
        
    except Exception as e:
        logger.error(f"❌ Failed to export feature table for {symbol}: {e}")
        return None


def export_news_items_to_csv(
    news_items: list[dict],
    symbol: str,
    timestamp: str,
    export_dir: str
) -> str | None:
    if not news_items:
        return None
    rows = []
    for item in news_items:
        published = item.get("published_date") or item.get("published")
        if isinstance(published, (datetime, pd.Timestamp)):
            published = published.isoformat()
        rows.append({
            "published": published or "",
            "title": item.get("title") or "",
            "source": item.get("source") or "",
            "url": item.get("url") or "",
            "sentiment_score": item.get("sentiment_score"),
            "sentiment_label": item.get("sentiment_label") or "",
            "summary": item.get("summary") or item.get("description") or "",
        })

    df_news = pd.DataFrame(rows)
    filename = f"{symbol}_{timestamp}_NEWS.csv"
    filepath = os.path.join(export_dir, filename)
    df_news.to_csv(filepath, index=False)
    return filepath


def export_analysis_metadata(
    analysis_context: dict,
    timestamp: str,
    export_dir: str
) -> str:
    news_items = analysis_context.get("news_items") or []
    meta = {
        "symbol": analysis_context.get("symbol"),
        "analysis_time": analysis_context.get("analysis_time"),
        "data_timestamp": analysis_context.get("data_timestamp"),
        "lookback_days": analysis_context.get("lookback_days"),
        "sentiment_provider": analysis_context.get("sentiment_provider"),
        "feature_table": analysis_context.get("feature_table"),
        "btc_feature_table": analysis_context.get("btc_feature_table"),
        "news_count": len(news_items),
        "news_summary": analysis_context.get("news_summary"),
        "risk_score": analysis_context.get("risk_score"),
    }
    filename = f"{analysis_context.get('symbol')}_{timestamp}_META.json"
    filepath = os.path.join(export_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return filepath


def build_complete_data_bundle(analysis_context: dict) -> str | None:
    """
    Build a complete data bundle with all collected data (OHLCV, on-chain, macro, sentiment).
    No time period filtering - exports ALL data collected for model training.
    
    Returns:
        Path to ZIP file containing all data exports, or None if export fails
    """
    export_dir = DATA_EXPORT_FOLDER
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
        logger.info("📁 Created data export directory: %s", export_dir)

    timestamp = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M")
    symbol = analysis_context.get("symbol") or "COIN"
    files_to_pack = []

    # 1. Feature table (OHLCV + on-chain + macro + technical indicators)
    feature_path = analysis_context.get("data_export_path")
    if feature_path and os.path.exists(feature_path):
        files_to_pack.append(feature_path)
        logger.info(f"📊 Added feature table: {os.path.basename(feature_path)}")
    else:
        logger.warning("Feature table export missing for %s", symbol)

    # 2. BTC feature table (if different from symbol)
    btc_feature_path = analysis_context.get("btc_data_export_path")
    if btc_feature_path and os.path.exists(btc_feature_path) and btc_feature_path not in files_to_pack:
        files_to_pack.append(btc_feature_path)
        logger.info(f"📊 Added BTC feature table: {os.path.basename(btc_feature_path)}")

    # 3. News items with sentiment scores (unfiltered by time)
    news_items = analysis_context.get("news_items") or []
    if news_items:
        news_path = export_news_items_to_csv(news_items, symbol, timestamp, export_dir)
        if news_path:
            files_to_pack.append(news_path)
            logger.info(f"📰 Added news export: {os.path.basename(news_path)} ({len(news_items)} items)")

    # 4. Analysis metadata (analysis parameters and results)
    meta_path = export_analysis_metadata(analysis_context, timestamp, export_dir)
    if meta_path:
        files_to_pack.append(meta_path)
        logger.info(f"📋 Added metadata: {os.path.basename(meta_path)}")

    if not files_to_pack:
        logger.warning("No data files to package for %s", symbol)
        return None

    # Create compressed ZIP archive
    bundle_name = f"{symbol}_{timestamp}_COMPLETE_DATA.zip"
    bundle_path = os.path.join(export_dir, bundle_name)
    
    try:
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in files_to_pack:
                arcname = os.path.basename(path)
                zf.write(path, arcname=arcname)
        
        bundle_size_kb = os.path.getsize(bundle_path) / 1024
        logger.info(
            f"✅ Data bundle created: {bundle_name} "
            f"({len(files_to_pack)} files, {bundle_size_kb:.2f} KB)"
        )
        return bundle_path
    except Exception as exc:
        logger.error(f"Failed to create data bundle: {exc}", exc_info=True)
        return None


def build_data_bundle(analysis_context: dict) -> str | None:
    """Legacy function - forwards to build_complete_data_bundle"""
    return build_complete_data_bundle(analysis_context)
import requests
import pandas as pd
from datetime import datetime

API_KEY = "537456fe97c8d896875c910fbe86a4882134a6bf5597202c5495d35d9d720cf5"

def get_onchain_history(
    symbol: str,
    start_date: str,
    end_date: str
):
    """
    Fetch on-chain daily data.
    Priority: CoinDesk API → CryptoCompare blockchain endpoint
    If the coin is not supported, the function will raise a controlled ValueError for upstream handling.
    """
    symbol = symbol.upper()
    # Validate symbol: only letters and numbers, short length
    if not (2 <= len(symbol) <= 10 and symbol.isalnum()):
        raise ValueError(f"Invalid symbol for on-chain data: {symbol}")

    # Try CoinDesk API first
    if coindesk_fetcher:
        try:
            df_coindesk = coindesk_fetcher.get_onchain_data(symbol, start_date, end_date)
            if df_coindesk is not None and not df_coindesk.empty:
                df_coindesk = df_coindesk.add_prefix("onchain_")
                logger.info(f"✓ On-chain data successfully retrieved from CoinDesk for {symbol}")
                return df_coindesk
        except Exception as e:
            logger.warning(f"CoinDesk on-chain fetch failed for {symbol}: {e}. Falling back to CryptoCompare...")

    # Fallback to CryptoCompare blockchain endpoint
    try:
        url = "https://min-api.cryptocompare.com/data/blockchain/histo/day"

        start_ts = int(pd.to_datetime(start_date).timestamp())
        end_ts = int(pd.to_datetime(end_date).timestamp())

        all_data = []
        to_ts = end_ts

        while True:
            params = {
                "fsym": symbol,
                "limit": 2000,
                "toTs": to_ts,
                "api_key": API_KEY
            }

            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            payload = r.json()

            # CryptoCompare returns Error when symbol not supported
            if payload.get("Response") == "Error":
                raise ValueError(f"On-chain data not supported for symbol {symbol}")

            batch = payload.get("Data", {}).get("Data", [])
            if not batch:
                break

            all_data.extend(batch)

            oldest_ts = batch[0]["time"]
            if oldest_ts <= start_ts:
                break

            to_ts = oldest_ts - 1

        if not all_data:
            raise ValueError(f"No on-chain data returned for {symbol}")

        df = pd.DataFrame(all_data)
        df["Date"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.sort_values("Date")

        df = df[
            (df["Date"] >= pd.to_datetime(start_date, utc=True)) &
            (df["Date"] <= pd.to_datetime(end_date, utc=True))
        ]

        # ----- Feature engineering -----
        if "transaction_count" in df.columns:
            df["transaction_count_all_time"] = df["transaction_count"].cumsum()

        df = df.set_index("Date").drop(columns=["time"], errors="ignore")
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.add_prefix("onchain_")

        logger.info(f"✓ On-chain data successfully retrieved from CryptoCompare for {symbol}")
        return df

    except Exception as e:
        logger.error(f"Failed to fetch on-chain data for {symbol}: {e}")
        raise

# ---------- Helper: Merge Crypto & On-chain ----------
def merge_crypto_onchain(df_crypto: pd.DataFrame, df_onchain: pd.DataFrame) -> pd.DataFrame:
    """
    Merge dữ liệu giá crypto với dữ liệu on-chain theo ngày.
    Forward-fill để tránh missing values, không tạo look-ahead bias.
    """
    merged = df_crypto.join(df_onchain, how="left")
    merged.ffill(inplace=True)
    return merged

# ---------- Recommendation Logic (weighted signals) ----------
def shorten_text(text: str, limit: int = 200) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return ""
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def format_compact_news_line(article: dict) -> str:
    title = shorten_text(article.get("title") or "", 100)
    title = html_escape(title)
    url = (article.get("url") or "").strip()
    source = html_escape(article.get("source") or "") if article.get("source") else ""
    published = article.get("published")
    sentiment = (article.get("sentiment_label") or "").lower()
    # sentiment_icon = {"positive": "✅", "neutral": "➖", "negative": "⚠️"}.get(sentiment, "•")
    summary_raw = article.get("summary") or article.get("description") or ""
    summary = html_escape(shorten_text(summary_raw, 90)) if summary_raw else ""

    if url:
        safe_url = html_escape(url, quote=True)
        link = f"<a href=\"{safe_url}\">Tại đây</a>"
    else:
        link = ""

    parts = [f"{title}:"]
    if published:
        try:
            published_dt = pd.to_datetime(published, utc=True)
            published_text = format_vietnam_time(published_dt)
            parts.append(f"[{published_text}]")
        except Exception:
            pass
    if link:
        parts.append(link)
    if source:
        parts.append(f"({source})")
    line = " ".join(part for part in parts if part)
    if summary:
        line = f"{line} — {summary}"
    return line


def technical_score(df_feat: pd.DataFrame) -> dict:
    """
    Score kỹ thuật trong [-1, +1], dựa trên:
    - Trend: Close so với EMA12/EMA26
    - Momentum: MACD Histogram
    - RSI: overbought/oversold (đảo chiều nhẹ)
    """
    last = df_feat.iloc[-1]
    close = float(last["Close"])
    ema12 = float(last["EMA_12"])
    ema26 = float(last["EMA_26"])
    rsi = float(last["RSI_14"])
    macd_hist = float(last["MACD_Histogram"])

    # Trend
    if close > ema12 > ema26:
        trend = 1.0
        trend_note = "Xu hướng: Tăng (Close > EMA12 > EMA26)"
    elif close < ema12 < ema26:
        trend = -1.0
        trend_note = "Xu hướng: Giảm (Close < EMA12 < EMA26)"
    else:
        trend = 0.0
        trend_note = "Xu hướng: Trung tính"

    # Momentum
    mom = 1.0 if macd_hist > 0 else (-1.0 if macd_hist < 0 else 0.0)

    # RSI signal (mean reversion nhẹ)
    if rsi >= 70:
        rsi_sig = -0.6
    elif rsi <= 30:
        rsi_sig = 0.6
    else:
        rsi_sig = 0.0

    score = 0.45 * trend + 0.35 * mom + 0.20 * rsi_sig
    score = float(np.clip(score, -1.0, 1.0))

    return {"score": score, "trend_note": trend_note}


def compute_risk_score(
    indicators: dict,
    technical: dict,
    news_sum: dict,
    macro_snapshot: dict | None = None
) -> dict:
    """
    Risk score 0-100 dựa trên biến động, xu hướng, tin tức và vĩ mô.
    """
    macro_snapshot = macro_snapshot or {}

    def _to_float(value, default=None):
        try:
            return float(value)
        except Exception:
            return default

    price = _to_float(indicators.get("last_close"))
    atr = _to_float(indicators.get("atr_14"))
    macd_hist = _to_float(indicators.get("macd_hist"), 0.0)
    last_return = _to_float(indicators.get("last_return"), 0.0)
    tech_score = _to_float(technical.get("score"), 0.0)

    # 1) Volatility shock (0-30)
    volatility_score = 10
    atr_pct = None
    if price and atr and price > 0:
        atr_pct = atr / price * 100.0
        if atr_pct >= 5.0:
            volatility_score = 30
        elif atr_pct >= 3.0:
            volatility_score = 20
        elif atr_pct >= 1.5:
            volatility_score = 12
        else:
            volatility_score = 6

    # 2) Trend breakdown (0-30)
    trend_score = 8
    if tech_score <= -0.5 or (macd_hist < 0 and last_return < 0):
        trend_score = 25
    elif tech_score <= -0.2:
        trend_score = 15
    elif tech_score >= 0.4:
        trend_score = 6

    # 3) Sentiment stress (0-20)
    sentiment_score = 6
    counts = (news_sum.get("counts") or {}) if news_sum else {}
    pos = int(counts.get("positive", 0))
    neg = int(counts.get("negative", 0))
    neu = int(counts.get("neutral", 0))
    total = pos + neg + neu
    neg_ratio = (neg / total) if total > 0 else 0.0
    avg_pol = _to_float(news_sum.get("avg_polarity"), 0.0) if news_sum else 0.0
    if neg_ratio >= 0.6 or avg_pol <= -0.2:
        sentiment_score = 20
    elif neg_ratio >= 0.45 or avg_pol <= -0.1:
        sentiment_score = 12
    elif total == 0:
        sentiment_score = 8

    # 4) Macro stress (0-20)
    macro_score = 6
    vix = _to_float(macro_snapshot.get("VIX"))
    dxy = _to_float(macro_snapshot.get("DXY"))
    macro_score = 0
    if vix is not None:
        if vix >= 25:
            macro_score += 15
        elif vix >= 20:
            macro_score += 10
        else:
            macro_score += 4
    if dxy is not None:
        if dxy >= 105:
            macro_score += 5
        elif dxy >= 102:
            macro_score += 3
        else:
            macro_score += 1
    if vix is None and dxy is None:
        macro_score = 6
    macro_score = min(20, macro_score)

    total_score = volatility_score + trend_score + sentiment_score + macro_score
    total_score = int(max(0, min(100, round(total_score))))

    if total_score >= 81:
        level = "Cực cao"
    elif total_score >= 61:
        level = "Cao"
    elif total_score >= 31:
        level = "Trung bình"
    else:
        level = "Thấp"

    return {
        "score": total_score,
        "level": level,
        "components": {
            "volatility": volatility_score,
            "trend": trend_score,
            "sentiment": sentiment_score,
            "macro": macro_score,
            "atr_percent": atr_pct,
        }
    }


def compute_ahp_weights(pairwise_matrix: list[list[float]]) -> np.ndarray:
    """
    Trả về vector trọng số chuẩn hóa từ ma trận so sánh cặp AHP.
    """
    matrix = np.array(pairwise_matrix, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_idx = int(np.argmax(eigenvalues.real))
    weights = np.abs(eigenvectors[:, max_idx].real)
    weights = weights / weights.sum()
    return weights


def calculate_financial_metrics(cumulative_returns: pd.Series, risk_free_rate: float = 0.0) -> tuple[float, float, float]:
    cumulative_returns = pd.Series(cumulative_returns).dropna()
    if len(cumulative_returns) < 2:
        return float("nan"), float("nan"), float("nan")

    daily_returns = (1 + cumulative_returns).pct_change().dropna()
    if len(daily_returns) < 1:
        return float("nan"), float("nan"), float("nan")

    avg_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()
    if std_daily_return == 0 or np.isnan(std_daily_return):
        sharpe_ratio = float("nan")
    else:
        sharpe_ratio = (avg_daily_return - risk_free_rate) / std_daily_return * np.sqrt(252)

    normalized_portfolio_value = 1 + cumulative_returns
    running_max = normalized_portfolio_value.expanding().max()
    drawdown = (normalized_portfolio_value - running_max) / running_max
    max_drawdown = float(drawdown.min()) if not drawdown.empty else float("nan")

    num_days = len(cumulative_returns)
    if num_days > 0 and cumulative_returns.iloc[-1] is not np.nan and cumulative_returns.iloc[-1] != -1:
        annualized_return = (1 + cumulative_returns.iloc[-1]) ** (252 / num_days) - 1
    else:
        annualized_return = float("nan")

    if np.isnan(annualized_return) or np.isnan(max_drawdown) or max_drawdown == 0:
        calmar_ratio = float("nan")
    else:
        calmar_ratio = annualized_return / abs(max_drawdown)

    return float(sharpe_ratio), float(max_drawdown), float(calmar_ratio)


def backtest_strategy(
    close_series: pd.Series,
    predictions: pd.Series,
    initial_capital: float = 10000.0,
    transaction_cost_rate: float = 0.001
) -> pd.Series:
    close_series = close_series.copy().dropna()
    predictions = predictions.reindex(close_series.index).fillna(0).astype(int)
    if close_series.empty:
        return pd.Series(dtype=float)

    cash = initial_capital
    shares = 0.0
    portfolio_values = [initial_capital]
    daily_strategy_returns = [0.0]

    prices = close_series.values
    signals = predictions.values

    for i in range(1, len(prices)):
        today_close = prices[i]
        prediction_for_today = signals[i]
        previous_portfolio_value = portfolio_values[-1]

        if prediction_for_today == 1:
            if shares == 0 and cash > 0:
                shares_to_buy = cash / (today_close * (1 + transaction_cost_rate))
                shares += shares_to_buy
                cash -= shares_to_buy * today_close * (1 + transaction_cost_rate)
        elif prediction_for_today == 0:
            if shares > 0:
                cash += shares * today_close * (1 - transaction_cost_rate)
                shares = 0.0

        current_portfolio_value = cash + shares * today_close
        if previous_portfolio_value != 0:
            daily_return = (current_portfolio_value - previous_portfolio_value) / previous_portfolio_value
        else:
            daily_return = 0.0

        portfolio_values.append(current_portfolio_value)
        daily_strategy_returns.append(daily_return)

    returns = pd.Series(daily_strategy_returns, index=close_series.index)
    cumulative_strategy_returns = (1 + returns).cumprod() - 1
    return cumulative_strategy_returns


def _score_to_prob(score: float) -> float:
    return float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))


def optimize_ahp_weights(
    close_series: pd.Series,
    ml_prob_series: pd.Series | None,
    sentiment_score_series: pd.Series | None,
    drl_score_series: pd.Series | None,
    weight_step: float = 0.1,
    min_points: int = 5,
    max_trim_rows: int = 5,
    sentiment_min_coverage: float = 0.5,
    component_min_coverage: float = 0.1
) -> tuple[dict, dict]:
    close_points = int(close_series.dropna().shape[0])

    def _filter_component(
        series: pd.Series | None,
        min_cov: float
    ) -> pd.Series | None:
        if series is None:
            return None
        points = int(series.dropna().shape[0])
        coverage = points / close_points if close_points else 0.0
        if points < min_points or coverage < min_cov:
            return None
        return series

    sentiment_series = _filter_component(sentiment_score_series, sentiment_min_coverage)
    ml_series = _filter_component(ml_prob_series, component_min_coverage)
    drl_series = _filter_component(drl_score_series, component_min_coverage)

    data = {"close": close_series}
    if ml_series is not None:
        data["ml_prob"] = ml_series
    if sentiment_series is not None:
        data["sentiment_score"] = sentiment_series
    if drl_series is not None:
        data["drl_score"] = drl_series

    def _trim_to_common_range(series_dict: dict, max_trim: int) -> dict:
        series_dict = {k: v.sort_index() for k, v in series_dict.items() if v is not None}
        if not series_dict:
            return series_dict
        min_dates = [s.index.min() for s in series_dict.values()]
        max_dates = [s.index.max() for s in series_dict.values()]
        common_start = max(min_dates)
        common_end = min(max_dates)
        trimmed = {}
        for key, series in series_dict.items():
            before = series.index < common_start
            after = series.index > common_end
            trim_count = int(before.sum() + after.sum())
            if trim_count <= max_trim:
                trimmed[key] = series.loc[~before & ~after]
            else:
                trimmed[key] = series
        return trimmed

    data = _trim_to_common_range(data, max_trim_rows)
    df = pd.DataFrame(data).dropna()
    components = [c for c in ("drl_score", "ml_prob", "sentiment_score") if c in df.columns]
    if len(df) < min_points or not components:
        weights = {c: 1.0 / len(components) for c in components} if components else {}
        return weights, {"data_points": int(len(df)), "sharpe": float("nan")}

    if len(components) == 1:
        weights = {components[0]: 1.0}
        return weights, {"data_points": int(len(df)), "sharpe": float("nan")}

    weight_grid = np.arange(weight_step, 1.0, weight_step)
    best_sharpe = -np.inf
    best_weights = None

    if len(components) == 2:
        a, b = components
        for w_a in weight_grid:
            w_b = 1.0 - w_a
            if w_b <= 0:
                continue
            combined_prob = None
            if a == "ml_prob":
                prob_a = df[a]
            else:
                prob_a = df[a].map(_score_to_prob)
            if b == "ml_prob":
                prob_b = df[b]
            else:
                prob_b = df[b].map(_score_to_prob)
            combined_prob = prob_a * w_a + prob_b * w_b
            preds = (combined_prob >= 0.5).astype(int)
            cumulative_returns = backtest_strategy(df["close"], preds)
            sharpe, _, _ = calculate_financial_metrics(cumulative_returns)
            if not np.isnan(sharpe) and sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = {a: float(w_a), b: float(w_b)}
    else:
        for w_drl in weight_grid:
            for w_ml in weight_grid:
                w_sent = 1.0 - w_drl - w_ml
                if w_sent <= 0:
                    continue
                prob_drl = df["drl_score"].map(_score_to_prob)
                prob_ml = df["ml_prob"]
                prob_sent = df["sentiment_score"].map(_score_to_prob)
                combined_prob = prob_drl * w_drl + prob_ml * w_ml + prob_sent * w_sent
                preds = (combined_prob >= 0.5).astype(int)
                cumulative_returns = backtest_strategy(df["close"], preds)
                sharpe, _, _ = calculate_financial_metrics(cumulative_returns)
                if not np.isnan(sharpe) and sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_weights = {
                        "drl_score": float(w_drl),
                        "ml_prob": float(w_ml),
                        "sentiment_score": float(w_sent)
                    }

    if not best_weights:
        best_weights = {c: 1.0 / len(components) for c in components}

    return best_weights, {"data_points": int(len(df)), "sharpe": float(best_sharpe)}


def _drl_score(info: dict | None) -> float | None:
    if not info:
        return None
    label = info.get("label", "HOLD")
    return 1.0 if label == "BUY" else (-1.0 if label == "SELL" else 0.0)


def _ml_score(info: dict | None) -> float | None:
    if not info:
        return None
    pu = float(info.get("proba_up", 0.5))
    pdn = float(info.get("proba_down", 0.5))
    return float(np.clip(pu - pdn, -1.0, 1.0))


def _sentiment_score(avg_polarity: float | None) -> float | None:
    if avg_polarity is None:
        return None
    return float(np.clip(avg_polarity, -1.0, 1.0))


def combine_scores_with_weights(
    drl_info: dict | None,
    ml_info: dict | None,
    sentiment_avg: float | None,
    weights: dict
) -> tuple[float, dict]:
    drl_score = _drl_score(drl_info)
    ml_score = _ml_score(ml_info)
    sent_score = _sentiment_score(sentiment_avg)

    total = 0.0
    weight_sum = 0.0
    detail_weights = {"drl_btc": 0.0, "xgb_direction": 0.0, "sentiment": 0.0}

    if drl_score is not None and "drl_score" in weights:
        w = float(weights["drl_score"])
        total += w * drl_score
        weight_sum += w
        detail_weights["drl_btc"] = w
    if ml_score is not None and "ml_prob" in weights:
        w = float(weights["ml_prob"])
        total += w * ml_score
        weight_sum += w
        detail_weights["xgb_direction"] = w
    if sent_score is not None and "sentiment_score" in weights:
        w = float(weights["sentiment_score"])
        total += w * sent_score
        weight_sum += w
        detail_weights["sentiment"] = w

    if weight_sum > 0:
        total = total / weight_sum

    detail = {
        "weights": detail_weights,
        "scores": {
            "drl_btc": drl_score,
            "xgb_direction": ml_score,
            "sentiment": sent_score,
        },
    }
    return float(np.clip(total, -1.0, 1.0)), detail


def combine_model_scores_with_ahp(drl_info: dict | None, ml_info: dict | None) -> tuple[float, dict]:
    """
    Dùng AHP để tính trọng số giữa DRL (thị trường BTC) và ML (coin riêng).
    Nếu thiếu một mô hình thì trọng số dồn về mô hình còn lại.
    """
    def _drl_score(info: dict | None) -> float:
        if not info:
            return 0.0
        label = info.get("label", "HOLD")
        return 1.0 if label == "BUY" else (-1.0 if label == "SELL" else 0.0)

    def _ml_score(info: dict | None) -> float:
        if not info:
            return 0.0
        pu = float(info.get("proba_up", 0.5))
        pdn = float(info.get("proba_down", 0.5))
        return float(np.clip(pu - pdn, -1.0, 1.0))

    drl_score = _drl_score(drl_info)
    ml_score = _ml_score(ml_info)

    if drl_info and ml_info:
        pairwise = [
            [1, 2],   # DRL được ưu tiên hơn vì phản ánh thị trường chung
            [0.5, 1],
        ]
        weights = compute_ahp_weights(pairwise)
    elif drl_info:
        weights = np.array([1.0, 0.0])
    elif ml_info:
        weights = np.array([0.0, 1.0])
    else:
        weights = np.array([0.0, 0.0])

    if weights.sum() > 0:
        weights = weights / weights.sum()

    combined = float(weights[0] * drl_score + weights[1] * ml_score)

    detail = {
        "weights": {
            "drl_btc": float(weights[0]),
            "xgb_direction": float(weights[1])
        },
        "scores": {
            "drl_btc": drl_score,
            "xgb_direction": ml_score
        }
    }
    return combined, detail


def final_recommendation(
    symbol: str,
    tech: dict,
    news_sum: dict,
    combined_model_score: float,
    model_weight_detail: dict,
    drl_info: dict | None,
    ml_info: dict | None
) -> dict:
    weights = (model_weight_detail or {}).get("weights", {}) if model_weight_detail else {}
    news_pol = float(news_sum.get("avg_polarity", 0.0))

    note_parts = []
    drl_w = float(weights.get("drl_btc", 0.0))
    if drl_info and drl_w > 0:
        label = drl_info.get("label", "HOLD")
        note_parts.append(f"DRL(BTC): {label} x{drl_w:.2f}")

    ml_w = float(weights.get("xgb_direction", 0.0))
    if ml_info and ml_w > 0:
        pu = float(ml_info.get("proba_up", 0.5))
        pdn = float(ml_info.get("proba_down", 0.5))
        note_parts.append(
            f"XGBoost: {ml_info.get('label','UNKNOWN')} (Up {pu:.0%}/Down {pdn:.0%}) x{ml_w:.2f}"
        )

    sent_w = float(weights.get("sentiment", 0.0))
    if sent_w > 0:
        note_parts.append(f"Sentiment: {news_pol:+.2f} x{sent_w:.2f}")

    model_note = " | ".join(note_parts) if note_parts else "Chưa phát hiện tín hiệu"
    total = float(np.clip(combined_model_score, -1.0, 1.0))
    score_percent = (total + 1.0) / 2.0 * 100.0
    confidence_percent = abs(total) * 100.0

    if total >= 0.25:
        rec = "BUY"
    elif total <= -0.25:
        rec = "SELL"
    else:
        rec = "HOLD"

    return {
        "recommendation": rec,
        "score": total,
        "score_percent": score_percent,
        "confidence": abs(total),
        "confidence_percent": confidence_percent,
        "model_note": model_note,
        "model_weights": {
            "drl_btc": drl_w,
            "xgb_direction": ml_w,
            "sentiment": sent_w,
        },
    }

def build_rag_context(analysis_context: dict, conversation_history: list[dict] | None = None) -> dict:
    indicators = analysis_context.get("indicators", {}) or {}
    tech = analysis_context.get("technical_score", {}) or {}
    news_sum = analysis_context.get("news_summary", {}) or {}
    sent = analysis_context.get("sentiment_info", {}) or {}
    rec = analysis_context.get("recommendation", {}) or {}
    ml = analysis_context.get("ml_info") or {}
    news_items = analysis_context.get("news_items") or []
    macro_snapshot = analysis_context.get("macro_snapshot") or {}
    risk_score = analysis_context.get("risk_score") if USE_RISK_SCORE else None
    history_tail = []
    if conversation_history:
        history_tail = conversation_history[-MAX_RAG_HISTORY:]

    def _to_float(x, default=None):
        try:
            return float(x)
        except Exception:
            return default

    def _summarize_news(items: list[dict], limit: int = 5) -> list[dict]:
        summarized = []
        for item in items[:limit]:
            published = item.get("published_date") or item.get("published")
            published_dt = pd.to_datetime(published, utc=True, errors="coerce")
            published_text = format_vietnam_time(published_dt) if not pd.isna(published_dt) else "N/A"
            summarized.append({
                "title": item.get("title") or "",
                "source": item.get("source") or "",
                "published": published_text,
                "sentiment_label": item.get("sentiment_label") or "",
                "sentiment_score": _to_float(item.get("sentiment_score")),
            })
        return summarized

    return {
        "symbol": analysis_context.get("symbol"),
        "price": _to_float(indicators.get("last_close")),
        "indicators": {
            "rsi_14": _to_float(indicators.get("rsi_14")),
            "macd_hist": _to_float(indicators.get("macd_hist")),
            "atr_14": _to_float(indicators.get("atr_14")),
            "last_return": _to_float(indicators.get("last_return")),
        },
        "technical": {
            "score": _to_float(tech.get("score")),
            "trend_note": tech.get("trend_note"),
        },
        "sentiment": {
            "label": sent.get("label"),
            "avg_polarity": _to_float(sent.get("avg_polarity"), 0.0),
            "news_counts": (news_sum.get("counts") or {"positive": 0, "neutral": 0, "negative": 0}),
        },
        "model_signal": rec.get("model_note"),
        "ml_signal": {
            "label": ml.get("label"),
            "proba_up": _to_float(ml.get("proba_up")),
            "proba_down": _to_float(ml.get("proba_down")),
        } if ml else None,
        "recommendation": {
            "action": rec.get("recommendation"),
            "score_percent": _to_float(rec.get("score_percent")),
            "confidence_percent": _to_float(rec.get("confidence_percent")),
        },
        "model_weights": analysis_context.get("model_weights"),
        "macro_snapshot": {k: _to_float(v) for k, v in macro_snapshot.items()},
        "news_headlines": _summarize_news(news_items, limit=5),
        "risk_score": {
            "score": _to_float(risk_score.get("score")),
            "level": risk_score.get("level"),
            "components": risk_score.get("components"),
        } if risk_score else None,
        "timestamp": analysis_context.get("timestamp"),
        "conversation_history": history_tail,
        "analysis_time": analysis_context.get("analysis_time"),
        "data_timestamp": analysis_context.get("data_timestamp"),
    }

# ---------- DRL Policy Engine (BTC only) ----------
class DRLPolicyEngine:
    """
    Load PPO cho BTC để đánh giá regime + recommendation.
    Khi file model thiếu hoặc hỏng sẽ fallback về rule-based signal.
    """

    def __init__(self, model_path: str, window_size: int = 60):
        self.window_size = window_size
        self.model = None
        self.available = False
        self.using_fallback = False
        self.model_path = model_path

        if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
            logger.warning(
                "DRL PPO model '%s' is missing or empty. Falling back to heuristic signals.",
                model_path
            )
            self.available = True
            self.using_fallback = True
            return

        try:
            self.model = PPO.load(model_path)
            self.available = True
            self.using_fallback = False
        except Exception as e:
            logger.error(f"Error loading PPO model from {model_path}: {e}")
            logger.warning("Switching DRL engine to heuristic fallback mode.")
            self.model = None
            self.available = True
            self.using_fallback = True

    def make_state_from_df(self, df_feat: pd.DataFrame) -> np.ndarray:
        """
        Lấy window cuối cùng để feed vào PPO.
        Ở đây mình giả sử bạn đã build state là:
        [flatten(features_window), cash_ratio, pos_ratio]
        nhưng trong chatbot này ta chỉ dùng để đánh giá market regime,
        nên có thể cho fixed cash_ratio=1, pos_ratio=0.
        """
        window = df_feat.tail(self.window_size)
        window = window.select_dtypes(include=[np.number])
        if window.empty:
            raise ValueError("DRL input has no numeric features after filtering")
        feats = window.values.flatten()
        cash_ratio = 1.0
        pos_ratio = 0.0
        state = np.concatenate([feats, [cash_ratio, pos_ratio]]).astype(np.float32)
        return state

    def get_policy_signal(self, df_feat: pd.DataFrame) -> dict:
        """
        Trả về:
        {
          "action_raw": int,
          "label": "BUY"/"HOLD"/"SELL",
          "explanation_short": "...",
        }
        """
        if not self.available:
            logger.warning("DRL model not available, returning default HOLD signal")
            return {
                "action_raw": 0,
                "label": "HOLD",
                "explanation_short": "Mô hình DRL chưa sẵn sàng. Khuyến nghị quan sát thị trường."
            }

        if self.model is None or self.using_fallback:
            return self._fallback_signal(df_feat)

        try:
            state = self.make_state_from_df(df_feat)
            action, _ = self.model.predict(state, deterministic=True)
            if action == 0:
                label = "HOLD"
                expl = "Thị trường đang ở trạng thái trung tính. Ưu tiên quan sát hơn là mở vị thế mới."
            elif action == 1:
                label = "BUY"
                expl = "Mô hình DRL đánh giá BTC đang trong trạng thái thuận lợi (risk-on). Có thể cân nhắc mua."
            else:
                label = "SELL"
                expl = "Mô hình DRL đánh giá rủi ro đang cao (risk-off). Nên ưu tiên giảm vị thế hoặc đứng ngoài."

            return {
                "action_raw": int(action),
                "label": label,
                "explanation_short": expl,
                "engine": "ppo"
            }
        except Exception as e:
            logger.error(f"Error getting policy signal: {e}")
            return {
                "action_raw": 0,
                "label": "HOLD",
                "explanation_short": "Lỗi khi chạy mô hình DRL. Khuyến nghị quan sát."
            }

    def _fallback_signal(self, df_feat: pd.DataFrame) -> dict:
        window = df_feat.tail(max(30, min(len(df_feat), self.window_size)))
        if window.empty:
            return {
                "action_raw": 0,
                "label": "HOLD",
                "explanation_short": "Thiếu dữ liệu cho DRL fallback. Khuyến nghị quan sát."
            }

        last = window.iloc[-1]
        close = float(last["Close"])
        ema12 = float(last["EMA_12"])
        ema26 = float(last["EMA_26"])
        rsi = float(last["RSI_14"])
        macd_hist = float(last["MACD_Histogram"])
        atr = float(last["ATR_14"])

        pct = window["Close"].pct_change().dropna()
        momentum = float(pct.mean()) if not pct.empty else 0.0
        vol_ratio = float(atr / close) if close else 0.0

        bull_score = 0.0
        bear_score = 0.0

        if close > ema12 > ema26:
            bull_score += 1.0
        elif close < ema12 < ema26:
            bear_score += 1.0

        if rsi >= 60:
            bull_score += 1.0
        elif rsi <= 40:
            bear_score += 1.0

        if macd_hist > 0:
            bull_score += 1.0
        elif macd_hist < 0:
            bear_score += 1.0

        if momentum > 0:
            bull_score += 0.5
        elif momentum < 0:
            bear_score += 0.5

        if vol_ratio > 0.04:
            bear_score += 0.25

        if bull_score - bear_score >= 1.0:
            action = 1
            label = "BUY"
            explanation = (
                "Fallback heuristic: Xu hướng và động lượng đang tích cực "
                "(price > EMA, RSI>60, MACD>0). Có thể cân nhắc mua."
            )
        elif bear_score - bull_score >= 1.0:
            action = 2
            label = "SELL"
            explanation = (
                "Fallback heuristic: Tín hiệu suy yếu (price < EMA, RSI<40 hoặc MACD<0). "
                "Ưu tiên giảm vị thế."
            )
        else:
            action = 0
            label = "HOLD"
            explanation = "Fallback heuristic: Tín hiệu trái chiều. Khuyến nghị quan sát."

        return {
            "action_raw": int(action),
            "label": label,
            "explanation_short": explanation,
            "engine": "heuristic"
        }

    def score_series(self, df_feat: pd.DataFrame, max_days: int | None = None) -> pd.Series:
        """
        Tính chuỗi điểm DRL (-1/0/1) cho nhiều ngày để phục vụ backtest.
        """
        if not self.available:
            return pd.Series(dtype=float)

        df_use = df_feat.tail(max_days) if max_days else df_feat
        if len(df_use) < self.window_size:
            return pd.Series(dtype=float)

        scores = []
        idx = []
        for i in range(self.window_size - 1, len(df_use)):
            df_slice = df_use.iloc[: i + 1]
            info = self.get_policy_signal(df_slice)
            label = info.get("label", "HOLD")
            score = 1.0 if label == "BUY" else (-1.0 if label == "SELL" else 0.0)
            scores.append(score)
            idx.append(df_use.index[i])

        return pd.Series(scores, index=idx)


# ---------- ML Model cho direction các coin khác ----------
class DirectionMLModel:
    """
    Model ML cho direction. Ở đây dùng XGBoost classifier,
    theo nhiều nghiên cứu là lựa chọn mạnh cho trend classification.
    >>>> Nếu bạn muốn đổi model, chỉ cần thay class này.
    """

    def __init__(self, model_path: str, features_path: str | None = None):
        # Lazy import xgboost to avoid OpenMP conflicts
        try:
            import xgboost as xgb
            self.xgb = xgb
            self.model = xgb.XGBClassifier()
            if os.path.exists(model_path):
                self.model.load_model(model_path)
            else:
                logger.warning("XGBoost model file not found. Using untrained stub.")
            if features_path is None:
                features_path = model_path + ".features.json"
            self.feature_cols = None
            if features_path and os.path.exists(features_path):
                try:
                    with open(features_path, "r") as f:
                        loaded_cols = json.load(f)
                    if isinstance(loaded_cols, list) and loaded_cols:
                        self.feature_cols = loaded_cols
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to load feature columns: %s", exc)
            self.available = True
        except ImportError as e:
            logger.error(f"XGBoost not available: {e}")
            self.model = None
            self.available = False
        except Exception as e:
            logger.error(f"Error loading XGBoost model: {e}")
            self.model = None
            self.available = False

    def prepare_features(self, df_feat: pd.DataFrame) -> np.ndarray:
        """
        Lấy các feature cuối cùng để dự báo direction ngày tới.
        Bạn cần đảm bảo thứ tự cột trùng với lúc train.
        """
        df_feat = df_feat.copy()
        df_feat = df_feat.ffill().bfill()
        feature_cols = self.feature_cols or [
            "Close",
            "Volume",
            "Log_return",
            "SMA_14",
            "EMA_12",
            "EMA_26",
            "MACD_Line",
            "MACD_Histogram",
            "RSI_14",
            "ATR_14",
            "Target_Return_1d",  # giữ target để đảm bảo bảng thống nhất, model có thể bỏ qua nếu không dùng
        ]
        missing = [col for col in feature_cols if col not in df_feat.columns]
        if "sentiment_score" in missing:
            df_feat["sentiment_score"] = 0.0
            missing = [col for col in feature_cols if col not in df_feat.columns]
        if missing:
            raise ValueError(f"Missing feature columns for ML model: {', '.join(missing)}")

        x_last = df_feat[feature_cols].iloc[-1:].values
        return x_last

    def prepare_feature_matrix(self, df_feat: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """
        Chuẩn hóa toàn bộ ma trận feature để dự báo xác suất cho nhiều ngày.
        """
        df_feat = df_feat.copy()
        df_feat = df_feat.ffill().bfill()
        feature_cols = self.feature_cols or [
            "Close",
            "Volume",
            "Log_return",
            "SMA_14",
            "EMA_12",
            "EMA_26",
            "MACD_Line",
            "MACD_Histogram",
            "RSI_14",
            "ATR_14",
            "Target_Return_1d",
        ]
        if "sentiment_score" in feature_cols and "sentiment_score" not in df_feat.columns:
            df_feat["sentiment_score"] = 0.0
        missing = [col for col in feature_cols if col not in df_feat.columns]
        if missing:
            raise ValueError(f"Missing feature columns for ML model: {', '.join(missing)}")
        x_matrix = df_feat[feature_cols].values
        return x_matrix, feature_cols

    def predict_proba_series(self, df_feat: pd.DataFrame) -> pd.Series:
        """
        Trả về xác suất UP cho toàn bộ chuỗi dữ liệu.
        """
        if not self.available or self.model is None:
            raise RuntimeError("XGBoost model not available for probability prediction")
        x_matrix, _ = self.prepare_feature_matrix(df_feat)
        proba = self.model.predict_proba(x_matrix)[:, 1]
        return pd.Series(proba, index=df_feat.index)

    def predict_direction(self, df_feat: pd.DataFrame) -> dict:
        """
        Trả về:
        {
          "proba_up": float,
          "proba_down": float,
          "label": "UP"/"DOWN"
        }
        """
        if not self.available or self.model is None:
            logger.warning("XGBoost model not available, returning default prediction")
            return {
                "proba_up": 0.5,
                "proba_down": 0.5,
                "label": "UNKNOWN"
            }
        
        x_last = self.prepare_features(df_feat)
        try:
            proba = self.model.predict_proba(x_last)[0]
            idx_up = np.argmax(proba)
            label = "UP" if idx_up == 1 else "DOWN"
            return {
                "proba_up": float(proba[1]),
                "proba_down": float(proba[0]),
                "label": label
            }
        except Exception as e:
            logger.error(f"Error predicting direction: {e}")
            return {
                "proba_up": 0.5,
                "proba_down": 0.5,
                "label": "UNKNOWN"
            }

# ---------- News & Sentiment (Alpha Vantage default) ----------
def _format_news_reply(
    symbol: str,
    news_items: list[dict],
    limit: int = 10,
    timeframe_label: str | None = None
) -> tuple[str, ParseMode]:
    """
    Build HTML reply with latest news including title, published time, sentiment, and link.
    Works with news articles from sentiment provider with format:
    {title, url, source, published_date, sentiment_score, description}
    """
    if not news_items:
        fallback_label = timeframe_label or "vài ngày gần đây"
        return (f"❌ Không tìm thấy tin tức mới cho coin này trong {fallback_label}.", ParseMode.HTML)

    if timeframe_label:
        header_line = f"🔥 Top tin tức mới nhất về {symbol} trong {timeframe_label}."
    else:
        header_line = f"🔥 Top tin tức mới nhất về {symbol} trong vài ngày gần đây."
    lines = [header_line]

    for idx, item in enumerate(news_items[:limit], 1):
        title = html_escape(item.get("title") or "")
        source = html_escape(item.get("source") or "") if item.get("source") else ""
        url = html_escape(item.get("url") or "", quote=True)
        link = f"<b><a href=\"{url}\">Tại đây</a></b>" if url else ""
        
        # Format sentiment
        sentiment_score = item.get("sentiment_score")
        if sentiment_score is not None:
            if sentiment_score > 0.15:
                sentiment_icon = "✅"
                sentiment_label = "Tích cực"
            elif sentiment_score < -0.15:
                sentiment_icon = "⚠️"
                sentiment_label = "Tiêu cực"
            else:
                sentiment_icon = "➖"
                sentiment_label = "Trung lập"
            sentiment_line = f"- Cảm xúc tin tức: {sentiment_label}"
        else:
            sentiment_line = ""

        # Format published date
        published_text = "N/A"
        published = item.get("published_date")
        if published:
            try:
                published_dt = pd.to_datetime(published, utc=True)
                published_text = format_vietnam_time(published_dt)
            except Exception:
                try:
                    parsed_dt = dateparser.parse(str(published))
                    published_text = format_vietnam_time(parsed_dt) if parsed_dt else "N/A"
                except Exception:
                    published_text = "N/A"

        meta_parts = [f"- Thời gian: {published_text}\n"]
        if source:
            meta_parts.append(f"- Nguồn: {source}")

        # Format description
        description = item.get("description") or ""
        snippet = html_escape(shorten_text(description)) if description else ""

        body = f"🔹 <b>{title}</b>\n"
        if snippet:
            body += f"- Mô tả: {snippet}\n"
        if sentiment_line:
            body += f"{sentiment_line}\n"
        if link:
            body += f"- Link bài: {link}"
        lines.append(body)

    return ("\n\n".join(lines), ParseMode.HTML)


def fetch_sentiment_news_for_display(symbol: str, days: int = None) -> tuple[str, ParseMode]:
    """
    Fetch sentiment news articles from the configured provider (Alpha Vantage by default)
    and format for Telegram display. Returns formatted message with title, date, sentiment, and URL.

    Args:
        symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        days: Number of days to look back (defaults to NEWS_LOOKBACK_DAYS from config)

    Returns:
        Tuple of (formatted_message, parse_mode)
    """
    if days is None:
        days = NEWS_LOOKBACK_DAYS
    
    if not sentiment_provider or not sentiment_provider.available:
        return ("⚠️ Sentiment provider not available", ParseMode.HTML)
    
    try:
        # Fetch full articles with sentiment from the configured provider
        articles = sentiment_provider.fetch_news_with_sentiment(symbol, days=days)
        
        if not articles:
            return (f"❌ Không tìm thấy tin tức mới cho {symbol} trong {days} ngày gần đây.", ParseMode.HTML)
        
        # Sort by published date (newest first)
        articles_sorted = sorted(articles, key=lambda x: x.get("published_date", datetime.min), reverse=True)
        
        # Format for display
        return _format_news_reply(symbol, articles_sorted, limit=10)
        
    except Exception as e:
        logger.warning(f"Error fetching sentiment news for {symbol}: {e}")
        return (f"⚠️ Lỗi khi tải tin tức: {str(e)}", ParseMode.HTML)


# ---------- Gemini RAG Engine (Gemini API) ----------
from google import genai
from google.genai import types as genai_types
from config import GEMINI_API_KEY

DEFAULT_EXPLANATION_TEMPLATE = "💡 LUẬN ĐIỂM ĐẦU TƯ CỦA {symbol}\n\n{answer}"


class RAGEngine:
    @staticmethod
    def _build_compact_context(ctx: dict) -> dict:
        return {
            "symbol": ctx.get("symbol"),
            "price": ctx.get("price"),
            "trend": ctx.get("technical", {}).get("trend_note"),
            "technical_score": ctx.get("technical", {}).get("score"),
            "rsi": ctx.get("indicators", {}).get("rsi_14"),
            "macd_hist": ctx.get("indicators", {}).get("macd_hist"),
            "sentiment": ctx.get("sentiment", {}).get("label"),
            "news_bias": ctx.get("sentiment", {}).get("avg_polarity"),
            "model_signal": ctx.get("model_signal"),
            "final_action": ctx.get("recommendation", {}).get("action"),
            "confidence_percent": ctx.get("recommendation", {}).get("confidence_percent"),
            "macro_snapshot": ctx.get("macro_snapshot"),
            "news_headlines": ctx.get("news_headlines"),
            "risk_score": ctx.get("risk_score"),
            "conversation_history": ctx.get("conversation_history"),
        }

    def __init__(self, model_name: str = "gemini-flash-latest"):
        if not GEMINI_API_KEY:
            raise RuntimeError("Thiếu GEMINI_API_KEY trong môi trường")

        self.client = genai.Client(api_key=GEMINI_API_KEY)
        normalized_name = model_name.split("/", 1)[-1] if model_name.startswith("models/") else model_name
        if normalized_name.startswith("gemini-1.5"):
            logger.warning("Model '%s' không khả dụng ở API hiện tại, chuyển sang 'gemini-flash-latest'", normalized_name)
            normalized_name = "gemini-flash-latest"
        self.model_name = normalized_name
        logger.info(f"✓ RAG Engine initialized with Gemini model '{normalized_name}'")

    def answer(self, user_query: str, analysis_context: dict) -> str:
        prompt = self._build_prompt(user_query, analysis_context)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=2048
                )
            )
            text = self._extract_text(response)
            if text:
                return text.strip()

            finish_reason = None
            if response and getattr(response, "candidates", None):
                finish_reason = response.candidates[0].finish_reason

            if finish_reason == 2:
                logger.warning("Gemini truncated explanation (MAX_TOKENS). Falling back to heuristic summary.")
                return self._fallback_explanation(analysis_context, note="Gemini bị giới hạn độ dài.")

            if finish_reason is not None and finish_reason != 1:
                human_reason = self._finish_reason_note(finish_reason)
                logger.warning("Gemini finish_reason=%s (%s). Using fallback explanation.", finish_reason, human_reason)
                return self._fallback_explanation(analysis_context, note=human_reason)

            logger.warning("Gemini response without text. Using fallback explanation.")
            return self._fallback_explanation(analysis_context, note="Gemini không trả về nội dung.")
        except Exception as e:
            logger.warning(f"Gemini RAG error: {e}. Using fallback explanation.")
            return self._fallback_explanation(analysis_context, note=str(e))

    @staticmethod
    def _build_prompt(user_query: str, analysis_context: dict) -> str:
        compact_ctx = RAGEngine._build_compact_context(analysis_context)
        context_json = json.dumps(compact_ctx, ensure_ascii=False)
        risk_line = (
            "\n- Nếu có risk_score, hãy giải thích mức rủi ro hiện tại và cách nó ảnh hưởng đến luận điểm đầu tư."
            if analysis_context.get("risk_score")
            else ""
        )

        return f"""
{RAG_SYSTEM_PROMPT.strip()}

Bạn là một chuyên gia phân tích thị trường crypto, đóng vai trò nhà tư vấn đầu tư chuyên nghiệp.

Dựa trên thông tin dưới đây, hãy giải thích khuyến nghị đầu tư
theo góc nhìn phân tích kỹ thuật, vĩ mô, tin tức thị trường và kết quả mô hình.

Yêu cầu:
- Không liệt kê dữ liệu dạng bảng
- Không nhắc đến từ "context" hay "dữ liệu đầu vào"
- Viết như đang tư vấn cho nhà đầu tư cá nhân
- Lập luận rõ ràng, mạch lạc; liên kết các ý bằng quan hệ nguyên nhân → hệ quả
- Độ dài khoảng 8–12 câu
- Chia thành 2–3 đoạn ngắn, mỗi đoạn 2–4 câu; có khoảng cách giữa các đoạn
- không cần câu chào luôn, vô giải thích luôn.
- Sử dụng ngôn ngữ tự nhiên, thân thiện, chuyên nghiệp.
- Diễn giải chi tiết hơn một chút, tránh liệt kê rời rạc.
- Chỉ cần trả lời chung về pha tích, ví dụ nói chung rằng phân tích kỹ thuật hiện tại đang ở pha tích lũy, không cần nói cụ thể là "pha 2" hay "pha 3". Tâm lý thị tường như thế nào, không cần nói cụ thể là "tâm lý sợ hãi" hay "tâm lý tham lam". Nói chung hãy phân bổ để câu trả lời ngắn gọn và đúng trọng tâm.
- Nếu người dùng tiếp tục hỏi nữa, hãy tập trung vào vào trọng tâm câu hỏi, không cần nhắc lại toàn bộ bối cảnh. Lấy ví dụ nếu người dùng hỏi "Còn về chỉ báo RSI thì sao?", bạn chỉ cần trả lời về RSI mà không cần nhắc lại toàn bộ bối cảnh phân tích kỹ thuật.
- Dùng những câu trả lời trước đó trong cuộc trò chuyện để bổ sung ngữ cảnh nếu cần.
- Nếu như người dùng hỏi những câu hỏi không liên quan đến phân tích kỹ thuật hay tâm lý thị trường, hãy lịch sự từ chối trả lời và nhắc người dùng tập trung vào chủ đề chính. Nếu coi hỏi ngoài nội dung liên quan thì hãy trả lời là bạn có thể làm rõ câu hỏi được không.
- Không đề cập dữ liệu on-chain.
{risk_line}
- Trình bày mạch lạc, tránh xuống dòng mỗi câu.
- Không nhắc tên cụ thể mô hình; chỉ mô tả chung theo nhóm: thị trường, dự báo mô hình, tin tức.

Thông tin:
{context_json}

Câu hỏi của người dùng:
{user_query}
"""

    @staticmethod
    def _finish_reason_note(code: int) -> str:
        """Map Gemini finish reason integer to a human-readable explanation."""
        return {
            0: "Không xác định",
            1: "STOP (thành công)",
            2: "MAX_TOKENS (bị cắt do giới hạn token)",
            3: "SAFETY (bị chặn do an toàn)",
            4: "RECITATION",
            5: "OTHER",
            6: "BLOCKLIST",
            7: "PROHIBITED_CONTENT",
            8: "SPII",
            9: "MALWARE",
        }.get(code, "UNKNOWN")

    @staticmethod
    def _extract_text(response) -> str | None:
        """Safe extraction of model text from a Gemini response."""
        if not response:
            return None

        direct_text = getattr(response, "text", None)
        if direct_text:
            return direct_text

        candidates = getattr(response, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) or []
            pieces = []
            for part in parts:
                part_text = getattr(part, "text", None)
                if part_text:
                    pieces.append(part_text)
            if pieces:
                return "\n".join(pieces)

        return None

    @staticmethod
    def _fallback_explanation(context: dict, note: str | None = None) -> str:
        def fmt(value, suffix=""):
            if value is None:
                return "N/A"
            try:
                if isinstance(value, (int, float)):
                    return f"{value:.2f}{suffix}"
                return str(value)
            except Exception:
                return str(value)

        def fmt_pct(value):
            if value is None:
                return "N/A"
            try:
                return f"{float(value) * 100:.1f}%"
            except Exception:
                return str(value)

        symbol = context.get("symbol", "tài sản")
        price = fmt(context.get("price"))

        indicators = context.get("indicators", {}) or {}
        technical = context.get("technical", {}) or {}
        sentiment = context.get("sentiment", {}) or {}
        news_counts = sentiment.get("news_counts", {"positive": 0, "neutral": 0, "negative": 0})
        recommendation = context.get("recommendation", {}) or {}
        model_signal = context.get("model_signal", "Chưa có tín hiệu mô hình")
        ml_signal = context.get("ml_signal")

        lines = [
            f"Phân tích nhanh cho {symbol}:",
            f"- Giá hiện tại: {price}",
            f"- RSI14: {fmt(indicators.get('rsi_14'))}, MACD Hist: {fmt(indicators.get('macd_hist'))}, ATR14: {fmt(indicators.get('atr_14'))}",
            f"- Điểm kỹ thuật: {fmt(technical.get('score'))} ({technical.get('trend_note') or 'Không có ghi chú'})",
            f"- Sentiment: {sentiment.get('label', 'Không xác định')} | Tin tích cực/trung lập/tiêu cực: {news_counts.get('positive',0)}/{news_counts.get('neutral',0)}/{news_counts.get('negative',0)}",
            f"- Tín hiệu mô hình: {model_signal}",
        ]

        if ml_signal:
            lines.append(
                f"- ML: {ml_signal.get('label','UNKNOWN')} (Up {fmt_pct(ml_signal.get('proba_up'))} / Down {fmt_pct(ml_signal.get('proba_down'))})"
            )

        rec_action = recommendation.get("action") or recommendation.get("recommendation") or "HOLD"
        rec_score = fmt(recommendation.get("score_percent"), "%")
        rec_conf = fmt(recommendation.get("confidence_percent"), "%")
        lines.append(f"- Khuyến nghị tổng hợp: {rec_action} (điểm {rec_score}, độ tin cậy {rec_conf})")

        lines.append("⚠️ Ghi chú: Đây là tóm tắt tự động khi AI không phản hồi.")
        if note:
            lines.append(f"Lý do kỹ thuật: {note}")

        return "\n".join(lines)


def format_explanation_reply(symbol: str, answer: str, analysis_context: dict) -> tuple[str, ParseMode | None]:
    """
    Build the final Telegram reply for explanation requests using a customizable template.
    Template placeholders available:
        {symbol}, {answer}, {recommendation}, {action}, {price},
        {score_percent}, {confidence_percent}, {sentiment_label},
        {news_positive}, {news_neutral}, {news_negative},
        {technical_score}, {technical_note}, {timestamp},
        {ml_label}, {ml_proba_up}, {ml_proba_down}, {context_json}
    """
    template = EXPLANATION_REPLY_TEMPLATE or DEFAULT_EXPLANATION_TEMPLATE
    parse_mode_key = (EXPLANATION_PARSE_MODE or "PLAIN").upper()

    recommendation = (analysis_context.get("recommendation") or {})
    indicators = (analysis_context.get("indicators") or {})
    sentiment = (analysis_context.get("sentiment_info") or {})
    news_sum = (analysis_context.get("news_summary") or {})
    news_counts = news_sum.get("counts") or {}
    tech = (analysis_context.get("technical_score") or {})
    ml_info = analysis_context.get("ml_info") or {}

    template_data = {
        "symbol": symbol,
        "answer": answer,
        "recommendation": recommendation.get("recommendation") or recommendation.get("action") or "",
        "action": recommendation.get("recommendation") or recommendation.get("action") or "",
        "price": _format_number(indicators.get("last_close")),
        "score_percent": _format_percent(recommendation.get("score_percent")),
        "confidence_percent": _format_percent(recommendation.get("confidence_percent")),
        "sentiment_label": sentiment.get("label", ""),
        "news_positive": news_counts.get("positive", 0),
        "news_neutral": news_counts.get("neutral", 0),
        "news_negative": news_counts.get("negative", 0),
        "technical_score": _format_number(tech.get("score")),
        "technical_note": tech.get("trend_note", ""),
        "timestamp": analysis_context.get("timestamp", ""),
        "ml_label": ml_info.get("label", ""),
        "ml_proba_up": _format_percent(ml_info.get("proba_up")),
        "ml_proba_down": _format_percent(ml_info.get("proba_down")),
        "context_json": json.dumps(analysis_context, ensure_ascii=False, indent=2),
    }

    escaped_data = {k: _escape_for_parse_mode(v, parse_mode_key) for k, v in template_data.items()}

    try:
        text = template.format(**escaped_data)
    except KeyError as err:
        logger.warning(
            "Explanation template missing placeholder '%s'. Falling back to default template.",
            err
        )
        text = DEFAULT_EXPLANATION_TEMPLATE.format(**escaped_data)

    parse_mode = _map_parse_mode(parse_mode_key)
    return text, parse_mode


def _escape_for_parse_mode(value, mode: str) -> str:
    if value is None:
        text = ""
    else:
        text = str(value)

    if mode == "HTML":
        return html_escape(text)
    if mode == "MARKDOWNV2":
        return escape_markdown(text, version=2)
    if mode == "MARKDOWN":
        return escape_markdown(text, version=1)
    return text


def _map_parse_mode(mode: str) -> ParseMode | None:
    if mode == "HTML":
        return ParseMode.HTML
    if mode == "MARKDOWNV2":
        return ParseMode.MARKDOWN_V2
    if mode == "MARKDOWN":
        return ParseMode.MARKDOWN
    return None


def _format_number(value) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def _format_percent(value) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.1f}%"
    except Exception:
        return str(value)


# ---------- Chart HTML generator ----------
def generate_candlestick_html(df: pd.DataFrame, symbol: str) -> str:
    os.makedirs(CHART_FOLDER, exist_ok=True)
    filename = os.path.join(CHART_FOLDER, f"{symbol}_chart.html")

    if df.empty:
        raise ValueError("DataFrame rỗng, không thể vẽ chart.")

    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).set_index("Date")
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[~df.index.isna()]

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Thiếu cột để vẽ chart: {', '.join(sorted(missing_cols))}")

    # Hiển thị 6 tháng gần nhất (fallback ~180 ngày nếu thiếu dữ liệu)
    latest_ts = df.index.max()
    if isinstance(latest_ts, pd.Timestamp):
        cutoff = latest_ts - pd.DateOffset(months=12)
        recent = df[df.index >= cutoff]
        df = recent if not recent.empty else df.tail(365)
    else:
        df = df.tail(365)

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.52, 0.16, 0.12, 0.12, 0.08]
    )
    trace_groups = {"candlestick": [], "indicators": []}

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name=symbol
    ), row=1, col=1)
    trace_groups["candlestick"].append(len(fig.data) - 1)

    volume_colors = np.where(df["Close"] >= df["Open"], "#2ecc71", "#e74c3c")
    fig.add_trace(go.Bar(
        x=df.index,
        y=df["Volume"],
        name="Volume",
        marker_color=volume_colors,
        opacity=0.4
    ), row=2, col=1)

    if {"BB_Upper_20", "BB_Mid_20", "BB_Lower_20"}.issubset(df.columns):
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["BB_Upper_20"],
            name="BB",
            mode="lines",
            line=dict(width=1, color="#9b59b6"),
            legendgroup="BB",
            showlegend=True
        ), row=1, col=1)
        trace_groups["indicators"].append(len(fig.data) - 1)
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["BB_Mid_20"],
            name="BB Mid (20)",
            mode="lines",
            line=dict(width=1, color="#8e44ad", dash="dot"),
            legendgroup="BB",
            showlegend=False
        ), row=1, col=1)
        trace_groups["indicators"].append(len(fig.data) - 1)
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["BB_Lower_20"],
            name="BB Lower (20)",
            mode="lines",
            line=dict(width=1, color="#9b59b6"),
            legendgroup="BB",
            showlegend=False
        ), row=1, col=1)
        trace_groups["indicators"].append(len(fig.data) - 1)

    if "RSI_14" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["RSI_14"],
            name="RSI 14",
            mode="lines",
            line=dict(width=1.5, color="#f39c12")
        ), row=3, col=1)
        trace_groups["indicators"].append(len(fig.data) - 1)

    if "MACD_Line_6_20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["MACD_Line_6_20"],
            name="MACD",
            mode="lines",
            line=dict(width=1.2, color="#2980b9"),
            legendgroup="MACD",
            showlegend=True
        ), row=4, col=1)
        trace_groups["indicators"].append(len(fig.data) - 1)
    if "MACD_Signal_6_20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["MACD_Signal_6_20"],
            name="MACD Signal",
            mode="lines",
            line=dict(width=1.2, color="#c0392b"),
            legendgroup="MACD",
            showlegend=False
        ), row=4, col=1)
        trace_groups["indicators"].append(len(fig.data) - 1)
    if "MACD_Histogram" in df.columns:
        fig.add_trace(go.Bar(
            x=df.index,
            y=df["MACD_Histogram"],
            name="MACD Histogram",
            marker_color="#95a5a6",
            opacity=0.6,
            legendgroup="MACD",
            showlegend=False
        ), row=4, col=1)
        trace_groups["indicators"].append(len(fig.data) - 1)

    if "ATR_14" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["ATR_14"],
            name="ATR 14",
            mode="lines",
            line=dict(width=1.2, color="#16a085")
        ), row=5, col=1)
        trace_groups["indicators"].append(len(fig.data) - 1)

    fig.update_layout(
        title=f"Biểu đồ giá {symbol}",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top"
        ),
        autosize=True,
        margin=dict(l=40, r=40, t=60, b=50),
        height=860
    )

    fig.update_xaxes(title_text="", row=1, col=1, rangeslider=dict(visible=False))
    fig.update_xaxes(title_text="", row=2, col=1)
    fig.update_xaxes(title_text="", row=3, col=1)
    fig.update_xaxes(title_text="", row=4, col=1)
    fig.update_xaxes(title_text="Ngày", row=5, col=1)

    fig.update_yaxes(title_text=f"Giá {symbol}", row=1, col=1, showgrid=True)
    fig.update_yaxes(title_text="Volume", row=2, col=1, showgrid=False)
    fig.update_yaxes(title_text="RSI", row=3, col=1, showgrid=True)
    fig.update_yaxes(title_text="MACD", row=4, col=1, showgrid=True)
    fig.update_yaxes(title_text="ATR", row=5, col=1, showgrid=True)

    fig.write_html(
        filename,
        include_plotlyjs="inline",
        full_html=True,
        config={"responsive": True, "displayModeBar": False}
    )
    return filename


# ---------- Khởi tạo các service ----------
# Data provider always works (uses free CoinGecko/Binance APIs, CMC is optional)
try:
    data_provider = CryptoDataProvider(CMC_API_KEY)
    logger.info("✓ Data provider initialized (free APIs available, CoinMarketCap optional)")
except Exception as e:
    logger.error(f"Failed to initialize data provider: {e}")
    data_provider = None

# Initialize CoinDesk API fetcher for on-chain and OHLCV data
coindesk_fetcher = None

# Initialize sentiment data provider (toggle via SENTIMENT_PROVIDER)
sentiment_provider = None
try:
    sentiment_provider = create_sentiment_provider(SENTIMENT_PROVIDER)
    if sentiment_provider and getattr(sentiment_provider, "available", False):
        logger.info("✓ Sentiment data provider initialized (%s)", sentiment_provider.provider_name)
    elif sentiment_provider is None:
        logger.info("Sentiment provider disabled (SENTIMENT_PROVIDER=%s)", SENTIMENT_PROVIDER)
    else:
        init_error = getattr(sentiment_provider, "init_error", None)
        if init_error:
            logger.warning("Sentiment data provider unavailable: %s", init_error)
        else:
            logger.warning("Sentiment data provider unavailable (missing deps or init failed)")
except Exception as e:
    logger.warning("Failed to initialize SentimentDataProvider: %s", e)
    sentiment_provider = None

try:
    coindesk_fetcher = CoinDeskDataFetcher(COINDESK_API_KEY)
    logger.info("✓ CoinDesk API fetcher initialized for on-chain and OHLCV data retrieval")
except Exception as e:
    logger.warning(f"CoinDesk API initialization warning: {e}")
    coindesk_fetcher = None

# Initialize macro data provider (FRED API)
macro_provider = None
if FRED_API_KEY:
    try:
        macro_provider = MacroDataProvider(FRED_API_KEY)
        logger.info("✓ Macro data provider initialized (FRED API)")
    except Exception as e:
        logger.warning(f"Failed to initialize MacroDataProvider: {e}")
        macro_provider = None
else:
    logger.info("FRED_API_KEY not configured, macro data provider skipped")
    macro_provider = None

# Initialize DRL engine with error handling
drl_engine = None
if USE_DRL:
    model_path = select_drl_model_path()
    if os.path.exists(model_path):
        try:
            drl_engine = DRLPolicyEngine(model_path, window_size=WINDOW_SIZE)
            if not drl_engine.available:
                drl_engine = None
                logger.warning("DRL engine initialized but model not available")
        except Exception as e:
            logger.error(f"Failed to initialize DRL engine: {e}")
            drl_engine = None
    else:
        logger.warning("DRL model path %s does not exist", model_path)

# Initialize direction model with error handling
direction_model = None
if os.path.exists(XGB_DIRECTION_MODEL_PATH):
    try:
        direction_model = DirectionMLModel(XGB_DIRECTION_MODEL_PATH)
        if not direction_model.available:
            direction_model = None
            logger.warning("Direction model initialized but not available")
    except Exception as e:
        logger.error(f"Failed to initialize direction model: {e}")
        direction_model = None

# Initialize RAG engine
rag_engine = None
try:
    rag_engine = RAGEngine()
except Exception as e:
    logger.warning(f"Gemini RAG disabled: {e}")
    rag_engine = None


# ---------- Telegram Handlers ----------
def get_commands_menu():
    """Returns formatted commands menu"""
    return (
        "📋 **MENU LỆNH**\n\n"
        "/start - Bắt đầu sử dụng bot\n"
        "/help - Hướng dẫn sử dụng\n"
        "/analyze <coin> - Phân tích coin (VD: /analyze BTC)\n"
        "/train - Huấn luyện lại mô hình DRL (BTC)\n"
        "/about - Giới thiệu về bot\n"
        "/renew - Làm mới chat (xóa lịch sử phân tích)\n\n"
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Send monitoring log
    user_id = update.effective_user.id if update.effective_user else "unknown"
    username = update.effective_user.username if update.effective_user else "unknown"
    monitor_msg = f"🚀 <b>Bot Started:</b>\n- User: @{username} (ID: {user_id})"
    await send_to_monitor_bot(monitor_msg)
    
    text = (
        "👋 **Chào mừng đến với Crypto Analysis Bot!**\n\n"
        "Tôi là AI Chatbot chuyên phân tích giá cryptocurrency.\n\n"
        f"{get_commands_menu()}"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for /help command"""
    # Send monitoring log
    user_id = update.effective_user.id if update.effective_user else "unknown"
    username = update.effective_user.username if update.effective_user else "unknown"
    monitor_msg = f"❓ <b>Help Requested:</b>\n- User: @{username} (ID: {user_id})"
    await send_to_monitor_bot(monitor_msg)
    
    text = (
        f"📖 <b>HƯỚNG DẪN SỬ DỤNG</b>\n\n"
        f"**1. Phân tích coin:**\n"
        f"• Dùng lệnh: `/analyze BTC`\n"
        "**2. Hỏi về phân tích:**\n"
        "Sau khi có kết quả phân tích, bạn có thể hỏi:\n"
        "• 'Lý do khuyến nghị này?'\n"
        "• 'Tại sao nên mua/bán?'\n"
        "• 'Phân tích chi tiết hơn'\n\n"
        "**3. Các tính năng:**\n"
        "• 📈 Xem biểu đồ và chỉ báo kỹ thuật\n"
        "• 📰 Đọc tin tức mới nhất\n"
        "• 💡 Giải thích chi tiết khuyến nghị\n\n"
        "**4. Làm mới chat:**\n"
        "Dùng `/renew` để xóa lịch sử và bắt đầu lại\n\n"
        f"{get_commands_menu()}"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for /about command"""
    # Send monitoring log
    user_id = update.effective_user.id if update.effective_user else "unknown"
    username = update.effective_user.username if update.effective_user else "unknown"
    monitor_msg = f"ℹ️ <b>About Viewed:</b>\n- User: @{username} (ID: {user_id})"
    await send_to_monitor_bot(monitor_msg)
    
    text = (
        "🤖 **GIỚI THIỆU VỀ BOT**\n\n"
        "**Crypto Analysis Bot** là Chatbot AI dùng để phân tích và đưa ra kết quả khuyến nghị đầu tư của các dòng tiền điện tử:\n\n"

        "**📊 Hệ thống sử dụng 3 mô hình chính:**\n"
        "- Mô hình dự báo bằng máy học (XGBoost).\n"
        "• Mô hình học tăng cường sâu tượng trưng cho xu hướng thị trường (lấy đồng tiền Bitcoin làm chuẩn).\n"
        "• Mô hình tổng hợp điểm cảm xúc từ các tin tức mới nhất.\n\n"
    
        "** 🟢 Khuyến nghị cuối cùng được đưa ra bằng cách tổng hợp kết quả của 3 mô hình trên.**\n\n"
        
        "**💡 Đưa ra chi tiết luận điểm đầu tư bằng AI.**\n\n"
        
        "**⚠️ Lưu ý:**\n"
        "- Đây là Chatbot được xây dựng nhằm mục đích nghiên cứu học thuật."
        "- Tất cả khuyến nghị chỉ mang tính tham khảo. Quyết định đầu tư cuối cùng thuộc về bạn.\n\n"
        
        "**📞 Hỗ trợ:**\n"
        "Dùng `/help` để xem hướng dẫn chi tiết."
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def renew_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for /renew command - clears user data"""
    # Send monitoring log
    user_id = update.effective_user.id if update.effective_user else "unknown"
    username = update.effective_user.username if update.effective_user else "unknown"
    monitor_msg = f"🔄 <b>Chat Renewed:</b>\n- User: @{username} (ID: {user_id})"
    await send_to_monitor_bot(monitor_msg)
    
    context.user_data.clear()
    context.user_data["history_last_reset"] = datetime.now(timezone.utc).isoformat()
    text = (
        "🔄 **Chat đã được làm mới!**\n\n"
        "Lịch sử phân tích đã được xóa.\n"
        "Bạn có thể bắt đầu phân tích mới.\n\n"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def train_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for /train command - manually trigger DRL model retraining"""
    # Send monitoring log
    user_id = update.effective_user.id if update.effective_user else "unknown"
    username = update.effective_user.username if update.effective_user else "unknown"
    monitor_msg = f"🤖 <b>DRL Training Started:</b>\n- User: @{username} (ID: {user_id})"
    await send_to_monitor_bot(monitor_msg)
    
    await update.message.reply_text(
        "🔄 **Bắt đầu huấn luyện lại mô hình DRL...**\n\n"
        "Quá trình này có thể mất vài phút.\n"
        "Vui lòng chờ...",
        parse_mode="Markdown"
    )
    
    logger.info("Manual DRL training triggered by user %s", update.effective_user.id)
    
    try:
        logger.info("📊 Rebuilding BTC feature table...")
        df_feat, feature_info = build_feature_table("BTC", DATA_LOOKBACK_DAYS)
        csv_path = "btc_features.csv"
        df_feat.to_csv(csv_path, index=True, index_label="Date")
        logger.info("✓ BTC feature CSV exported to %s (%d rows)", csv_path, len(df_feat))
        
        logger.info("🤖 Executing DRL training script...")
        success = run_drl_script()
        
        if success:
            # Copy produced model to dated file
            dated_path = get_dated_model_path()
            try:
                shutil.copy2(PPO_BTC_MODEL_PATH, dated_path)
                logger.info("✓ Model saved to %s", dated_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to copy to dated file: %s", exc)
            
            reply_text = (
                "✅ **Huấn luyện mô hình hoàn tất!**\n\n"
                f"📊 BTC feature table: {len(df_feat)} dòng dữ liệu\n"
                f"🤖 Mô hình DRL đã được cập nhật\n"
                f"💾 Lưu tại: {PPO_BTC_MODEL_PATH}\n\n"
                "Mô hình mới sẽ được sử dụng cho các phân tích tiếp theo."
            )
        else:
            reply_text = (
                "❌ **Lỗi khi huấn luyện mô hình!**\n\n"
                "Vui lòng kiểm tra logs để biết chi tiết.\n"
                "Mô hình cũ sẽ tiếp tục được sử dụng."
            )
        
        await update.message.reply_text(reply_text, parse_mode="Markdown")
        
    except Exception as exc:  # noqa: BLE001
        logger.error("Error in manual DRL training: %s", exc, exc_info=True)
        error_text = (
            "❌ **Lỗi khi huấn luyện!**\n\n"
            f"Chi tiết: `{str(exc)[:100]}`\n\n"
            "Vui lòng kiểm tra logs hoặc thử lại sau."
        )
        await update.message.reply_text(error_text, parse_mode="Markdown")


async def show_commands_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Shows commands menu when user types '/'"""
    await update.message.reply_text(get_commands_menu(), parse_mode="Markdown")


def parse_symbol_from_text(text: str) -> str | None:
    text = text.strip().upper()
    # Rất đơn giản: lấy từ cuối cùng nếu là chữ + số
    tokens = text.replace("/", " ").split()
    for t in reversed(tokens):
        if t.isalpha() and 2 <= len(t) <= 10:
            return t
    return None


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler cho /analyze <symbol>
    """
    message = update.message
    await refresh_history_if_needed(message, context)
    args = context.args
    if not args:
        await message.reply_text("Hãy nhập: /analyze BTC hoặc /analyze ETH,…")
        return

    symbol = normalize_symbol(args[0])
    if message:
        log_user_message(context.user_data, message.text, symbol)
    
    # Send monitoring log
    user_id = message.from_user.id if message else "unknown"
    username = message.from_user.username if message and message.from_user else "unknown"
    monitor_msg = f"📊 \n<b>Analysis Request:</b>\n- User: @{username} (ID: {user_id})\n- Symbol: <b>{symbol}</b>"
    await send_to_monitor_bot(monitor_msg)
    
    await run_full_analysis(update, context, symbol)


async def data_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Export all collected data from the analysis (OHLCV, onchain, macro, sentiment).
    Downloads complete unfiltered dataset without time period restrictions.
    """
    message = update.message
    await refresh_history_if_needed(message, context)

    # Send monitoring log
    user_id = message.from_user.id if message else "unknown"
    username = message.from_user.username if message and message.from_user else "unknown"
    
    symbol = context.user_data.get("last_symbol")
    if symbol:
        monitor_msg = f"💾 <b>Data Export Requested:</b>\n- User: @{username} (ID: {user_id})\n- Symbol: <b>{symbol}</b>"
        await send_to_monitor_bot(monitor_msg)
    if not symbol:
        await message.reply_text(
            "❌ Chưa có dữ liệu. Hãy chạy /analyze <coin> trước để thu thập dữ liệu.\n"
            "Ví dụ: /analyze BTC"
        )
        return

    analysis_key = f"last_analysis_{symbol}"
    analysis_ctx = context.user_data.get(analysis_key)
    if not analysis_ctx:
        await message.reply_text(
            f"❌ Dữ liệu hết hạn. Hãy chạy lại: /analyze {symbol}"
        )
        return

    # Build or retrieve complete data bundle
    bundle_path = analysis_ctx.get("data_bundle_path")
    if not bundle_path or not os.path.exists(bundle_path):
        await message.reply_text("⏳ Đang chuẩn bị dữ liệu để tải xuống...")
        bundle_path = build_complete_data_bundle(analysis_ctx)
        if not bundle_path:
            await message.reply_text(
                "❌ Không tạo được file dữ liệu. Vui lòng chạy /analyze lại."
            )
            return
        analysis_ctx["data_bundle_path"] = bundle_path
        analysis_ctx["data_bundle_created_at"] = datetime.now(timezone.utc).isoformat()
        context.user_data[analysis_key] = analysis_ctx

    try:
        bundle_filename = os.path.basename(bundle_path)
        file_size_mb = os.path.getsize(bundle_path) / (1024 * 1024)
        logger.info(f"Sending data bundle: {bundle_filename} ({file_size_mb:.2f} MB)")
        
        with open(bundle_path, "rb") as f:
            await message.reply_document(
                document=InputFile(f, filename=bundle_filename),
                caption=(
                    f"📦 Dữ liệu của đồng {symbol}:\n"
                    f"- Bao gồm: OHLCV, On-chain, Macro, Sentiment.\n"
                    f"- Kích thước: {file_size_mb:.2f} MB."
                    f"Nhấn đề tải xuống." 
                )
            )
        logger.info(f"Data bundle sent successfully to user")
    except Exception as exc:
        logger.error(f"Failed to send data bundle: {exc}", exc_info=True)
        await message.reply_text(
            "❌ Gửi file dữ liệu thất bại. Vui lòng thử lại sau vài giây."
        )


async def analyze_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Sau khi chạy /analyze, mọi tin nhắn tiếp theo đều được dùng để hỏi đáp dựa trên dữ liệu đã phân tích.
    """
    message = update.message
    await refresh_history_if_needed(message, context)
    raw_text = message.text or ""
    last_symbol = context.user_data.get("last_symbol")
    log_user_message(context.user_data, raw_text, last_symbol)
    
    # Send monitoring log for follow-up questions
    user_id = message.from_user.id if message else "unknown"
    username = message.from_user.username if message and message.from_user else "unknown"
    if last_symbol:
        monitor_msg = f"💬 <b>Follow-up Question:</b>\n- User: @{username} (ID: {user_id})\n- Symbol: <b>{last_symbol}</b>\n📝 Query: <code>{raw_text[:100]}</code>"
        await send_to_monitor_bot(monitor_msg)

    if not last_symbol:
        response_text = (
            "Tôi chưa có dữ liệu để trả lời.\n"
            "Vui lòng bắt đầu bằng lệnh /analyze <mã_coin> (ví dụ: /analyze BTC)."
        )
        await message.reply_text(response_text)
        log_bot_message(context.user_data, response_text)
        return

    if _is_unclear_query(raw_text):
        await message.reply_text(UNCLEAR_REPLY)
        log_bot_message(context.user_data, UNCLEAR_REPLY, last_symbol)
        return

    await handle_rag_question(update, context, raw_text, last_symbol)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled exception: %s", context.error)
    message = getattr(update, "effective_message", None)
    if not message:
        return
    error_text = f"⚠️ Có lỗi khi chạy phân tích: {context.error}"
    try:
        await message.reply_text(error_text)
    except Exception:
        pass


async def handle_news_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Gửi danh sách tin tức đã fetch từ provider cấu hình với link 'Tại đây'.
    """
    query = update.callback_query
    if not query:
        return
    await query.answer()

    data = query.data or ""
    _, _, symbol = data.partition(":")
    symbol = symbol.upper() if symbol else ""
    
    # Send monitoring log
    user_id = query.from_user.id if query.from_user else "unknown"
    username = query.from_user.username if query.from_user else "unknown"
    monitor_msg = f"📰 <b>Latest News Clicked:</b>\n👤 User: @{username} (ID: {user_id})\n💱 Symbol: <b>{symbol}</b>"
    await send_to_monitor_bot(monitor_msg)

    analysis_key = f"last_analysis_{symbol}"
    analysis_ctx = context.user_data.get(analysis_key) or {}
    news_items_display = analysis_ctx.get("news_items_display")
    if news_items_display is None:
        news_items = analysis_ctx.get("news_items") or []
    else:
        news_items = news_items_display

    def _published_dt(item: dict) -> datetime:
        published = item.get("published_date") or item.get("published")
        published_dt = pd.to_datetime(published, utc=True, errors="coerce")
        return published_dt if not pd.isna(published_dt) else datetime.min.replace(tzinfo=timezone.utc)

    filtered_items = sorted(news_items, key=_published_dt, reverse=True)
    timeframe_label = "gần đây"

    text, parse_mode = _format_news_reply(
        symbol or "COIN",
        filtered_items,
        limit=10,
        timeframe_label=timeframe_label
    )
    await query.message.reply_text(text, parse_mode=parse_mode, disable_web_page_preview=True)


async def handle_thesis_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Gửi luận điểm đầu tư đã tạo bởi RAG.
    """
    query = update.callback_query
    if not query:
        return
    await query.answer()

    data = query.data or ""
    _, _, symbol = data.partition(":")
    symbol = symbol.upper() if symbol else ""
    
    # Send monitoring log
    user_id = query.from_user.id if query.from_user else "unknown"
    username = query.from_user.username if query.from_user else "unknown"
    monitor_msg = f"💡 <b>Investment Thesis Viewed:</b>\n👤 User: @{username} (ID: {user_id})\n💱 Symbol: <b>{symbol}</b>"
    await send_to_monitor_bot(monitor_msg)

    analysis_key = f"last_analysis_{symbol}"
    analysis_ctx = context.user_data.get(analysis_key) or {}
    thesis_text = analysis_ctx.get("investment_thesis") or ""
    if not thesis_text:
        await query.message.reply_text("❌ Chưa có luận điểm đầu tư cho coin này. Vui lòng chạy lại /analyze.")
        return

    safe_symbol = html_escape(symbol or "COIN")
    lines = [line.strip() for line in thesis_text.splitlines() if line.strip()]
    formatted_text = "\n\n".join(lines) if lines else thesis_text
    safe_text = html_escape(formatted_text)
    title = f"<b>💡 Luận điểm đầu tư của - {safe_symbol}</b>\n\n"
    note = "⚠️ Lưu ý: Đây chỉ là khuyến nghị tham khảo. Quyết định cuối cùng thuộc về nhà đầu tư."
    await query.message.reply_text(
        title + safe_text + "\n\n" + note,
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True
    )

async def handle_chart_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Gửi file HTML chart giá đã render.
    """
    query = update.callback_query
    if not query:
        return
    await query.answer()

    data = query.data or ""
    _, _, symbol = data.partition(":")
    symbol = symbol.upper() if symbol else ""
    
    # Send monitoring log
    user_id = query.from_user.id if query.from_user else "unknown"
    username = query.from_user.username if query.from_user else "unknown"
    monitor_msg = f"📈 <b>Chart Downloaded:</b>\n👤 User: @{username} (ID: {user_id})\n💱 Symbol: <b>{symbol}</b>"
    await send_to_monitor_bot(monitor_msg)

    analysis_key = f"last_analysis_{symbol}"
    analysis_ctx = context.user_data.get(analysis_key) or {}
    chart_path = analysis_ctx.get("chart_html_path")

    if not chart_path or not os.path.exists(chart_path):
        await query.message.reply_text("❌ Không tìm thấy file chart. Vui lòng chạy lại /analyze.")
        return

    try:
        with open(chart_path, "rb") as f:
            await query.message.reply_document(
                document=InputFile(f, filename=os.path.basename(chart_path)),
                caption=(
                    f"📦 Biểu đồ giá {symbol} trong 1 năm gần nhất.\n"
                    f"Kích thước: {os.path.getsize(chart_path) / (1024 * 1024):.2f} MB.\n"
                    f"Nhấn để tải xuống."
                )
            )
    except Exception as exc:  # noqa: BLE001
        logger.error("Gửi chart thất bại: %s", exc)
        await query.message.reply_text("❌ Gửi chart thất bại. Vui lòng thử lại.")


async def handle_rag_question(update: Update, context: ContextTypes.DEFAULT_TYPE, user_question: str, symbol: str):
    """
    Xử lý câu hỏi về phân tích sử dụng RAG engine
    """
    message = update.message
    conversation_history = get_recent_history(context.user_data)
    
    # Lấy analysis context từ lần phân tích gần nhất
    analysis_key = f"last_analysis_{symbol}"
    analysis_context = context.user_data.get(analysis_key)
    
    if not analysis_context:
        response_text = (
            f"Tôi chưa có dữ liệu phân tích cho {symbol}.\n"
            f"Hãy chạy phân tích trước: /analyze {symbol}"
        )
        await message.reply_text(response_text)
        log_bot_message(context.user_data, response_text, symbol)
        return
    
    if not rag_engine:
        response_text = (
            "❌ RAG engine chưa sẵn sàng.\n\n"
            "Vui lòng kiểm tra:\n"
            "• Đã đăng nhập: gcloud auth application-default login\n"
            "• Đã cài: pip install google-generativeai\n\n"
            "Bot vẫn hoạt động bình thường, chỉ thiếu phần giải thích bằng AI."
        )
        await message.reply_text(response_text)
        log_bot_message(context.user_data, response_text, symbol)
        return
    
    # Gửi message "đang suy nghĩ"
    thinking_msg = await message.reply_text("⏳ Đang phân tích và tạo câu trả lời chi tiết...")
    
    try:
        # Gọi RAG engine
        rag_context = build_rag_context(analysis_context, conversation_history)
        answer = rag_engine.answer(user_question, rag_context)
        
        # Xóa message "đang suy nghĩ" và gửi câu trả lời
        await thinking_msg.delete()
        reply_text = f"{answer}"
        await message.reply_text(reply_text, parse_mode="Markdown")
        log_bot_message(context.user_data, reply_text, symbol)
    except Exception as e:
        logger.error(f"RAG question error: {e}")
        await thinking_msg.delete()
        response_text = (
            f"❌ Có lỗi khi tạo câu trả lời.\n"
            f"Vui lòng thử lại sau hoặc chạy lại phân tích: /analyze {symbol}"
        )
        await message.reply_text(response_text)
        log_bot_message(context.user_data, response_text, symbol)


async def run_full_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE, symbol: str):
    message = update.message

    loading_msg = await message.reply_text(f"⏳ Đang phân tích {symbol}... (Quá trình này có thể mất vài phút).")

    lookback_days = get_analysis_lookback_days()
    btc_export_path = None

    try:
        df_feat, feature_table_info = build_feature_table(symbol, lookback_days)
        if df_feat.empty:
            await message.reply_text(
                f"❌ Không tìm thấy dữ liệu cho {symbol}.\n"
                "Vui lòng kiểm tra lại mã coin (ví dụ: BTC, ETH, SOL)."
            )
            return
        
        # Export feature table to CSV for data verification
        export_path = export_feature_table_to_csv(df_feat, symbol, feature_table_info)
        if export_path:
            logger.info(f"📊 Data verification file: {export_path}")
        
    except ValueError as e:
        error_msg = str(e)
        logger.error(f"Error fetching data for {symbol}: {error_msg}")
        await message.reply_text(
            f"❌ Không thể lấy dữ liệu cho {symbol}.\n\n"
            "Vui lòng thử:\n"
            "- Kiểm tra lại mã coin (ví dụ: BTC, ETH, SOL)\n"
            "- Thử lại sau vài giây"
        )
        return
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching data for {symbol}: {e}")
        await message.reply_text(
            f"❌ Lỗi kết nối khi lấy dữ liệu cho {symbol}.\n"
            "Vui lòng thử lại sau vài giây."
        )
        return
    except Exception as e:
        logger.error(f"Unexpected error fetching data for {symbol}: {e}", exc_info=True)
        await message.reply_text(
            f"❌ Lỗi không xác định khi lấy dữ liệu cho {symbol}.\n"
            "Vui lòng thử lại sau hoặc liên hệ hỗ trợ."
        )
        return

    # 2. DRL cho BTC (đại diện thị trường) + ML cho coin
    drl_info = None
    btc_feature_info = None
    if USE_DRL and drl_engine:
        try:
            if symbol.upper() == "BTC":
                df_drl = df_feat
                btc_feature_info = feature_table_info
            else:
                df_drl, btc_feature_info = build_feature_table("BTC", lookback_days)
                btc_export_path = export_feature_table_to_csv(df_drl, "BTC", btc_feature_info)
            drl_info = drl_engine.get_policy_signal(df_drl)
        except Exception as exc:  # noqa: BLE001
            logger.warning("DRL engine error: %s", exc)

    ml_info = None
    # Use the simplified XGBoost model from ml_model.py
    try:
        logger.info(f"Running ML model on feature data for {symbol}...")
        ml_info = get_ml_direction_prediction(df_feat)
        logger.info(f"ML prediction: {ml_info.get('label')} (confidence: {ml_info.get('score'):.2%})")
    except Exception as exc:  # noqa: BLE001
        logger.warning("ML direction prediction error for %s: %s", symbol, exc)
        ml_info = None

    # 3. Tin tức + sentiment
    news_items: list[dict] = []
    sentiment_info = {
        "label": "NEUTRAL",
        "avg_polarity": 0.0,
        "counts": {"positive": 0, "neutral": 0, "negative": 0}
    }
    latest_day_counts = {"positive": 0, "neutral": 0, "negative": 0}
    latest_day_str = None

    try:
        # Fetch news with sentiment from the configured provider (use NEWS_LOOKBACK_DAYS)
        if sentiment_provider and sentiment_provider.available:
            news_cache_key = (symbol.upper(), NEWS_LOOKBACK_DAYS, sentiment_provider.provider_name)
            cached_news = _SENTIMENT_NEWS_CACHE.get(news_cache_key)
            if cached_news:
                cached_at = cached_news.get("fetched_at")
                if cached_at and datetime.now(timezone.utc) - cached_at < timedelta(seconds=SENTIMENT_NEWS_TTL_SECONDS):
                    fetched_news = cached_news.get("items") or []
                else:
                    fetched_news = sentiment_provider.fetch_news_with_sentiment(symbol, days=NEWS_LOOKBACK_DAYS)
                    _SENTIMENT_NEWS_CACHE[news_cache_key] = {
                        "fetched_at": datetime.now(timezone.utc),
                        "items": fetched_news,
                    }
            else:
                fetched_news = sentiment_provider.fetch_news_with_sentiment(symbol, days=NEWS_LOOKBACK_DAYS)
                _SENTIMENT_NEWS_CACHE[news_cache_key] = {
                    "fetched_at": datetime.now(timezone.utc),
                    "items": fetched_news,
                }
            if fetched_news:
                latest_day = None
                for item in fetched_news:
                    published = item.get("published_date") or item.get("published")
                    if not published:
                        continue
                    published_dt = pd.to_datetime(published, utc=True, errors="coerce")
                    if pd.isna(published_dt):
                        continue
                    published_day = published_dt.date()
                    if latest_day is None or published_day > latest_day:
                        latest_day = published_day
                if latest_day:
                    latest_day_str = latest_day.isoformat()
                    for item in fetched_news:
                        published = item.get("published_date") or item.get("published")
                        published_dt = pd.to_datetime(published, utc=True, errors="coerce")
                        if pd.isna(published_dt) or published_dt.date() != latest_day:
                            continue
                        sentiment_label = (item.get("sentiment_label") or "").lower()
                        if sentiment_label in latest_day_counts:
                            latest_day_counts[sentiment_label] += 1

                # Compute sentiment statistics
                scores = [item.get("sentiment_score") for item in fetched_news]
                scores = [s for s in scores if s is not None]
                avg_sentiment = float(np.mean(scores)) if scores else 0.0
                
                # Classify sentiment
                if avg_sentiment > 0.15:
                    sentiment_label = "POSITIVE"
                elif avg_sentiment < -0.15:
                    sentiment_label = "NEGATIVE"
                else:
                    sentiment_label = "NEUTRAL"
                
                # Count sentiments
                positive_count = sum(1 for s in scores if s > 0.15)
                negative_count = sum(1 for s in scores if s < -0.15)
                neutral_count = len(scores) - positive_count - negative_count
                
                sentiment_info = {
                    "label": sentiment_label,
                    "avg_polarity": float(avg_sentiment),
                    "counts": {
                        "positive": positive_count,
                        "neutral": neutral_count,
                        "negative": negative_count
                    },
                    "items": fetched_news,
                    "latest_day": latest_day_str,
                    "latest_day_counts": latest_day_counts
                }
                news_items = fetched_news
        else:
            logger.warning(f"Sentiment provider not available for {symbol}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("News/Sentiment error for %s: %s", symbol, exc)
        # Ensure sentiment_info has all required fields even on error
        sentiment_info = {
            "label": "NEUTRAL",
            "avg_polarity": 0.0,
            "counts": {"positive": 0, "neutral": 0, "negative": 0},
            "items": []
        }
        news_items = []

    sentiment_daily = pd.DataFrame()
    if news_items and sentiment_provider:
        df_sentiment_news = pd.DataFrame({
            "Ngày": [item.get("published_date") or item.get("published") for item in news_items],
            "Sentiment_Score": [item.get("sentiment_score") for item in news_items],
        })
        df_sentiment_news = df_sentiment_news.dropna(subset=["Sentiment_Score"])
        if not df_sentiment_news.empty:
            try:
                sentiment_daily = sentiment_provider.summarize_daily_sentiment(df_sentiment_news)
                if not sentiment_daily.empty and "avg_sentiment_score" in sentiment_daily.columns:
                    sentiment_daily = sentiment_daily.rename(columns={"avg_sentiment_score": "sentiment_score"})
            except Exception as exc:  # noqa: BLE001
                logger.warning("Sentiment daily aggregation failed: %s", exc)
        logger.info("Sentiment dataset built: %s rows (separate from model features)", len(df_sentiment_news))

    # 4. Features summary cho RAG / explanation
    features_summary = {
        "symbol": symbol,
        "last_close": float(df_feat["Close"].iloc[-1]),
        "last_return": float(df_feat["Log_return"].iloc[-1]),
        "rsi_14": float(df_feat["RSI_14"].iloc[-1]),
        "macd_hist": float(df_feat["MACD_Histogram"].iloc[-1]),
        "atr_14": float(df_feat["ATR_14"].iloc[-1]),
        "bb_mid_20": float(df_feat["BB_Mid_20"].iloc[-1]) if "BB_Mid_20" in df_feat.columns else None,
        "bb_upper_20": float(df_feat["BB_Upper_20"].iloc[-1]) if "BB_Upper_20" in df_feat.columns else None,
        "bb_lower_20": float(df_feat["BB_Lower_20"].iloc[-1]) if "BB_Lower_20" in df_feat.columns else None,
    }

    macro_cols = ["VIX", "DXY", "SP500", "DOWJONES", "GOLD", "BRENT"]
    for col in macro_cols:
        if col in df_feat.columns:
            features_summary[col] = float(df_feat[col].iloc[-1])
    macro_snapshot = {
        col: features_summary.get(col)
        for col in macro_cols
        if features_summary.get(col) is not None
    }

    # Giá real-time từ CoinGecko để hiển thị
    realtime_price = get_realtime_price_coingecko(symbol, vs_currency="USD")

    last_data_timestamp = df_feat.index[-1]
    data_time_vn = to_vietnam_time(last_data_timestamp)
    analysis_time_vn = to_vietnam_time(datetime.now(timezone.utc))
    data_time_text = format_vietnam_time(data_time_vn)
    analysis_time_text = format_vietnam_time(analysis_time_vn)

    # 5. Tổng hợp kết quả
    news_counts_raw = sentiment_info.get("counts") or {
        "positive": 0,
        "neutral": 0,
        "negative": 0,
    }
    news_counts = {
        "positive": int(news_counts_raw.get("positive", 0)),
        "neutral": int(news_counts_raw.get("neutral", 0)),
        "negative": int(news_counts_raw.get("negative", 0)),
    }
    news_sum = {
        "counts": news_counts,
        "avg_polarity": sentiment_info.get("avg_polarity", 0.0)
    }

    sentiment_label = "Trung lập"
    if news_sum["avg_polarity"] > 0.15:
        sentiment_label = "Tích cực"
    elif news_sum["avg_polarity"] < -0.15:
        sentiment_label = "Tiêu cực"

    # Use weighted recommendation engine with specific weights:
    # - ML Model (XGBoost): 0.6794
    # - DRL Model: 0.1
    # - News Sentiment (3-day avg): 0.2206
    rec_detailed = generate_final_recommendation(
        symbol=symbol,
        ml_info=ml_info,
        drl_info=drl_info,
        sentiment_daily=sentiment_daily,
        weights=RECOMMENDATION_WEIGHTS
    )
    
    # Convert to legacy format for backward compatibility with rest of code
    rec = {
        "recommendation": rec_detailed.get("action", "HOLD"),
        "score": rec_detailed.get("final_score", 0.0),
        "score_percent": (rec_detailed.get("final_score", 0.0) + 1.0) / 2.0 * 100.0,
        "confidence": rec_detailed.get("confidence", 0.0),
        "confidence_percent": rec_detailed.get("confidence_percent", 0.0),
        "model_note": (
            f"ML: {rec_detailed['component_scores'].get('ml_model', 0.0):+.2f} x{rec_detailed['weights'].get('ml_model', 0.0):.2f} | "
            f"DRL: {rec_detailed['component_scores'].get('drl_model', 0.0):+.2f} x{rec_detailed['weights'].get('drl_model', 0.0):.2f} | "
            f"News: {rec_detailed['component_scores'].get('news_sentiment', 0.0):+.2f} x{rec_detailed['weights'].get('news_sentiment', 0.0):.2f}"
        ),
        "model_weights": rec_detailed.get("weights", {}),
        # Add new recommendation details for enhanced analysis
        "weighted_recommendation_details": rec_detailed,
    }

    tech = technical_score(df_feat)

    displayed_price = realtime_price if realtime_price is not None else features_summary["last_close"]

    chart_file = None
    try:
        chart_file = generate_candlestick_html(df_feat, symbol)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Chart generation error for %s: %s", symbol, exc)

    displayed_news_items = []
    display_news_label = f"{DISPLAY_NEWS_DAYS} ngày"
    now_utc = datetime.now(timezone.utc)
    if DISPLAY_NEWS_DAYS <= 0:
        filtered_news_items = [item for item in news_items if item]
    else:
        cutoff_date = (now_utc - timedelta(days=DISPLAY_NEWS_DAYS)).date()
        filtered_news_items = []
        for item in news_items:
            if not item:
                continue
            published = item.get("published_date") or item.get("published")
            published_dt = pd.to_datetime(published, utc=True, errors="coerce")
            if pd.isna(published_dt):
                continue
            if published_dt.date() >= cutoff_date:
                filtered_news_items.append(item)
    try:
        displayed_news_items = filtered_news_items
        news_lines = [format_compact_news_line(item) for item in displayed_news_items]
        news_block = "\n".join(news_lines) if news_lines else f"Chưa có tin tức mới trong {display_news_label} qua."
    except Exception as exc:  # noqa: BLE001
        logger.warning("News formatting error: %s", exc)
        news_block = f"Chưa có tin tức mới trong {display_news_label} qua."

    display_counts = {"positive": 0, "neutral": 0, "negative": 0}
    display_scores = []
    for item in displayed_news_items:
        score = item.get("sentiment_score")
        if score is None:
            continue
        display_scores.append(score)
        if score > 0.15:
            display_counts["positive"] += 1
        elif score < -0.15:
            display_counts["negative"] += 1
        else:
            display_counts["neutral"] += 1

    display_avg_sentiment = float(np.mean(display_scores)) if display_scores else 0.0
    display_sentiment_label = "Trung lập"
    if display_avg_sentiment > 0.15:
        display_sentiment_label = "Tích cực"
    elif display_avg_sentiment < -0.15:
        display_sentiment_label = "Tiêu cực"

    sentiment_label_html = html_escape(display_sentiment_label)
    symbol_html = html_escape(symbol)
    analysis_time_html = html_escape(analysis_time_text)
    bb_mid = features_summary.get("bb_mid_20")
    bb_upper = features_summary.get("bb_upper_20")
    bb_lower = features_summary.get("bb_lower_20")
    bb_line = (
        f"- BB: Mid {bb_mid:.2f} - Upper {bb_upper:.2f} - Lower {bb_lower:.2f}\n"
        if bb_mid is not None and bb_upper is not None and bb_lower is not None
        else "- BB(20): N/A\n"
    )
    label_vi_map = {
        "BUY": "Mua",
        "SELL": "Bán",
        "HOLD": "Giữ",
        "UP": "Tăng",
        "DOWN": "Giảm",
        "UNKNOWN": "Không rõ",
        "N/A": "N/A",
    }

    xgb_line = "N/A"
    if ml_info:
        raw_xgb_line = f"{ml_info.get('label','N/A')}"
        xgb_line = html_escape(label_vi_map.get(raw_xgb_line, raw_xgb_line))

    drl_line = "N/A"
    if drl_info:
        raw_drl_line = f"{drl_info.get('label','N/A')}"
        drl_line = html_escape(label_vi_map.get(raw_drl_line, raw_drl_line))

    analysis_context = {
        "symbol": symbol,
        "lookback_days": lookback_days,
        "indicators": {
            "last_close": displayed_price,
            "last_return": features_summary["last_return"],
            "rsi_14": features_summary["rsi_14"],
            "macd_hist": features_summary["macd_hist"],
            "atr_14": features_summary["atr_14"],
        },
        "technical_score": tech,
        "sentiment_info": sentiment_info,
        "news_summary": news_sum,
        "news_items": news_items,
        "news_items_display": displayed_news_items,
        "drl_info": drl_info,
        "ml_info": ml_info,
        "model_weights": rec_detailed.get("weights", {}),
        "recommendation": rec,
        "timestamp": analysis_time_text,
        "analysis_time": analysis_time_text,
        "data_timestamp": data_time_text,
        "feature_table": feature_table_info,
        "btc_feature_table": btc_feature_info,
        "chart_html_path": chart_file,
        "macro_snapshot": macro_snapshot,
        "data_export_path": export_path,
        "btc_data_export_path": btc_export_path,
        "sentiment_provider": sentiment_provider.provider_name if sentiment_provider else None,
    }

    # 6. Soạn thảo câu trả lời ngắn gọn
    risk_score = None
    if USE_RISK_SCORE:
        risk_score = compute_risk_score(
            indicators=analysis_context["indicators"],
            technical=tech,
            news_sum=news_sum,
            macro_snapshot=macro_snapshot
        )
    analysis_context["risk_score"] = risk_score

    conversation_history = get_recent_history(context.user_data)
    rag_context = build_rag_context(analysis_context, conversation_history)
    if rag_engine:
        thesis_prompt = "Hãy viết luận điểm đầu tư ngắn gọn cho khuyến nghị hiện tại."
        if USE_RISK_SCORE and risk_score:
            thesis_prompt += " Nêu rõ mức risk score và tác động tới quản trị rủi ro."
        investment_thesis = rag_engine.answer(thesis_prompt, rag_context)
    else:
        investment_thesis = RAGEngine._fallback_explanation(
            rag_context,
            note="RAG engine chưa sẵn sàng"
        )
    if not investment_thesis:
        investment_thesis = "Chưa có luận điểm đầu tư phù hợp từ dữ liệu hiện tại."
    analysis_context["investment_thesis"] = investment_thesis
    rec_label = html_escape(rec.get("recommendation") or "N/A")
    weight_detail = rec_detailed.get("weights", {}) if rec_detailed else {}
    w_news = float(weight_detail.get("news_sentiment", 0.0))
    w_drl = float(weight_detail.get("drl_model", 0.0))
    w_xgb = float(weight_detail.get("ml_model", 0.0))
    
    risk_block = ""
    if USE_RISK_SCORE and risk_score:
        risk_block = (
            f"🛡️ <b>Risk Score:</b> {risk_score['score']}/100 ({risk_score['level']})\n"
            f"- Biến động: {risk_score['components']['volatility']}/30\n"
            f"- Xu hướng: {risk_score['components']['trend']}/30\n"
            f"- Tin tức: {risk_score['components']['sentiment']}/20\n"
            f"- Vĩ mô: {risk_score['components']['macro']}/20\n\n"
        )

    answer_text = (
        f"📌 <b>BÁO CÁO PHÂN TÍCH {symbol_html} - {analysis_time_html}</b>\n\n"
        f"💰 <b>Giá hiện tại:</b> {displayed_price:.2f}\n\n"
        # f"📅 <b>Dữ liệu OHLCV cập nhật đến:</b> {data_time_text}\n\n"
        "📈 <b>Phân tích kỹ thuật:</b>\n"
        f"- RSI: {features_summary['rsi_14']:.2f}\n"
        f"- MACD Histogram: {features_summary['macd_hist']:.4f}\n"
        # f"{bb_line}"
        f"- ATR: {features_summary['atr_14']:.4f}\n\n"
        f"📌 <b>Cảm xúc từ tin tức:</b> {sentiment_label_html}\n"
        f"- Tích cực: {display_counts['positive']}\n"
        f"- Trung lập: {display_counts['neutral']}\n"
        f"- Tiêu cực: {display_counts['negative']}\n\n"
        "🤖 <b>Kết quả mô hình dự báo:</b>\n"
        f"- Xu hướng thị trường: {drl_line}\n"
        f"- Kết quả mô hình: {xgb_line}\n\n"
        f"🎯 <b>Khuyến nghị cuối cùng:</b> {rec_label}\n\n"
        f"{risk_block}"
        "⚠️ <i>Lưu ý: Đây chỉ là khuyến nghị tham khảo. Quyết định cuối cùng thuộc về nhà đầu tư.</i>"
    )

    reply_markup = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📰 Tin tức mới nhất", callback_data=f"news:{symbol}"),
            InlineKeyboardButton("📈 Biểu đồ giá", callback_data=f"chart:{symbol}"),
            InlineKeyboardButton("💡 Luận điểm đầu tư", callback_data=f"thesis:{symbol}"),
        ]
    ])

    # 7. Trả lời người dùng
    try:
        await loading_msg.edit_text(
            answer_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True
        )
    except Exception:
        await message.reply_text(
            answer_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True
        )

    analysis_key = f"last_analysis_{symbol}"
    context.user_data["last_symbol"] = symbol
    context.user_data[analysis_key] = analysis_context
    log_bot_message(context.user_data, answer_text, symbol)


def _build_application() -> Application:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Thiếu TELEGRAM_BOT_TOKEN trong môi trường .env")

    _ensure_httpx_version()
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("about", about_command))
    app.add_handler(CommandHandler("renew", renew_command))
    app.add_handler(CommandHandler("train", train_command))
    app.add_handler(CommandHandler("analyze", analyze_command))
    app.add_handler(CommandHandler("data", data_command))
    app.add_handler(CallbackQueryHandler(handle_news_callback, pattern=r"^news:"))

    # schedule a daily BTC data refresh and DRL training at 00:30 UTC (07:30 Vietnam time)
    try:
        scheduler = AsyncIOScheduler()
        trigger = CronTrigger(hour=0, minute=30, timezone=pytz.UTC)
        scheduler.add_job(daily_btc_update, trigger=trigger)
        scheduler.start()
        logger.info("Scheduler started: daily BTC update at 00:30 UTC")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to start scheduler: %s", exc)
    app.add_handler(CallbackQueryHandler(handle_thesis_callback, pattern=r"^thesis:"))
    app.add_handler(CallbackQueryHandler(handle_chart_callback, pattern=r"^chart:"))
    app.add_error_handler(error_handler)

    # Text after an analysis will be treated as follow-up questions
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_text))

    return app


def main():
    try:
        app = _build_application()
    except Exception as exc:  # noqa: BLE001
        logger.error("Không khởi tạo được bot: %s", exc)
        return

    logger.info("🚀 Bot đang chạy. Nhấn Ctrl+C để dừng.")
    app.run_polling()


if __name__ == "__main__":
    main()
