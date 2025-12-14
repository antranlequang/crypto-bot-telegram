# bot.py

import os
import io
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from html import escape as html_escape

import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from keep_alive import keep_alive

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

from config import (
    TELEGRAM_BOT_TOKEN,
    CMC_API_KEY, CMC_BASE_URL,
    CRYPTOPANIC_API_KEY, CRYPTOPANIC_BASE_URL,
    CRYPTONEWS_API_KEY, CRYPTO_NEWS_BASE_URL,
    PPO_BTC_MODEL_PATH, XGB_DIRECTION_MODEL_PATH,
    WINDOW_SIZE, DATA_LOOKBACK_DAYS, CHART_FOLDER,
    RAG_SYSTEM_PROMPT, ANALYSIS_QUESTION_KEYWORDS,
    EXPLANATION_REPLY_TEMPLATE, EXPLANATION_PARSE_MODE
)

# ---------- Logging ----------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ---------- Utils ----------
def normalize_symbol(symbol: str) -> str:
    return symbol.upper().replace("USDT", "").replace("USD", "")


MAX_CONVERSATION_HISTORY = 50
MAX_RAG_HISTORY = 10
VIETNAM_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
AUTO_REFRESH_INTERVAL = timedelta(hours=1)
UNCLEAR_REPLY = "Vui lòng làm rõ lại câu hỏi, tôi vẫn chưa hiểu rõ ý bạn."


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


# ---------- Data Provider (OHLCV daily) ----------
class CryptoDataProvider:
    """
    Lấy OHLCV daily từ nhiều nguồn:
    - CoinGecko (free, no API key needed) - primary
    - CoinMarketCap (paid API) - fallback if key provided
    - Binance (free public API) - fallback
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key

    def _get_coingecko_ohlcv(self, symbol: str, vs_currency: str = "usd", days: int = 365):
        """CoinGecko API - free, no API key required"""
        try:
            # Map common symbols to CoinGecko IDs
            symbol_map = {
                "BTC": "bitcoin",
                "ETH": "ethereum",
                "BNB": "binancecoin",
                "SOL": "solana",
                "ADA": "cardano",
                "XRP": "ripple",
                "DOT": "polkadot",
                "DOGE": "dogecoin",
                "MATIC": "matic-network",
                "AVAX": "avalanche-2",
                "LINK": "chainlink",
                "UNI": "uniswap",
                "ATOM": "cosmos",
                "LTC": "litecoin",
                "ETC": "ethereum-classic",
            }
            
            coin_id = symbol_map.get(symbol.upper(), symbol.lower())
            
            # CoinGecko API endpoint
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
            params = {
                "vs_currency": vs_currency,
                "days": min(days, 365)  # CoinGecko free tier limit
            }
            
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            if not data:
                raise ValueError(f"No data returned from CoinGecko for {symbol}")
            
            records = []
            for item in data:
                # CoinGecko returns: [timestamp, open, high, low, close]
                ts = pd.to_datetime(item[0], unit='ms')
                o = item[1]
                h = item[2]
                l = item[3]
                c = item[4]
                v = 0  # CoinGecko OHLC doesn't include volume in this endpoint
                records.append([ts, o, h, l, c, v])
            
            df = pd.DataFrame(records, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)
            
            # Get volume from market data endpoint if needed
            try:
                market_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
                market_params = {"vs_currency": vs_currency, "days": min(days, 365), "interval": "daily"}
                market_resp = requests.get(market_url, params=market_params, timeout=10)
                if market_resp.status_code == 200:
                    market_data = market_resp.json()
                    volumes = market_data.get("total_volumes", [])
                    if volumes:
                        volume_dict = {pd.to_datetime(v[0], unit='ms'): v[1] for v in volumes}
                        df["Volume"] = df.index.map(lambda x: volume_dict.get(x, 0))
            except:
                pass  # Volume is optional
            
            logger.info(f"✓ CoinGecko: Fetched {len(df)} days of data for {symbol}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"CoinGecko API request error: {e}")
            raise
        except (KeyError, ValueError, IndexError) as e:
            logger.warning(f"CoinGecko data parsing error: {e}")
            raise ValueError(f"Could not parse CoinGecko data for {symbol}")

    def _get_binance_ohlcv(self, symbol: str, days: int = 365):
        """Binance public API - free, no API key needed"""
        try:
            # Binance uses trading pairs like BTCUSDT
            pair = f"{symbol.upper()}USDT"
            
            # Calculate start time
            end_time = int(datetime.utcnow().timestamp() * 1000)
            start_time = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
            
            url = "https://api.binance.com/api/v3/klines"
            params = {
                "symbol": pair,
                "interval": "1d",
                "startTime": start_time,
                "endTime": end_time,
                "limit": 1000
            }
            
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            if not data:
                raise ValueError(f"No data returned from Binance for {symbol}")
            
            records = []
            for kline in data:
                ts = pd.to_datetime(kline[0], unit='ms')
                o = float(kline[1])
                h = float(kline[2])
                l = float(kline[3])
                c = float(kline[4])
                v = float(kline[5])
                records.append([ts, o, h, l, c, v])
            
            df = pd.DataFrame(records, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)
            
            logger.info(f"✓ Binance: Fetched {len(df)} days of data for {symbol}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Binance API request error: {e}")
            raise
        except (KeyError, ValueError, IndexError) as e:
            logger.warning(f"Binance data parsing error: {e}")
            raise ValueError(f"Could not parse Binance data for {symbol}")

    def _get_cmc_ohlcv(self, symbol: str, vs_currency: str = "USD", days: int = 365):
        """CoinMarketCap API - requires paid API key"""
        if not self.api_key:
            raise ValueError("CoinMarketCap API key required but not provided")
        
        try:
            url = f"{CMC_BASE_URL}/cryptocurrency/ohlcv/historical"
            params = {
                "symbol": symbol,
                "time_start": (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d"),
                "time_end": datetime.utcnow().strftime("%Y-%m-%d"),
                "interval": "daily"
            }
            headers = {"X-CMC_PRO_API_KEY": self.api_key}

            resp = requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if "data" not in data or "quotes" not in data["data"]:
                raise ValueError(f"Unexpected CoinMarketCap response structure")

            quotes = data["data"]["quotes"]

            records = []
            for q in quotes:
                ts = pd.to_datetime(q["time_open"])
                o = q["quote"][vs_currency]["open"]
                h = q["quote"][vs_currency]["high"]
                l = q["quote"][vs_currency]["low"]
                c = q["quote"][vs_currency]["close"]
                v = q["quote"][vs_currency]["volume"]
                records.append([ts, o, h, l, c, v])

            df = pd.DataFrame(records, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)
            
            logger.info(f"✓ CoinMarketCap: Fetched {len(df)} days of data for {symbol}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"CoinMarketCap API request error: {e}")
            raise
        except (KeyError, ValueError, IndexError) as e:
            logger.warning(f"CoinMarketCap data parsing error: {e}")
            raise ValueError(f"Could not parse CoinMarketCap data for {symbol}")

    def get_daily_ohlcv(self, symbol: str, vs_currency: str = "USD", days: int = 365):
        """
        Trả về DataFrame có cột: ['open','high','low','close','volume'] index = datetime
        Tự động thử nhiều nguồn: CoinGecko (free) -> Binance (free) -> CoinMarketCap (paid)
        """
        errors = []
        
        # Try CoinGecko first (free, no API key needed)
        try:
            return self._get_coingecko_ohlcv(symbol, vs_currency.lower(), days)
        except Exception as e:
            error_msg = f"CoinGecko: {str(e)}"
            errors.append(error_msg)
            logger.debug(error_msg)
        
        # Try Binance as fallback (free, no API key needed)
        try:
            return self._get_binance_ohlcv(symbol, days)
        except Exception as e:
            error_msg = f"Binance: {str(e)}"
            errors.append(error_msg)
            logger.debug(error_msg)
        
        # Try CoinMarketCap if API key is available (paid API)
        if self.api_key:
            try:
                return self._get_cmc_ohlcv(symbol, vs_currency, days)
            except Exception as e:
                error_msg = f"CoinMarketCap: {str(e)}"
                errors.append(error_msg)
                logger.debug(error_msg)
        
        # If all sources failed, raise with detailed error message
        error_summary = "; ".join(errors)
        raise ValueError(
            f"Failed to fetch data for {symbol} from all sources. "
            f"Errors: {error_summary}. "
            f"Please check if the symbol is correct and try again."
        )


# ---------- Technical Indicators ----------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính một số chỉ báo cơ bản. Bạn có thể mở rộng thêm theo bộ indicator bạn đã làm.
    """
    df = df.copy()
    df["Log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # SMA, EMA
    df["SMA_14"] = df["Close"].rolling(14).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # MACD
    df["MACD_Line"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal_Line"] = df["MACD_Line"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD_Line"] - df["MACD_Signal_Line"]

    # RSI 14
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # ATR 14 (True Range simplified)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()

    df.dropna(inplace=True)
    return df

# ---------- Recommendation Logic (weighted signals) ----------

def summarize_news_sentiment(news_items):
    """
    Đếm số tin Tích cực/Trung lập/Tiêu cực theo TextBlob polarity.
    """
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    polarities = []

    if not news_items:
        return {"counts": counts, "avg_polarity": 0.0}

    for n in news_items:
        title = (n.get("title") or "").strip()
        if not title:
            continue
        try:
            pol = TextBlob(title).sentiment.polarity
        except Exception:
            continue

        polarities.append(pol)
        if pol > 0.15:
            counts["positive"] += 1
        elif pol < -0.15:
            counts["negative"] += 1
        else:
            counts["neutral"] += 1

    avg_pol = float(np.mean(polarities)) if polarities else 0.0
    return {"counts": counts, "avg_polarity": avg_pol}


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


def final_recommendation(symbol: str, tech: dict, news_sum: dict, drl_info: dict | None, ml_info: dict | None) -> dict:
    """
    Tổng hợp BUY/SELL/HOLD theo trọng số:
    BTC (DRL): 0.45 model, 0.40 tech, 0.15 sentiment
    Alt (ML):  0.40 model, 0.45 tech, 0.15 sentiment
    """
    # Model score
    model_score = 0.0
    model_note = "Chưa phát hiện tín hiệu"

    if symbol == "BTC" and drl_info:
        label = drl_info.get("label", "HOLD")
        model_score = 1.0 if label == "BUY" else (-1.0 if label == "SELL" else 0.0)
        model_note = f"{label}"
    elif ml_info:
        pu = float(ml_info.get("proba_up", 0.5))
        pdn = float(ml_info.get("proba_down", 0.5))
        model_score = float(np.clip(pu - pdn, -1.0, 1.0))
        model_note = f"ML: {ml_info.get('label','UNKNOWN')} (Up {pu:.0%} / Down {pdn:.0%})"

    # News score (scaled)
    news_pol = float(news_sum.get("avg_polarity", 0.0))
    news_score = float(np.clip(news_pol / 0.5, -1.0, 1.0))

    if symbol == "BTC" and drl_info:
        w_model, w_tech, w_news = 0.45, 0.40, 0.15
    else:
        w_model, w_tech, w_news = 0.40, 0.45, 0.15

    total = w_model * model_score + w_tech * float(tech["score"]) + w_news * news_score
    total = float(np.clip(total, -1.0, 1.0))
    score_percent = (total + 1.0) / 2.0 * 100.0        # 0% = SELL mạnh, 50% = trung lập, 100% = BUY mạnh
    confidence_percent = abs(total) * 100.0            # độ mạnh tín hiệu

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
        "model_note": model_note
    }

def build_rag_context(analysis_context: dict, conversation_history: list[dict] | None = None) -> dict:
    indicators = analysis_context.get("indicators", {}) or {}
    tech = analysis_context.get("technical_score", {}) or {}
    news_sum = analysis_context.get("news_summary", {}) or {}
    sent = analysis_context.get("sentiment_info", {}) or {}
    rec = analysis_context.get("recommendation", {}) or {}
    ml = analysis_context.get("ml_info") or {}
    history_tail = []
    if conversation_history:
        history_tail = conversation_history[-MAX_RAG_HISTORY:]

    def _to_float(x, default=None):
        try:
            return float(x)
        except Exception:
            return default

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


# ---------- ML Model cho direction các coin khác ----------
class DirectionMLModel:
    """
    Model ML cho direction. Ở đây dùng XGBoost classifier,
    theo nhiều nghiên cứu là lựa chọn mạnh cho trend classification.
    >>>> Nếu bạn muốn đổi model, chỉ cần thay class này.
    """

    def __init__(self, model_path: str):
        # Lazy import xgboost to avoid OpenMP conflicts
        try:
            import xgboost as xgb
            self.xgb = xgb
            self.model = xgb.XGBClassifier()
            if os.path.exists(model_path):
                self.model.load_model(model_path)
            else:
                logger.warning("XGBoost model file not found. Using untrained stub.")
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
        feature_cols = [
            "Close", "Volume", "Log_return",
            "SMA_14", "EMA_12", "EMA_26",
            "MACD_Line", "MACD_Histogram",
            "RSI_14", "ATR_14"
        ]
        x_last = df_feat[feature_cols].iloc[-1:].values
        return x_last

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


# ---------- News & Sentiment ----------
from textblob import TextBlob
from datetime import datetime

# Try to import feedparser for RSS feeds (optional)
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    logger.info("feedparser not available - RSS feeds will be skipped. Install with: pip install feedparser")

class NewsSentimentProvider:
    """
    Lấy news từ nhiều nguồn miễn phí, uy tín:
    - CoinGecko API (free, no key needed)
    - CryptoPanic (free tier)
    - RSS feeds từ CoinDesk, Cointelegraph
    Tự động bỏ qua nguồn lỗi, chỉ log ra terminal.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.sources = []

    def _fetch_coingecko_news(self, symbol: str, limit: int = 10):
        """CoinGecko API - free, no API key required"""
        try:
            # CoinGecko doesn't have a direct news API, but we can use their trending endpoint
            # For actual news, we'll use RSS feeds instead
            return []
        except Exception as e:
            logger.debug(f"CoinGecko news fetch skipped: {e}")
            return []

    def _fetch_cryptopanic_news(self, symbol: str, limit: int = 10):
        """CryptoPanic API - free tier available"""
        news_list = []
        if not self.api_key:
            logger.debug("CryptoPanic API key not provided, skipping")
            return news_list
        
        try:
            url = "https://cryptopanic.com/api/v1/posts/"
            params = {
                "auth_token": self.api_key,
                "currencies": symbol.upper(),
                "kind": "news",
                "public": "true",
                "filter": "hot"
            }
            resp = requests.get(url, params=params, timeout=5)
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("results", [])[:limit]:
                title = item.get("title")
                news_url = item.get("url")
                created = item.get("created_at")

                if title:
                    news_list.append({
                        "title": title,
                        "url": news_url or "",
                        "published": created or ""
                    })

            logger.info(f"✓ CryptoPanic: Fetched {len(news_list)} news items")
            return news_list
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠ CryptoPanic API error (skipping): {e}")
            return []
        except Exception as e:
            logger.warning(f"⚠ CryptoPanic error (skipping): {e}")
            return []

    def _fetch_rss_news(self, symbol: str, limit: int = 10):
        """Fetch news from RSS feeds - completely free, no API key needed"""
        news_list = []
        
        if not FEEDPARSER_AVAILABLE:
            logger.debug("RSS feeds skipped: feedparser not installed")
            return news_list
        
        # Reputable crypto news RSS feeds
        rss_feeds = [
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://cointelegraph.com/rss",
            "https://www.theblock.co/rss.xml",
        ]
        
        for feed_url in rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                if feed.bozo:
                    logger.debug(f"RSS feed parse error for {feed_url}: {feed.bozo_exception}")
                    continue
                
                for entry in feed.entries[:limit]:
                    title = entry.get("title", "")
                    link = entry.get("link", "")
                    published = entry.get("published", "")
                    
                    # Filter by symbol if mentioned in title or summary
                    title_lower = title.lower()
                    summary_lower = entry.get("summary", "").lower()
                    symbol_lower = symbol.lower()
                    
                    # Check if symbol is mentioned (or get general crypto news)
                    if symbol_lower in title_lower or symbol_lower in summary_lower or symbol.upper() in title:
                        news_list.append({
                            "title": title,
                            "url": link,
                            "published": published
                        })
                    elif symbol.upper() == "BTC" or symbol.upper() == "ETH":  # For major coins, include general crypto news
                        news_list.append({
                            "title": title,
                            "url": link,
                            "published": published
                        })
                
                if news_list:
                    logger.info(f"✓ RSS feed ({feed_url}): Fetched {len(news_list)} relevant news items")
                    break  # Got enough from one source
                        
            except requests.exceptions.RequestException as e:
                logger.debug(f"RSS feed request error for {feed_url} (skipping): {e}")
                continue
            except Exception as e:
                logger.debug(f"RSS feed error for {feed_url} (skipping): {e}")
                continue
        
        return news_list[:limit]

    def fetch_news(self, symbol: str, limit: int = 10):
        """
        Fetch news from multiple sources, gracefully skip failures.
        Returns empty list if all sources fail (bot continues normally).
        """
        all_news = []
        
        # Try CryptoPanic first (if API key available)
        try:
            cryptopanic_news = self._fetch_cryptopanic_news(symbol, limit)
            all_news.extend(cryptopanic_news)
        except Exception as e:
            logger.debug(f"CryptoPanic fetch exception (skipped): {e}")
        
        # Try RSS feeds (always free)
        try:
            rss_news = self._fetch_rss_news(symbol, limit)
            all_news.extend(rss_news)
        except Exception as e:
            logger.debug(f"RSS fetch exception (skipped): {e}")
        
        # Remove duplicates based on title
        seen_titles = set()
        unique_news = []
        for item in all_news:
            title_lower = item["title"].lower()
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_news.append(item)
        
        # Limit results
        result = unique_news[:limit]
        
        if not result:
            logger.info(f"ℹ No news found for {symbol} (all sources skipped or returned empty)")
        else:
            logger.info(f"✓ Total news fetched for {symbol}: {len(result)} items")
        
        return result

    def analyze_sentiment(self, news_items):
        """
        Sentiment tổng hợp từ TextBlob
        """
        if not news_items:
            return {"label": "NEUTRAL", "avg_polarity": 0, "subjectivity": 0}

        try:
            polarities = []
            subjects = []

            for item in news_items:
                try:
                    blob = TextBlob(item["title"])
                    polarities.append(blob.sentiment.polarity)
                    subjects.append(blob.sentiment.subjectivity)
                except Exception as e:
                    logger.debug(f"TextBlob sentiment error for item (skipping): {e}")
                    continue

            if not polarities:
                return {"label": "NEUTRAL", "avg_polarity": 0, "subjectivity": 0}

            avg_pol = np.mean(polarities)
            avg_subj = np.mean(subjects)

            if avg_pol > 0.15:
                label = "BULLISH"
            elif avg_pol < -0.15:
                label = "BEARISH"
            else:
                label = "NEUTRAL"

            return {
                "label": label,
                "avg_polarity": float(avg_pol),
                "subjectivity": float(avg_subj)
            }
        except Exception as e:
            logger.warning(f"Sentiment analysis error (returning neutral): {e}")
            return {"label": "NEUTRAL", "avg_polarity": 0, "subjectivity": 0}


# ---------- Gemini RAG Engine (Gemini API) ----------
import google.generativeai as genai
from config import GEMINI_API_KEY

DEFAULT_EXPLANATION_TEMPLATE = "💡 Giải thích khuyến nghị cho {symbol}:\n\n{answer}"


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
        }

    def __init__(self, model_name: str = "gemini-flash-latest"):
        if not GEMINI_API_KEY:
            raise RuntimeError("Thiếu GEMINI_API_KEY trong môi trường")

        genai.configure(api_key=GEMINI_API_KEY)
        normalized_name = model_name.split("/", 1)[-1] if model_name.startswith("models/") else model_name
        if normalized_name.startswith("gemini-1.5"):
            logger.warning("Model '%s' không khả dụng ở API hiện tại, chuyển sang 'gemini-flash-latest'", normalized_name)
            normalized_name = "gemini-flash-latest"
        self.model = genai.GenerativeModel(normalized_name)
        logger.info(f"✓ RAG Engine initialized with Gemini model '{normalized_name}'")

    def answer(self, user_query: str, analysis_context: dict) -> str:
        prompt = self._build_prompt(user_query, analysis_context)

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 2048
                }
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

        return f"""
Bạn là một chuyên gia phân tích thị trường crypto.

Dựa trên thông tin dưới đây, hãy giải thích khuyến nghị đầu tư
theo góc nhìn phân tích kỹ thuật và tâm lý thị trường.

Yêu cầu:
- Không liệt kê dữ liệu dạng bảng
- Không nhắc đến từ "context" hay "dữ liệu đầu vào"
- Viết như đang tư vấn cho nhà đầu tư cá nhân
- Lập luận rõ ràng, mạch lạc
- Độ dài tối đa 6–8 câu
- Ngắt đoạn thành các ý nhỏ, tránh trình bày một đoạn dài
- không cần câu chào luôn, vô giải thích luôn.
- Sử dụng ngôn ngữ tự nhiên, thân thiện.
- Chỉ cần trả lời chung về pha tích, ví dụ nói chung rằng phân tích kỹ thuật hiện tại đang ở pha tích lũy, không cần nói cụ thể là "pha 2" hay "pha 3". Tâm lý thị tường như thế nào, không cần nói cụ thể là "tâm lý sợ hãi" hay "tâm lý tham lam". Nói chung hãy phân bổ để câu trả lời ngắn gọn và đúng trọng tâm.
- Nếu người dùng tiếp tục hỏi nữa, hãy tập trung vào vào trọng tâm câu hỏi, không cần nhắc lại toàn bộ bối cảnh. Lấy ví dụ nếu người dùng hỏi "Còn về chỉ báo RSI thì sao?", bạn chỉ cần trả lời về RSI mà không cần nhắc lại toàn bộ bối cảnh phân tích kỹ thuật.
- Dùng những câu trả lời trước đó trong cuộc trò chuyện để bổ sung ngữ cảnh nếu cần.
- Nếu như người dùng hỏi những câu hỏi không liên quan đến phân tích kỹ thuật hay tâm lý thị trường, hãy lịch sự từ chối trả lời và nhắc người dùng tập trung vào chủ đề chính. Nếu coi hỏi ngoài nội dung liên quan thì hãy trả lời là bạn có thể làm rõ câu hỏi được không.

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

        lines.append("⚠️ Ghi chú: Đây là tóm tắt tự động khi Gemini không phản hồi.")
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

    # Lấy 250 ngày gần nhất
    df = df.tail(250)

    fig = go.Figure()

    # Giá đóng cửa
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Close"],
        name=symbol,
        line=dict(width=2)
    ))

    fig.update_layout(
        title=f"{symbol} - Biểu đồ giá",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(title="Ngày"),
        yaxis=dict(title=f"Giá {symbol}", showgrid=True),
        height=600
    )

    fig.write_html(filename, include_plotlyjs="cdn", full_html=True)
    return filename


# ---------- Khởi tạo các service ----------
# Data provider always works (uses free CoinGecko/Binance APIs, CMC is optional)
try:
    data_provider = CryptoDataProvider(CMC_API_KEY)
    logger.info("✓ Data provider initialized (free APIs available, CoinMarketCap optional)")
except Exception as e:
    logger.error(f"Failed to initialize data provider: {e}")
    data_provider = None

# Initialize DRL engine with error handling
drl_engine = None
if os.path.exists(PPO_BTC_MODEL_PATH):
    try:
        drl_engine = DRLPolicyEngine(PPO_BTC_MODEL_PATH, window_size=WINDOW_SIZE)
        if not drl_engine.available:
            drl_engine = None
            logger.warning("DRL engine initialized but model not available")
    except Exception as e:
        logger.error(f"Failed to initialize DRL engine: {e}")
        drl_engine = None

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

# Initialize news provider (always works - RSS feeds don't need API key)
# API key is optional (only needed for CryptoPanic premium features)
try:
    news_provider = NewsSentimentProvider(CRYPTOPANIC_API_KEY)
    logger.info("✓ News provider initialized (RSS feeds available, CryptoPanic optional)")
except Exception as e:
    logger.warning(f"News provider initialization error (will use fallback): {e}")
    news_provider = NewsSentimentProvider()  # Initialize without API key

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
        "/about - Giới thiệu về bot\n"
        "/renew - Làm mới chat (xóa lịch sử phân tích)\n\n"
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "👋 **Chào mừng đến với Crypto Analysis Bot!**\n\n"
        "Tôi là AI Chatbot chuyên phân tích giá cryptocurrency.\n\n"
        f"{get_commands_menu()}"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for /help command"""
    text = (
        "📖 **HƯỚNG DẪN SỬ DỤNG**\n\n"
        "**1. Phân tích coin:**\n"
        "• Dùng lệnh: `/analyze BTC`\n"
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
    text = (
        "🤖 **GIỚI THIỆU VỀ BOT**\n\n"
        "**Crypto Analysis Bot** là AI chatbot phân tích cryptocurrency sử dụng:\n\n"
        "**📊 Tính năng:**\n"
        "• Phân tích đa yếu tố (kỹ thuật + sentiment + AI models)\n"
        "• Khuyến nghị BUY/SELL/HOLD có trọng số\n"
        "• Biểu đồ nến và chỉ báo kỹ thuật\n"
        "• Tin tức crypto real-time\n"
        "• Giải thích chi tiết bằng AI\n\n"
        "**⚠️ Lưu ý:**\n"
        "Tất cả khuyến nghị chỉ mang tính tham khảo.\n"
        "Quyết định đầu tư cuối cùng thuộc về bạn.\n\n"
        "**📞 Hỗ trợ:**\n"
        "Dùng `/help` để xem hướng dẫn chi tiết."
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def renew_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for /renew command - clears user data"""
    context.user_data.clear()
    context.user_data["history_last_reset"] = datetime.now(timezone.utc).isoformat()
    text = (
        "🔄 **Chat đã được làm mới!**\n\n"
        "Lịch sử phân tích đã được xóa.\n"
        "Bạn có thể bắt đầu phân tích mới.\n\n"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


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
    await run_full_analysis(update, context, symbol)


async def analyze_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Sau khi chạy /analyze, mọi tin nhắn tiếp theo đều được dùng để hỏi đáp dựa trên dữ liệu đã phân tích.
    """
    message = update.message
    await refresh_history_if_needed(message, context)
    raw_text = message.text or ""
    last_symbol = context.user_data.get("last_symbol")
    log_user_message(context.user_data, raw_text, last_symbol)

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
    thinking_msg = await message.reply_text("🤔 Đang phân tích và tạo câu trả lời chi tiết...")
    
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

    await message.reply_text(f"Đang lấy dữ liệu và phân tích {symbol}...")

    # 1. Lấy dữ liệu
    # Data provider is always initialized (uses free APIs)
    try:
        df_raw = data_provider.get_daily_ohlcv(symbol, days=DATA_LOOKBACK_DAYS)
        if df_raw.empty:
            await message.reply_text(
                f"❌ Không tìm thấy dữ liệu cho {symbol}.\n"
                "Vui lòng kiểm tra lại mã coin (ví dụ: BTC, ETH, SOL)."
            )
            return
    except ValueError as e:
        # User-friendly error message
        error_msg = str(e)
        logger.error(f"Error fetching data for {symbol}: {error_msg}")
        await message.reply_text(
            f"❌ Không thể lấy dữ liệu cho {symbol}.\n\n"
            "Vui lòng thử:\n"
            "- Kiểm tra lại mã coin (ví dụ: BTC, ETH, SOL)\n"
            "- Thử lại sau vài giây"
        )
        logger.debug(f"Error details: {error_msg}")
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

    # 2. Tính indicator
    df_feat = compute_indicators(df_raw)

    # 3. DRL cho BTC, ML cho coin khác
    drl_info = None
    ml_info = None
    if symbol == "BTC":
        if drl_engine:
            drl_info = drl_engine.get_policy_signal(df_feat)
        else:
            logger.warning("DRL engine not available")
    else:
        if direction_model:
            ml_info = direction_model.predict_direction(df_feat)
        else:
            logger.warning("Direction model not available")

    # 4. Tin tức + sentiment
    news_items = []
    sentiment_info = {"label": "Trung lập", "avg_polarity": 0, "subjectivity": 0}
    try:
        news_items = news_provider.fetch_news(symbol)
        if news_items:
            sentiment_info = news_provider.analyze_sentiment(news_items)
        # If no news, sentiment stays neutral (bot continues normally)
    except Exception as e:
        # Log error but don't crash - bot continues without news
        logger.warning(f"News fetch error for {symbol} (continuing without news): {e}")
        news_items = []
        sentiment_info = {"label": "Trung lập", "avg_polarity": 0, "subjectivity": 0}

    # 5. Features summary cho RAG / explanation
    features_summary = {
        "symbol": symbol,
        "last_close": float(df_feat["Close"].iloc[-1]),
        "last_return": float(df_feat["Log_return"].iloc[-1]),
        "rsi_14": float(df_feat["RSI_14"].iloc[-1]),
        "macd_hist": float(df_feat["MACD_Histogram"].iloc[-1]),
        "atr_14": float(df_feat["ATR_14"].iloc[-1]),
    }
    last_data_timestamp = df_feat.index[-1]
    data_time_vn = to_vietnam_time(last_data_timestamp)
    analysis_time_vn = to_vietnam_time(datetime.now(timezone.utc))
    data_time_text = format_vietnam_time(data_time_vn)
    analysis_time_text = format_vietnam_time(analysis_time_vn)

    user_question = f"Phân tích và khuyến nghị cho {symbol} dựa trên mô hình hiện có."

    # 6. Answer
    news_sum = summarize_news_sentiment(news_items)
    news_counts = news_sum["counts"]

    sentiment_label = "Trung lập"
    if news_sum["avg_polarity"] > 0.15:
        sentiment_label = "Tích cực"
    elif news_sum["avg_polarity"] < -0.15:
        sentiment_label = "Tiêu cực"

    tech = technical_score(df_feat)
    rec = final_recommendation(symbol, tech, news_sum, drl_info, ml_info)

    answer_text = (
        f"📌 BÁO CÁO PHÂN TÍCH {symbol} - {analysis_time_text}\n\n"
        f"💰 Giá hiện tại: {features_summary['last_close']:.2f} (cập nhật lúc {data_time_text})\n\n"
        "📈 Phân tích kỹ thuật:\n"
        f"- RSI: {features_summary['rsi_14']:.2f}\n"
        f"- MACD Histogram: {features_summary['macd_hist']:.4f}\n"
        f"- ATR: {features_summary['atr_14']:.4f}\n"
        f"- {tech['trend_note']}\n\n"
        # f"📰 Tin tức & sentiment: {sentiment_label}\n"
        # f"- Tin tích cực: {news_counts['positive']}\n"
        # f"- Tin trung lập: {news_counts['neutral']}\n"
        # f"- Tin tiêu cực: {news_counts['negative']}\n\n"
        f"📰 Tin tức & sentiment (tính năng đang được phát triển)\n\n"
        f"🤖 Tín hiệu mô hình: {rec['model_note']}\n\n"
        f"✅ Khuyến nghị: {rec['recommendation']} (score = {rec['score_percent']:+.2f}%, confidence ≈ {rec['confidence_percent']:.2f}%)\n\n"
        "⚠️ Lưu ý: Đây chỉ là khuyến nghị mang tính tham khảo. Quyết định cuối cùng ở Nhà đầu tư."
    )

    # Lưu analysis context để dùng cho RAG sau này
    analysis_context = {
        "symbol": symbol,
        "indicators": features_summary,
        # "df_feat": df_feat,  # Store for potential future use
        # "drl_info": drl_info,
        "ml_info": ml_info,
        "sentiment_info": sentiment_info,
        "news_items": news_items,
        "technical_score": tech,
        "recommendation": rec,
        "news_summary": news_sum,
        "timestamp": datetime.now().isoformat(),
        "analysis_time": analysis_time_vn.isoformat() if analysis_time_vn else None,
        "data_timestamp": data_time_vn.isoformat() if data_time_vn else None,
    }
    # Store in user_data for this chat
    context.user_data[f"last_analysis_{symbol}"] = analysis_context
    context.user_data["last_symbol"] = symbol

    # 7. Gửi khuyến nghị + nút chọn thêm
    keyboard = [
        [
            InlineKeyboardButton("📈 Biểu đồ & Chỉ báo", callback_data=f"chart_full|{symbol}")
        ],
        [
            InlineKeyboardButton("📰 Tin tức mới nhất", callback_data=f"news|{symbol}")
        ],
        [
            InlineKeyboardButton("💡 Giải thích khuyến nghị", callback_data=f"explain|{symbol}")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await message.reply_text(answer_text, reply_markup=reply_markup)
    log_bot_message(context.user_data, answer_text, symbol)


# ---------- Callback cho các nút Inline ----------
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    action, symbol = data.split("|")

    symbol = normalize_symbol(symbol)

    if action == "explain":
        # Handle explanation request using RAG
        analysis_key = f"last_analysis_{symbol}"
        analysis_context = context.user_data.get(analysis_key)
        
        if not analysis_context:
            await query.message.reply_text(
                f"Tôi chưa có dữ liệu phân tích cho {symbol}.\n"
                f"Hãy chạy phân tích trước: /analyze {symbol}"
            )
            return
        
        if not rag_engine:
            await query.message.reply_text(
                "❌ RAG engine chưa sẵn sàng.\n\n"
                "Vui lòng kiểm tra:\n"
                "• Đã đăng nhập: gcloud auth application-default login\n"
                "• Đã cài: pip install google-generativeai\n\n"
                "Bot vẫn hoạt động bình thường, chỉ thiếu phần giải thích bằng AI."
            )
            return
        
        # Show thinking message
        thinking_msg = await query.message.reply_text("🤔 Đang phân tích và tạo giải thích chi tiết...")
        
        try:
            # Default question for explanation button
            default_question = f"Giải thích chi tiết lý do khuyến nghị {analysis_context.get('recommendation', {}).get('recommendation', 'HOLD')} cho {symbol}. Phân tích các yếu tố kỹ thuật, sentiment, và tín hiệu từ mô hình AI."
            
            rag_history = get_recent_history(context.user_data)
            rag_context = build_rag_context(analysis_context, rag_history)
            answer = rag_engine.answer(default_question, rag_context)
            
            # Delete thinking message and send answer
            await thinking_msg.delete()
            reply_text, reply_parse_mode = format_explanation_reply(symbol, answer, analysis_context)
            await query.message.reply_text(reply_text, parse_mode=reply_parse_mode)
        except Exception as e:
            logger.error(f"RAG explanation error: {e}")
            await thinking_msg.delete()
            await query.message.reply_text(
                f"❌ Có lỗi khi tạo giải thích.\n"
                f"Vui lòng thử lại sau."
            )
        return

    if action == "chart_full":
        if not data_provider:
            await query.message.reply_text("❌ Data provider chưa được cấu hình.")
            return
        
        try:
            # Lấy lại dữ liệu (hoặc cache lại để không gọi API nhiều)
            df_raw = data_provider.get_daily_ohlcv(symbol, days=DATA_LOOKBACK_DAYS)
            if df_raw.empty:
                await query.message.reply_text(f"❌ Không tìm thấy dữ liệu cho {symbol}.")
                return
            
            df_feat = compute_indicators(df_raw)

            # tạo html và gửi file
            html_path = generate_candlestick_html(df_feat, symbol)
            with open(html_path, "rb") as f:
                await query.message.reply_document(
                    document=InputFile(f, filename=os.path.basename(html_path)),
                    caption=f"Biểu đồ nến {symbol} (HTML, mở bằng trình duyệt)."
                )
        except ValueError as e:
            logger.error(f"Error generating chart for {symbol}: {e}")
            await query.message.reply_text(
                f"❌ Không thể lấy dữ liệu cho {symbol}.\n"
                "Vui lòng kiểm tra lại mã coin và thử lại."
            )
        except Exception as e:
            logger.error(f"Error generating chart for {symbol}: {e}")
            await query.message.reply_text("❌ Không thể tạo biểu đồ. Vui lòng thử lại sau.")

    elif action == "news":
        try:
            news_items = news_provider.fetch_news(symbol)
            if not news_items:
                await query.message.reply_text(
                    f"📰 Không có tin tức đáng chú ý cho {symbol} trong thời gian gần đây."
                )
                return

            lines = [f"📰 **Tin tức liên quan đến {symbol}:**\n"]

            for n in news_items[:5]:
                blob = TextBlob(n["title"])
                pol = blob.sentiment.polarity

                if pol > 0.15:
                    tag = "🟢 **TÍCH CỰC**"
                elif pol < -0.15:
                    tag = "🔴 **TIÊU CỰC**"
                else:
                    tag = "🟡 **TRUNG LẬP**"

                line = f"{tag}\n• {n['title']}"
                if n.get("url"):
                    line += f"\n🔗 <a href='{n['url']}'>Nhấn vào đây</a>"
                lines.append(line)

            await query.message.reply_text(
                "\n\n".join(lines),
                parse_mode="HTML",
                disable_web_page_preview=True
            )

        except Exception as e:
            logger.warning(f"News error: {e}")
            await query.message.reply_text(
                "⚠️ Không thể tải tin tức lúc này. Vui lòng thử lại sau."
            )


# ---------- Main ----------
def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Bạn cần set TELEGRAM_BOT_TOKEN trong biến môi trường.")

    keep_alive()
    
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Set up bot commands menu (shows when user presses "/")
    # This will be set when bot starts
    async def post_init(application: Application):
        commands = [
            BotCommand("start", "Bắt đầu sử dụng bot"),
            BotCommand("help", "Hướng dẫn sử dụng"),
            BotCommand("analyze", "Phân tích coin (VD: /analyze BTC)"),
            BotCommand("about", "Giới thiệu về bot"),
            BotCommand("renew", "Làm mới chat (xóa lịch sử)"),
        ]
        await application.bot.set_my_commands(commands)
    
    app.post_init = post_init

    # Register command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("analyze", analyze_command))
    app.add_handler(CommandHandler("about", about_command))
    app.add_handler(CommandHandler("renew", renew_command))

    app.add_handler(CallbackQueryHandler(button_handler))

    # Handler for showing commands menu when user types "/"
    app.add_handler(MessageHandler(filters.TEXT & filters.Regex("^/$"), show_commands_menu))

    # Text handler cho "phân tích BTC", "định giá ETH"...
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_text))

    logger.info("Bot is starting...")
    app.run_polling()


if __name__ == "__main__":
    main()
