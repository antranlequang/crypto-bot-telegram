# config.py

import os
from dotenv import load_dotenv

load_dotenv()


def _clean_env(value: str | None) -> str | None:
    """
    Normalize environment variables so empty strings or literal 'None' are treated as unset.
    """
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped.lower() in {"none", "null"}:
        return None
    return stripped


TELEGRAM_BOT_TOKEN = _clean_env(os.getenv("TELEGRAM_BOT_TOKEN"))
SECONDARY_BOT_TOKEN = _clean_env(os.getenv("SECONDARY_BOT_TOKEN"))  # Bot for monitoring/logging
MONITOR_CHAT_ID = _clean_env(os.getenv("MONITOR_CHAT_ID"))  # Chat ID to send logs to

# ===== CRYPTO DATA API =====
# CoinMarketCap
CMC_API_KEY = _clean_env(os.getenv("CMC_API_KEY"))
CMC_BASE_URL = "https://pro-api.coinmarketcap.com/v1"

# CoinDesk (for OHLCV and on-chain data)
COINDESK_API_KEY = _clean_env(os.getenv("COINDESK_API_KEY"))
COINDESK_BASE_URL = "https://api.coindesk.com/v1"

# ===== NEWS / SENTIMENT API =====
COINMARKETCAL_API_KEY = _clean_env(os.getenv("COINMARKETCAL_API_KEY"))
TWITTER_BEARER_TOKEN = _clean_env(os.getenv("TWITTER_BEARER_TOKEN"))
REDDIT_USER_AGENT = _clean_env(os.getenv("REDDIT_USER_AGENT")) or "crypto-telegram-bot/1.0"
ALPHA_VANTAGE_API_KEY = _clean_env(os.getenv("ALPHA_VANTAGE_API_KEY"))  # For sentiment API
SENTIMENT_PROVIDER = _clean_env(os.getenv("SENTIMENT_PROVIDER")) or "alpha"  # alpha | gnews | off


# ===== LLM / RAG (Gemini via Vertex AI) =====
GEMINI_API_KEY = _clean_env(os.getenv("GEMINI_API_KEY"))
GOOGLE_PROJECT_ID = _clean_env(os.getenv("GOOGLE_PROJECT_ID"))
GOOGLE_LOCATION = _clean_env(os.getenv("GOOGLE_LOCATION")) or "us-central1"

# ===== Macro Data API (fallback) =====
FRED_API_KEY = _clean_env(os.getenv("FRED_API_KEY"))  # tùy chọn, dùng khi yfinance lỗi

# ===== Reply Formatting =====
EXPLANATION_REPLY_TEMPLATE = (
    _clean_env(os.getenv("EXPLANATION_REPLY_TEMPLATE"))
    or "LUẬN ĐIỂM ĐẦU TƯ - {symbol}\n💡 Giải thích khuyến nghị:\n{answer}"
)
EXPLANATION_PARSE_MODE = (_clean_env(os.getenv("EXPLANATION_PARSE_MODE")) or "PLAIN").upper()

# ===== MODEL PATHS =====
PPO_BTC_MODEL_PATH = "models/drl_ppo_btc.zip"
XGB_DIRECTION_MODEL_PATH = "models/xgb_direction.bin"
XGB_DIRECTION_MODEL_WITH_ONCHAIN_PATH = "models/xgb_direction_with_onchain.bin"
XGB_DIRECTION_MODEL_NO_ONCHAIN_PATH = "models/xgb_direction_no_onchain.bin"

# ===== OTHER SETTINGS ======
WINDOW_SIZE = 365  # số ngày đưa vào state DRL
DATA_LOOKBACK_DAYS = 365 * 5  # dùng dữ liệu tối đa 5 năm để huấn luyện mô hình
DISPLAY_NEWS_DAYS = 3  # số ngày tin tức hiển thị trong phần phản hồi
NEWS_LOOKBACK_DAYS = 3  # số ngày tin tức để phân tích sentiment
CHART_FOLDER = "charts"
DATA_EXPORT_FOLDER = "data_exports"  # CSV export folder for data verification
USE_MACRO_DATA = (_clean_env(os.getenv("USE_MACRO_DATA")) or "0").lower() not in {"0", "false", "no"}
USE_RISK_SCORE = (_clean_env(os.getenv("USE_RISK_SCORE")) or "0").lower() not in {"0", "false", "no"}

# ===== RAG PROMPT CONFIGURATION =====
RAG_SYSTEM_PROMPT = """
Bạn là chuyên gia phân tích crypto chuyên nghiệp với nhiều năm kinh nghiệm.
Nhiệm vụ của bạn là phân tích dữ liệu từ các mô hình AI và đưa ra giải thích chi tiết, dễ hiểu.

QUY TẮC QUAN TRỌNG:
1. CHỈ sử dụng dữ liệu có trong CONTEXT. KHÔNG được bịa đặt thông tin.
2. TẤT CẢ câu trả lời PHẢI viết bằng TIẾNG VIỆT (trừ thuật ngữ chuyên môn như RSI, MACD, DRL, ML, BUY, SELL, HOLD).
3. Giải thích rõ ràng, logic, dựa trên dữ liệu thực tế.
4. Tránh ngôn ngữ chắc chắn tuyệt đối - luôn nhắc nhở về rủi ro.
5. Nếu dữ liệu mâu thuẫn, hãy phân tích sự khác biệt và đưa ra khuyến nghị thận trọng.
6. Chỉ mô tả theo nhóm: thị trường, dự báo mô hình, tin tức.
7. Không đưa lời khuyên đầu tư mang tính bắt buộc - chỉ phân tích và gợi ý.
8. Không đề cập dữ liệu on-chain trong phần giải thích.
9. Tập trung vào vĩ mô, chỉ báo kỹ thuật, tin tức thị trường và kết quả mô hình.

PHONG CÁCH TRẢ LỜI:
- Trang trọng, ngắn gọn, mỗi ý một dòng
- Không nhắc tên cụ thể mô hình hay trọng số
- Nhắc chung theo nhóm: thị trường, dự báo mô hình, tin tức
- Không cần lời chào, đi thẳng vào nội dung
"""

# Keywords để nhận diện câu hỏi về phân tích
ANALYSIS_QUESTION_KEYWORDS = [
    "lý do", "ly do", "reason", "tại sao", "tai sao", "why",
    "phân tích", "phan tich", "analysis", "phân tích chi tiết",
    "giải thích", "giai thich", "explain", "giải thích chi tiết",
    "thesis", "luận điểm", "luan diem", "khuyến nghị", "khuyen nghi",
    "recommendation", "mua", "buy", "bán", "sell", "nên", "nen",
    "đánh giá", "danh gia", "evaluate", "đánh giá chi tiết"
]
