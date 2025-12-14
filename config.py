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

# ===== CRYPTO DATA API =====
# Ví dụ CoinMarketCap
CMC_API_KEY = _clean_env(os.getenv("CMC_API_KEY"))
CMC_BASE_URL = "https://pro-api.coinmarketcap.com/v1"

# ===== NEWS / SENTIMENT API =====
# Ví dụ CryptoPanic
CRYPTOPANIC_API_KEY = _clean_env(os.getenv("CRYPTOPANIC_API_KEY"))
CRYPTOPANIC_BASE_URL = "https://cryptopanic.com/api/v1/posts/"

# (hoặc CryptoNews-API)
CRYPTONEWS_API_KEY = _clean_env(os.getenv("CRYPTONEWS_API_KEY"))
CRYPTO_NEWS_BASE_URL = "https://cryptonews-api.com/api/v1"


# ===== LLM / RAG (Gemini via Vertex AI) =====
GEMINI_API_KEY = _clean_env(os.getenv("GEMINI_API_KEY"))
GOOGLE_PROJECT_ID = _clean_env(os.getenv("GOOGLE_PROJECT_ID"))
GOOGLE_LOCATION = _clean_env(os.getenv("GOOGLE_LOCATION")) or "us-central1"

# ===== Reply Formatting =====
EXPLANATION_REPLY_TEMPLATE = (
    _clean_env(os.getenv("EXPLANATION_REPLY_TEMPLATE"))
    or "💡 Giải thích khuyến nghị cho {symbol}:\n\n{answer}"
)
EXPLANATION_PARSE_MODE = (_clean_env(os.getenv("EXPLANATION_PARSE_MODE")) or "PLAIN").upper()

# ===== MODEL PATHS =====
PPO_BTC_MODEL_PATH = "models/drl_ppo_btc.zip"
XGB_DIRECTION_MODEL_PATH = "models/xgb_direction.bin"

# ===== OTHER SETTINGS =====
WINDOW_SIZE = 60   # số ngày đưa vào state DRL
DATA_LOOKBACK_DAYS = 365 * 3  # dữ liệu 3 năm nếu cần
CHART_FOLDER = "charts"

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
6. Phải nêu rõ dữ kiện nào đến từ mô hình DRL, ML, chỉ báo kỹ thuật, hay sentiment.
7. Không đưa lời khuyên đầu tư mang tính bắt buộc - chỉ phân tích và gợi ý.

PHONG CÁCH TRẢ LỜI:
- Chuyên nghiệp nhưng dễ hiểu
- Có cấu trúc rõ ràng (đánh số hoặc bullet points)
- Kết hợp phân tích kỹ thuật với sentiment và mô hình AI
- Luôn kết thúc bằng lưu ý về rủi ro
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
