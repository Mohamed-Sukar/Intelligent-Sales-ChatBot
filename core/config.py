import logging
import os
import re
from typing import Dict, Optional, Tuple

from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────────────
#  Environment & secrets
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_secret(name: str) -> Optional[str]:
    """Read a secret from Colab userdata (if running there), else from env."""
    try:
        from google.colab import userdata  # type: ignore
        try:
            value = userdata.get(name)
            if value:
                return value
        except Exception:
            pass
    except ImportError:
        pass
    return os.getenv(name)

# ──────────────────────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_CSV_CANDIDATES = [
    "data/products_clean.csv",
    "products_clean.csv",
    "/content/products_clean.csv",
]
TEXT_COLUMNS = [
    "title", "category", "product_description", "product_specifications",
    "what_customers_said", "currency", "seller_name", "images"
]
NUMERIC_COLUMNS = ["final_price", "initial_price", "discount", "rating", "ratings_count"]

DEFAULT_DB_PATH = "database/chatbot.db"
DEFAULT_FAISS_PATH = "database/faiss.index"
DEFAULT_BM25_PATH = "database/bm25_index.pkl"
RRF_K = 60
INSTALMENT_RATES: Dict[int, float] = {3: 0.10, 6: 0.15, 12: 0.20}
HANDOFF_KEYWORDS: Tuple[str, ...] = (
    "talk to human", "agent", "representative", "مشكلة", "خدمة العملاء",
)
_ARABIC_DIACRITICS = "\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u0659\u065A\u065B\u065C\u065D\u065E\u065F"
_ARABIC_TATWEEL = "\u0640"
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*|[\u0621-\u064A0-9]+")
