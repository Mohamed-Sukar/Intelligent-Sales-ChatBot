"""
SmartSales Bot — Gradio Deployment
===================================
A modern, fully-styled Gradio chatbot UI wrapping the RAG pipeline from
the Jupyter notebook. Run with:

    python app.py

Requires:
    pip install gradio faiss-cpu sentence-transformers rank-bm25 pandas \
                torch transformers langchain-openai langchain-core \
                requests python-dotenv
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import sys
import unicodedata
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────────────────────────────────────
#  0.  Environment & secrets
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
#  1.  Product catalogue loader
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_CSV_CANDIDATES = [
    "products_clean.csv",
    "data/products_clean.csv",
    "/content/products_clean.csv",
]
TEXT_COLUMNS = [
    "title", "category", "product_description", "product_specifications",
    "what_customers_said", "currency", "seller_name",
]
NUMERIC_COLUMNS = ["final_price", "initial_price", "discount", "rating", "ratings_count"]


def _find_csv_path(explicit_path: Optional[str] = None) -> str:
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path
    for candidate in DEFAULT_CSV_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "products_clean.csv not found. Place it next to this script."
    )


def load_products_from_csv(csv_path: Optional[str] = None) -> List[Dict[str, Any]]:
    path = _find_csv_path(csv_path)
    df = pd.read_csv(path)
    if "is_active" in df.columns:
        df = df[df["is_active"] == 1].copy()
    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["product_id"] = df["product_id"].astype(str)
    products = df.to_dict(orient="records")
    print(f"Loaded {len(products)} active products from {path!r}")
    return products


# ──────────────────────────────────────────────────────────────────────────────
#  2.  Retrieval Engine  (Hybrid FAISS + BM25)
# ──────────────────────────────────────────────────────────────────────────────
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


class SalesRetrievalEngine:
    """Deterministic hybrid retrieval + business logic engine (English/Arabic)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self.faiss_index: Optional[faiss.IndexFlatL2] = None
        self.bm25: Optional[BM25Okapi] = None
        self.products: List[Dict[str, Any]] = []
        self.categories: List[str] = []
        self.rich_texts: List[str] = []
        self._vectors: Optional[np.ndarray] = None
        self._is_built: bool = False

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model '%s' ...", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @staticmethod
    def _normalize(text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFKC", str(text))
        text = text.lower()
        text = re.sub(f"[{_ARABIC_DIACRITICS}{_ARABIC_TATWEEL}]", "", text)
        text = text.replace("\u0623", "\u0627").replace("\u0625", "\u0627").replace("\u0622", "\u0627")
        text = text.replace("\u0629", "\u0647")
        text = text.replace("\u0649", "\u064A")
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        return _TOKEN_RE.findall(cls._normalize(text))

    @staticmethod
    def _build_rich_text(product: Dict[str, Any]) -> str:
        parts = [
            str(product.get("title", "")),
            str(product.get("category", "")),
            str(product.get("product_description", "")),
            str(product.get("product_specifications", "")),
        ]
        return " ".join(p for p in parts if p).strip()

    def _require_built(self) -> None:
        if not self._is_built:
            raise RuntimeError("Indexes are not built yet. Call build_indexes() first.")

    def _require_query(self, query: str) -> str:
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        return query.strip()

    def build_indexes(
        self,
        products: List[Dict[str, Any]],
        faiss_path: str = DEFAULT_FAISS_PATH,
        bm25_path: str = DEFAULT_BM25_PATH,
    ) -> Tuple[str, str]:
        if not products:
            raise ValueError("Cannot build indexes from an empty product list.")
        for path in (faiss_path, bm25_path):
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
        self.products = list(products)
        self.rich_texts = [self._build_rich_text(p) for p in self.products]
        self.categories = sorted({
            str(p.get("category", "")).strip()
            for p in self.products
            if str(p.get("category", "")).strip()
        })
        model = self._get_model()
        vectors = model.encode(
            self.rich_texts,
            normalize_embeddings=False,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        dim = self._vectors.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dim)
        self.faiss_index.add(self._vectors)
        faiss.write_index(self.faiss_index, faiss_path)
        tokenized = [self._tokenize(t) for t in self.rich_texts]
        self.bm25 = BM25Okapi(tokenized)
        with open(bm25_path, "wb") as fh:
            pickle.dump(self.bm25, fh)
        self._is_built = True
        return faiss_path, bm25_path

    @staticmethod
    def _rrf_add(score_map: Dict[int, float], positions: List[int]) -> None:
        for rank, pos in enumerate(positions):
            score_map[pos] = score_map.get(pos, 0.0) + 1.0 / (rank + RRF_K)

    def _dense_search_positions(self, query_vec: np.ndarray, top_n: int) -> List[int]:
        assert self.faiss_index is not None
        k = min(top_n, self.faiss_index.ntotal)
        if k <= 0:
            return []
        _d, idx = self.faiss_index.search(query_vec, k)
        return [int(i) for i in idx[0] if i != -1]

    def _sparse_search_positions(self, query_tokens: List[str], top_n: int) -> List[int]:
        assert self.bm25 is not None
        scores = np.asarray(self.bm25.get_scores(query_tokens))
        if scores.size == 0:
            return []
        k = min(top_n, scores.size)
        part = np.argpartition(-scores, k - 1)[:k]
        order = part[np.argsort(-scores[part], kind="stable")]
        return [int(i) for i in order]

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        self._require_built()
        query = self._require_query(query)
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        dense_query_vec = self._get_model().encode(
            [self._normalize(query)],
            normalize_embeddings=False,
            convert_to_numpy=True,
        )
        dense_top15 = self._dense_search_positions(dense_query_vec, 15)
        sparse_top15 = self._sparse_search_positions(self._tokenize(query), 15)
        score_map: Dict[int, float] = {}
        self._rrf_add(score_map, dense_top15)
        self._rrf_add(score_map, sparse_top15)
        ranked = sorted(score_map.items(), key=lambda kv: (-kv[1], kv[0]))
        return [dict(self.products[pos]) for pos, _score in ranked[:top_k]]

    def get_installment_terms(self) -> Dict[int, float]:
        return dict(INSTALMENT_RATES)

    def get_all_categories(self) -> List[str]:
        self._require_built()
        return list(self.categories)

    def keyword_lookup(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        self._require_built()
        norm_query = self._normalize(query)
        if not norm_query:
            return []
        matches: List[int] = []
        for pos, product in enumerate(self.products):
            norm_title = self._normalize(str(product.get("title", "")))
            norm_id = self._normalize(str(product.get("product_id", "")))
            if norm_title and (norm_title in norm_query or norm_query in norm_title):
                matches.append(pos)
            elif norm_id and norm_id in norm_query:
                matches.append(pos)
        return [dict(self.products[pos]) for pos in matches[:limit]]

    def get_recommendations(self, product_id: str, top_k: int = 3) -> List[Dict[str, Any]]:
        self._require_built()
        try:
            origin_pos = next(
                i for i, p in enumerate(self.products) if str(p.get("product_id")) == str(product_id)
            )
        except StopIteration:
            raise LookupError(f"Product id '{product_id}' not found in corpus.")
        original_price = float(self.products[origin_pos].get("final_price", 0.0))
        if original_price <= 0:
            raise ValueError(f"Product '{product_id}' has no valid price.")
        query_vec = np.ascontiguousarray(
            self._get_model().encode(
                [self.rich_texts[origin_pos]],
                normalize_embeddings=False,
                convert_to_numpy=True,
            )
        )
        pool = self._dense_search_positions(query_vec, 20)
        if not pool:
            return []
        candidates = []
        for pos in pool:
            if pos == origin_pos:
                continue
            price = float(self.products[pos].get("final_price", 0.0))
            if price <= 0:
                continue
            if not (0.7 <= price / original_price <= 1.3):
                continue
            candidates.append((pos, price))
        candidates.sort(
            key=lambda t: (
                -float(self.products[t[0]].get("rating", 0.0)),
                -float(self.products[t[0]].get("ratings_count", 0.0)),
                t[0],
            )
        )
        return [dict(self.products[pos]) for pos, _price in candidates[:top_k]]

    def calculate_installment(self, price: float, months: int = 6) -> Dict[str, Any]:
        if price <= 0:
            raise ValueError("price must be positive.")
        if months not in INSTALMENT_RATES:
            raise ValueError(f"Unsupported instalment months {months!r}; choose from {sorted(INSTALMENT_RATES)}.")
        rate = INSTALMENT_RATES[months]
        total = price * (1.0 + rate)
        monthly = total / months
        return {
            "monthly_payment": round(monthly, 2),
            "total_with_interest": round(total, 2),
            "months": months,
        }

    def should_handoff(self, sentiment_score: float, user_text: str) -> bool:
        if sentiment_score < -0.5:
            return True
        lowered = (user_text or "").lower()
        return any(kw in lowered for kw in HANDOFF_KEYWORDS)


# ──────────────────────────────────────────────────────────────────────────────
#  3.  LLM Manager  (OpenRouter → Gemini → static fallback)
# ──────────────────────────────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import requests


class LLMManager:
    FALLBACK_MESSAGE = (
        "I'm sorry, I'm having temporary technical issues. "
        "Please try again in a moment."
    )

    def __init__(
        self,
        openrouter_model: str = None,
        gemini_model: str = "gemini-2.5-flash",
        temperature: float = 0.2,
        timeout: int = 10,
    ) -> None:
        self.openrouter_model = openrouter_model or os.getenv(
            "OPENROUTER_MODEL", "cohere/north-mini-code:free"
        )
        self.gemini_model = gemini_model
        self.gemini_key = get_secret("GEMINI_API_KEY")
        self.gemini_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.gemini_model}:generateContent"
        )
        self.primary_llm = ChatOpenAI(
            model=self.openrouter_model,
            openai_api_key=get_secret("OPEN_ROUTER_KEY") or "not-set",
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=temperature,
            max_retries=1,
            timeout=timeout,
        )
        self.usage_log: List[Dict[str, str]] = []

    def chat(self, system_prompt: str, user_prompt: str) -> Dict[str, str]:
        try:
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            response = self.primary_llm.invoke(messages)
            self.usage_log.append({"source": "openrouter", "status": "success"})
            return {"content": response.content, "source": "openrouter", "status": "success"}
        except Exception as e:
            print(f"[LLMManager] OpenRouter failed: {e}")
            try:
                text = self._call_gemini(system_prompt, user_prompt)
                self.usage_log.append({"source": "gemini_backup", "status": "fallback"})
                return {"content": text, "source": "gemini_backup", "status": "fallback"}
            except Exception as e2:
                print(f"[LLMManager] Gemini also failed: {e2}")
                self.usage_log.append({"source": "none", "status": "failed"})
                return {"content": self.FALLBACK_MESSAGE, "source": "none", "status": "failed"}

    def _call_gemini(self, system_prompt: str, user_prompt: str) -> str:
        if not self.gemini_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        url = f"{self.gemini_url}?key={self.gemini_key}"
        payload = {
            "contents": [{"parts": [{"text": f"{system_prompt}\n\nUser: {user_prompt}"}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 500},
        }
        r = requests.post(url, json=payload, timeout=15)
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


# ──────────────────────────────────────────────────────────────────────────────
#  4.  Prompt Builder
# ──────────────────────────────────────────────────────────────────────────────
class PromptBuilder:
    SYSTEM_PROMPT = """You are SmartSales Bot, a friendly, professional sales assistant for an e-commerce store.

Rules:
1. Use ONLY the provided product data for product facts, availability, prices, and ratings. Never invent them.
2. If the provided data is insufficient, say "I don't have that information."
3. Keep responses concise, helpful, and professional.
4. Include prices and ratings when mentioning products.
5. When the user describes a support problem, acknowledge it clearly and offer appropriate next steps.
6. Show a supplied installment plan accurately; do not invent plan terms. You may also state the store's general installment durations/rates (given below) even when no specific dollar plan has been computed for a product.
7. Recommend related products only when they are supported by the retrieved product data.
8. Reply in the same language the user wrote in (Arabic or English)."""

    def build_prompt(
        self,
        user_message: str,
        products_found: List[Dict[str, Any]],
        viewed_products: List[Dict[str, Any]],
        chat_history: List[Dict[str, str]],
        installment_info: Optional[Dict[str, Any]] = None,
        all_categories: Optional[List[str]] = None,
        installment_terms: Optional[Dict[int, float]] = None,
    ) -> str:
        return f"""
===============================================
USER MESSAGE: {user_message}
===============================================

POTENTIALLY RELEVANT PRODUCTS FROM SEARCH:
===============================================
{self._format_products(products_found)}

===============================================
FULL CATALOGUE CATEGORY LIST (only when supplied — this is EVERY category
in the store, not just what was retrieved above; use it, and only it, when
the user asks what categories/kinds of products the store carries):
===============================================
{self._format_categories(all_categories)}

===============================================
GENERAL INSTALLMENT TERMS (store policy — always safe to mention, even
without a specific computed plan):
===============================================
{self._format_installment_terms(installment_terms)}

===============================================
INSTALLMENT PLAN COMPUTED FOR A SPECIFIC PRODUCT (only when supplied):
===============================================
{self._format_installment(installment_info)}

===============================================
PRODUCTS USER ALREADY VIEWED:
===============================================
{self._format_viewed(viewed_products)}

===============================================
CONVERSATION HISTORY (last 5 messages):
===============================================
{self._format_history(chat_history)}

===============================================
INSTRUCTIONS:
===============================================
Answer the user's message directly. Use a retrieved product only when it is relevant to the
question. State exact prices and ratings only from the supplied product data. If an installment
plan is supplied, show its terms accurately. If product data does not answer the question, say
what information is unavailable rather than making up a product fact.
"""

    def _format_products(self, products: List[Dict[str, Any]]) -> str:
        if not products:
            return "No products found."
        lines = []
        for i, p in enumerate(products, 1):
            price = p.get("final_price", "N/A")
            initial = p.get("initial_price")
            discount = p.get("discount")
            price_line = f"   Price: ${price}"
            if initial and discount:
                price_line += f" (was ${initial}, {discount}% off)"
            lines.append(
                f"{i}. {p.get('title', 'Unknown product')}\n"
                f"{price_line}\n"
                f"   Rating: {p.get('rating', 'N/A')}/5 ({p.get('ratings_count', 0)} reviews)\n"
                f"   Category: {p.get('category', 'N/A')}\n"
                f"   Customers said: {p.get('what_customers_said', 'N/A')}"
            )
        return "\n\n".join(lines)

    def _format_categories(self, categories: Optional[List[str]]) -> str:
        if not categories:
            return "Not requested for this message."
        return ", ".join(categories)

    def _format_installment_terms(self, terms: Optional[Dict[int, float]]) -> str:
        if not terms:
            return "Not available."
        lines = [f"- {months} months at {rate * 100:.0f}% interest" for months, rate in sorted(terms.items())]
        return "\n".join(lines)

    def _format_installment(self, installment_info: Optional[Dict[str, Any]]) -> str:
        if not installment_info:
            return "No installment plan was supplied."
        return (
            f"{installment_info['months']} months -> "
            f"${installment_info['monthly_payment']}/month "
            f"(total ${installment_info['total_with_interest']})"
        )

    def _format_viewed(self, viewed: List[Dict[str, Any]]) -> str:
        if not viewed:
            return "User hasn't viewed any products yet."
        return ", ".join(p.get("title", "Unknown") for p in viewed)

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        if not history:
            return "This is the start of the conversation."
        lines = []
        for msg in history[-5:]:
            role = "User" if msg.get("role") == "user" else "Bot"
            lines.append(f"{role}: {msg.get('content', '')}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
#  5.  Conversation Memory
# ──────────────────────────────────────────────────────────────────────────────
class ConversationMemory:
    def __init__(self) -> None:
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def _session(self, session_id: str) -> Dict[str, Any]:
        return self._sessions.setdefault(session_id, {"messages": [], "viewed_products": []})

    def save_message(self, session_id: str, role: str, content: str) -> None:
        self._session(session_id)["messages"].append({"role": role, "content": content})

    def get_last_n_messages(self, session_id: str, n: int = 5) -> List[Dict[str, str]]:
        return self._session(session_id)["messages"][-n:]

    def get_all_messages(self, session_id: str) -> List[Dict[str, str]]:
        return self._session(session_id)["messages"]

    def add_viewed_products(self, session_id: str, products: List[Dict[str, Any]]) -> None:
        seen_ids = {p.get("product_id") for p in self._session(session_id)["viewed_products"]}
        for p in products:
            if p.get("product_id") not in seen_ids:
                self._session(session_id)["viewed_products"].append(p)
                seen_ids.add(p.get("product_id"))

    def get_viewed_products(self, session_id: str) -> List[Dict[str, Any]]:
        return self._session(session_id)["viewed_products"]

    def get_last_viewed_product(self, session_id: str) -> Optional[Dict[str, Any]]:
        viewed = self._session(session_id)["viewed_products"]
        return viewed[-1] if viewed else None


# ──────────────────────────────────────────────────────────────────────────────
#  6.  Human Handoff Policy
# ──────────────────────────────────────────────────────────────────────────────
class HumanHandoffPolicy:
    _EXPLICIT_REQUEST_PHRASES = (
        "talk to a human", "speak to a human", "talk to a person",
        "speak to a person", "human agent", "live agent",
        "human representative", "customer representative",
        "contact support", "customer service",
        "\u0645\u0648\u0638\u0641 \u062e\u062f\u0645\u0629 \u0627\u0644\u0639\u0645\u0644\u0627\u0621",
        "\u062f\u0639\u0645 \u0628\u0634\u0631\u064a",
        "\u0627\u062a\u0643\u0644\u0645 \u0645\u0639 \u062d\u062f",
        "\u0627\u062a\u0643\u0644\u0645 \u0645\u0639 \u0634\u062e\u0635",
        "\u0627\u0643\u0644\u0645 \u062d\u062f",
        "\u0623\u0643\u0644\u0645 \u062d\u062f",
        "\u0623\u062a\u0643\u0644\u0645 \u0645\u0639 \u062d\u062f",
        "\u0645\u0648\u0638\u0641",
    )

    def evaluate(self, user_message: str, request_handoff: bool = False) -> Optional[str]:
        if request_handoff:
            return "frontend_request"
        normalized = " ".join((user_message or "").casefold().split())
        if any(phrase in normalized for phrase in self._EXPLICIT_REQUEST_PHRASES):
            return "explicit_customer_request"
        return None


# ──────────────────────────────────────────────────────────────────────────────
#  7.  RAG Pipeline
# ──────────────────────────────────────────────────────────────────────────────
class RAGPipeline:
    _CATEGORY_INTENT_RE = re.compile(
        r"(what|which).{0,20}\bcategor(y|ies)\b"
        r"|\blist\b.{0,20}\bcategor(y|ies)\b"
        r"|\bcategories\b.{0,20}\b(you|do you|available)\b"
        r"|\bwhat.{0,15}(kinds?|types?).{0,10}(of )?products?\b"
        r"|\u0627\u0644\u0641\u0626\u0627\u062a\b|\u0627\u0644\u0623\u0642\u0633\u0627\u0645\b"
        r"|\u0623\u0646\u0648\u0627\u0639\s+\u0627\u0644\u0645\u0646\u062a\u062c\u0627\u062a",
        re.IGNORECASE,
    )

    def __init__(
        self,
        engine: SalesRetrievalEngine,
        llm_manager: Optional[LLMManager] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        memory: Optional[ConversationMemory] = None,
        handoff_policy: Optional[HumanHandoffPolicy] = None,
    ) -> None:
        self.engine = engine
        self.llm = llm_manager or LLMManager()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.memory = memory or ConversationMemory()
        self.handoff_policy = handoff_policy or HumanHandoffPolicy()

    def process_message(
        self,
        user_message: str,
        session_id: str,
        request_handoff: bool = False,
        installment_months: Optional[int] = None,
    ) -> Dict[str, Any]:
        handoff_reason = self.handoff_policy.evaluate(user_message, request_handoff)
        self.memory.save_message(session_id, "user", user_message)

        if handoff_reason:
            summary = self._generate_handoff_summary(session_id, handoff_reason)
            return {
                "type": "handoff",
                "message": "I\u2019ll connect you with a human representative.",
                "summary_for_agent": summary,
                "handoff_reason": handoff_reason,
            }

        products = self._retrieve_products(user_message)
        installment_info = self._build_installment_plan(session_id, installment_months)
        installment_terms = self.engine.get_installment_terms()
        all_categories = (
            self.engine.get_all_categories()
            if self._CATEGORY_INTENT_RE.search(user_message or "")
            else None
        )

        chat_history = self.memory.get_last_n_messages(session_id, n=5)
        viewed_products = self.memory.get_viewed_products(session_id)
        prompt = self.prompt_builder.build_prompt(
            user_message=user_message,
            products_found=products,
            viewed_products=viewed_products,
            chat_history=chat_history,
            installment_info=installment_info,
            all_categories=all_categories,
            installment_terms=installment_terms,
        )

        llm_response = self.llm.chat(
            system_prompt=self.prompt_builder.SYSTEM_PROMPT,
            user_prompt=prompt,
        )

        surfaced_products = self._filter_surfaced_products(products, llm_response["content"])

        self.memory.save_message(session_id, "bot", llm_response["content"])
        if surfaced_products:
            self.memory.add_viewed_products(session_id, surfaced_products[:1])

        return {
            "type": "response",
            "message": llm_response["content"],
            "products": surfaced_products,
            "installment": installment_info,
            "llm_source": llm_response["source"],
        }

    def _filter_surfaced_products(
        self, products: List[Dict[str, Any]], llm_content: str
    ) -> List[Dict[str, Any]]:
        """Filter retrieved products to only those actually mentioned in the LLM text."""
        if not products or not llm_content:
            return []
        surfaced = []
        for p in products:
            title = p.get("title", "").strip()
            if not title:
                continue

            words = title.split()
            first_word = words[0] if words else ""

            matched = False
            # Check 1: First word (e.g. brand name like "BLAUPUNKT", "CrossBeats", "NOISE") as a whole word in LLM output
            if len(first_word) >= 3:
                pattern = r"\b" + re.escape(first_word) + r"\b"
                if re.search(pattern, llm_content, re.IGNORECASE):
                    matched = True

            # Check 2: First two words (e.g. "Lenovo Legion")
            if not matched and len(words) >= 2:
                two_words = " ".join(words[:2])
                if len(two_words) >= 4:
                    pattern = r"\b" + re.escape(two_words) + r"\b"
                    if re.search(pattern, llm_content, re.IGNORECASE):
                        matched = True

            if matched:
                surfaced.append(p)

        return surfaced

    def _retrieve_products(self, user_message: str) -> List[Dict[str, Any]]:
        text = (user_message or "").strip()
        if not text:
            return []
        try:
            results = self.engine.hybrid_search(text, top_k=5)
        except Exception as error:
            print(f"[RAGPipeline] Retrieval failed: {error}")
            results = []
        try:
            exact_matches = self.engine.keyword_lookup(text, limit=3)
        except Exception as error:
            print(f"[RAGPipeline] Keyword lookup failed: {error}")
            exact_matches = []
        seen_ids = {p.get("product_id") for p in results}
        for product in exact_matches:
            if product.get("product_id") not in seen_ids:
                results.append(product)
                seen_ids.add(product.get("product_id"))
        return results

    def _build_installment_plan(
        self, session_id: str, installment_months: Optional[int]
    ) -> Optional[Dict[str, Any]]:
        if (
            not isinstance(installment_months, int)
            or isinstance(installment_months, bool)
            or installment_months <= 0
        ):
            return None
        viewed = self.memory.get_last_viewed_product(session_id)
        if not viewed or not viewed.get("final_price"):
            return None
        return self.engine.calculate_installment(
            float(viewed["final_price"]), months=installment_months
        )

    def _generate_handoff_summary(self, session_id: str, handoff_reason: str) -> str:
        history = self.memory.get_all_messages(session_id)
        summary_prompt = (
            "Summarize this customer conversation for a human agent. "
            "Include the latest request, products viewed, any unresolved support issue, and the "
            f"handoff reason code ({handoff_reason}). Be concise and factual.\n\n"
            f"Conversation:\n{history}"
        )
        result = self.llm.chat(
            "You are a conversation summarizer. Be concise and factual.",
            summary_prompt,
        )
        return result["content"]


# ──────────────────────────────────────────────────────────────────────────────
#  8.  Build the pipeline (runs at import/startup time)
# ──────────────────────────────────────────────────────────────────────────────
print("⏳ Loading product catalogue and building search indexes …")
products = load_products_from_csv()
engine = SalesRetrievalEngine()
engine.build_indexes(products)
pipeline = RAGPipeline(engine)
print("✅ Pipeline ready — launching Gradio …")


# ──────────────────────────────────────────────────────────────────────────────
#  9.  Gradio UI
# ──────────────────────────────────────────────────────────────────────────────
import gradio as gr

# ── Product card HTML renderer ────────────────────────────────────────────────
def _render_product_cards(products: List[Dict[str, Any]]) -> str:
    """Render product results as styled HTML cards."""
    if not products:
        return ""
    cards = []
    for p in products[:5]:
        title = p.get("title", "Unknown")
        price = p.get("final_price", 0)
        initial = p.get("initial_price", 0)
        discount = p.get("discount", 0)
        rating = float(p.get("rating", 0))
        reviews = int(p.get("ratings_count", 0))
        category = p.get("category", "")
        description = str(p.get("product_description", ""))[:120]
        if description and len(str(p.get("product_description", ""))) > 120:
            description += "…"

        # Star rating visual
        full_stars = int(rating)
        half_star = 1 if (rating - full_stars) >= 0.3 else 0
        empty_stars = 5 - full_stars - half_star
        stars_html = "★" * full_stars + ("½" if half_star else "") + "☆" * empty_stars

        # Discount badge
        discount_html = ""
        if discount and float(discount) > 0:
            discount_html = f'<span class="discount-badge">-{int(float(discount))}%</span>'

        # Price line
        price_html = f'<span class="product-price">${price:.2f}</span>'
        if initial and float(initial) > float(price):
            price_html += f' <span class="product-original-price">${float(initial):.2f}</span>'

        cards.append(f"""
        <div class="product-card">
            <div class="product-card-header">
                <span class="product-category">{category}</span>
                {discount_html}
            </div>
            <h4 class="product-title">{title}</h4>
            <p class="product-desc">{description}</p>
            <div class="product-meta">
                <div class="product-rating">
                    <span class="stars">{stars_html}</span>
                    <span class="review-count">{rating}/5 ({reviews:,})</span>
                </div>
                <div class="product-price-row">{price_html}</div>
            </div>
        </div>
        """)
    return f'<div class="product-cards-grid">{"".join(cards)}</div>'


# ── Custom CSS ────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
/* ─── Root tokens (dark) ──────────────────────────────────────────── */
:root {
    --brand-400: #5ea4f8;
    --brand-500: #3b82f6;
    --brand-600: #60a5fa;
    --brand-700: #93bbfc;

    --surface-0:  #1a1a2e;
    --surface-50: #12121e;
    --surface-100:#1e1e32;
    --surface-200:#2a2a40;
    --surface-300:#3a3a52;

    --text-900: #eaeaf0;
    --text-700: #c8c8d4;
    --text-500: #8888a0;
    --text-400: #6a6a82;

    --success:  #34d399;
    --warning:  #fbbf24;
    --danger:   #f87171;

    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-xl: 20px;

    --shadow-sm: 0 1px 2px rgba(0,0,0,.25);
    --shadow-md: 0 4px 12px rgba(0,0,0,.35);
    --shadow-lg: 0 8px 30px rgba(0,0,0,.45);
}

/* ─── Global overrides ────────────────────────────────────────────── */
body, .gradio-container, .main, .contain {
    background: var(--surface-50) !important;
    color: var(--text-900) !important;
}
.gradio-container {
    max-width: 960px !important;
    margin: 0 auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ─── Header area ─────────────────────────────────────────────────── */
.app-header {
    text-align: center;
    padding: 28px 20px 20px;
}
.app-header h1 {
    font-size: 1.65rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: var(--text-900);
    margin: 0 0 6px;
}
.app-header p {
    font-size: 0.88rem;
    color: var(--text-500);
    margin: 0;
    line-height: 1.5;
}

/* ─── Chatbot bubbles ─────────────────────────────────────────────── */
.chatbot-container .message {
    border-radius: var(--radius-md) !important;
    padding: 14px 18px !important;
    line-height: 1.6 !important;
    font-size: 0.92rem !important;
}

/* ─── Gradio dark overrides ───────────────────────────────────────── */
.block, .wrap, .panel {
    background: var(--surface-0) !important;
    border-color: var(--surface-200) !important;
}
input, textarea, .input-container {
    background: var(--surface-100) !important;
    color: var(--text-900) !important;
    border-color: var(--surface-300) !important;
}
.chatbot, .chatbot > div {
    background: var(--surface-0) !important;
}

/* ─── Quick-action chips ──────────────────────────────────────────── */
.quick-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 4px 0 0;
}
.quick-actions button,
.quick-chip {
    background: var(--surface-100) !important;
    border: 1px solid var(--surface-300) !important;
    border-radius: 40px !important;
    padding: 7px 16px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: var(--text-700) !important;
    cursor: pointer !important;
    transition: all .18s ease !important;
    min-height: unset !important;
    line-height: 1.3 !important;
}
.quick-actions button:hover,
.quick-chip:hover {
    background: var(--surface-200) !important;
    border-color: var(--brand-500) !important;
    color: var(--brand-600) !important;
}

/* ─── Handoff / control buttons ───────────────────────────────────── */
.handoff-btn {
    background: var(--surface-100) !important;
    border: 1px solid var(--surface-300) !important;
    border-radius: var(--radius-md) !important;
    padding: 10px 20px !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    color: var(--text-700) !important;
    transition: all .18s ease !important;
}
.handoff-btn:hover {
    background: var(--surface-200) !important;
    border-color: var(--danger) !important;
    color: var(--danger) !important;
}

/* ─── Product cards ───────────────────────────────────────────────── */
.product-cards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 12px;
    padding: 4px 0;
}
.product-card {
    background: var(--surface-100);
    border: 1px solid var(--surface-200);
    border-radius: var(--radius-md);
    padding: 16px;
    transition: box-shadow .2s ease, border-color .2s ease;
}
.product-card:hover {
    box-shadow: var(--shadow-md);
    border-color: var(--brand-400);
}
.product-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.product-category {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-400);
}
.discount-badge {
    background: rgba(248,113,113,.15);
    color: var(--danger);
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 40px;
}
.product-title {
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--text-900);
    margin: 0 0 6px;
    line-height: 1.35;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.product-desc {
    font-size: 0.78rem;
    color: var(--text-500);
    line-height: 1.45;
    margin: 0 0 10px;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.product-meta {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
}
.product-rating .stars {
    color: var(--warning);
    font-size: 0.82rem;
    letter-spacing: 1px;
}
.product-rating .review-count {
    font-size: 0.72rem;
    color: var(--text-400);
    display: block;
    margin-top: 2px;
}
.product-price {
    font-size: 1rem;
    font-weight: 700;
    color: var(--brand-600);
}
.product-original-price {
    font-size: 0.78rem;
    color: var(--text-400);
    text-decoration: line-through;
    margin-left: 4px;
}

/* ─── Status bar ──────────────────────────────────────────────────── */
.status-bar {
    text-align: center;
    padding: 10px;
    font-size: 0.75rem;
    color: var(--text-400);
}

/* ─── Footer ──────────────────────────────────────────────────────── */
footer { display: none !important; }

/* ─── Scrollbar (dark) ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--surface-100); }
::-webkit-scrollbar-thumb { background: var(--surface-300); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-400); }

/* ─── Responsive ──────────────────────────────────────────────────── */
@media (max-width: 640px) {
    .gradio-container { padding: 8px !important; }
    .product-cards-grid { grid-template-columns: 1fr; }
    .app-header h1 { font-size: 1.3rem; }
}
"""

# ── Gradio app ────────────────────────────────────────────────────────────────
def create_app() -> gr.Blocks:
    with gr.Blocks(
        css=CUSTOM_CSS,
        title="SmartSales Bot — AI Shopping Assistant",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.slate,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
            radius_size=gr.themes.sizes.radius_md,
        ).set(
            body_background_fill="#12121e",
            body_background_fill_dark="#12121e",
            block_background_fill="#1a1a2e",
            block_background_fill_dark="#1a1a2e",
            block_border_color="#2a2a40",
            block_border_color_dark="#2a2a40",
            block_label_text_color="#c8c8d4",
            block_label_text_color_dark="#c8c8d4",
            block_title_text_color="#eaeaf0",
            block_title_text_color_dark="#eaeaf0",
            body_text_color="#eaeaf0",
            body_text_color_dark="#eaeaf0",
            body_text_color_subdued="#8888a0",
            body_text_color_subdued_dark="#8888a0",
            input_background_fill="#1e1e32",
            input_background_fill_dark="#1e1e32",
            input_border_color="#3a3a52",
            input_border_color_dark="#3a3a52",
            button_primary_background_fill="#3b82f6",
            button_primary_background_fill_dark="#3b82f6",
            button_primary_text_color="#ffffff",
            button_primary_text_color_dark="#ffffff",
            button_secondary_background_fill="#1e1e32",
            button_secondary_background_fill_dark="#1e1e32",
            button_secondary_border_color="#3a3a52",
            button_secondary_border_color_dark="#3a3a52",
            button_secondary_text_color="#c8c8d4",
            button_secondary_text_color_dark="#c8c8d4",
        ),
    ) as app:
        # ── State ────────────────────────────────────────────────────────
        session_state = gr.State(value=lambda: str(uuid.uuid4()))

        # ── Header ───────────────────────────────────────────────────────
        gr.HTML("""
        <div class="app-header">
            <h1>SmartSales Bot</h1>
            <p>Your AI-powered shopping assistant. Ask about products, prices,<br>
            installment plans, or categories — in English or Arabic.</p>
        </div>
        """)

        # ── Chat area ────────────────────────────────────────────────────
        chatbot = gr.Chatbot(
            value=[],
            elem_classes=["chatbot-container"],
            height=380,
            show_label=False,
            show_copy_button=True,
            avatar_images=(None, None),
            type="messages",
            placeholder="<p style='text-align:center;color:#6a6a82;padding:30px 20px;font-size:0.92rem;'>Ask me anything about our products...</p>",
        )

        # ── Product cards display ────────────────────────────────────────
        product_display = gr.HTML(value="", visible=False)

        # ── Quick action buttons ─────────────────────────────────────────
        with gr.Row(elem_classes=["quick-actions"]):
            qa_trending = gr.Button("Trending products", elem_classes=["quick-chip"], size="sm")
            qa_cheap    = gr.Button("Budget-friendly picks", elem_classes=["quick-chip"], size="sm")
            qa_cats     = gr.Button("Browse categories", elem_classes=["quick-chip"], size="sm")
            qa_install  = gr.Button("Installment plans", elem_classes=["quick-chip"], size="sm")

        # ── Text input + send ────────────────────────────────────────────
        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Type your message…",
                show_label=False,
                container=False,
                scale=7,
                autofocus=True,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)

        # ── Bottom controls ──────────────────────────────────────────────
        with gr.Row():
            handoff_btn = gr.Button("Contact a Representative", elem_classes=["handoff-btn"], size="sm")
            clear_btn   = gr.Button("Clear Chat", elem_classes=["handoff-btn"], size="sm")

        # ── Status bar ───────────────────────────────────────────────────
        status_html = gr.HTML('<div class="status-bar">Powered by Hybrid RAG — FAISS + BM25 · Dual-API LLM (OpenRouter / Gemini)</div>')

        # ══════════════════════════════════════════════════════════════════
        #  Event handlers
        # ══════════════════════════════════════════════════════════════════
        def respond(user_message: str, chat_history: list, session_id: str):
            if not user_message or not user_message.strip():
                return "", chat_history, gr.update(visible=False, value=""), session_id

            chat_history = chat_history + [{"role": "user", "content": user_message}]

            result = pipeline.process_message(user_message, session_id=session_id)

            bot_reply = result["message"]
            chat_history = chat_history + [{"role": "assistant", "content": bot_reply}]

            # Product cards
            products_html = ""
            show_products = False
            if result.get("products"):
                products_html = _render_product_cards(result["products"])
                show_products = True

            return "", chat_history, gr.update(visible=show_products, value=products_html), session_id

        def handle_handoff(chat_history: list, session_id: str):
            chat_history = chat_history + [{"role": "user", "content": "[Contact a Representative]"}]

            result = pipeline.process_message(
                "I want to talk to a human representative",
                session_id=session_id,
                request_handoff=True,
            )
            bot_reply = result["message"]
            if result.get("summary_for_agent"):
                bot_reply += f"\n\n*Agent briefing prepared.*"

            chat_history = chat_history + [{"role": "assistant", "content": bot_reply}]
            return chat_history, gr.update(visible=False, value=""), session_id

        def send_quick_action(action_text: str, chat_history: list, session_id: str):
            return respond(action_text, chat_history, session_id)

        def clear_chat():
            new_session = str(uuid.uuid4())
            return [], gr.update(visible=False, value=""), new_session

        # ── Wire events ──────────────────────────────────────────────────
        send_inputs  = [msg_input, chatbot, session_state]
        send_outputs = [msg_input, chatbot, product_display, session_state]

        msg_input.submit(respond, send_inputs, send_outputs)
        send_btn.click(respond, send_inputs, send_outputs)

        qa_trending.click(
            lambda h, s: respond("What are the best trending products right now?", h, s),
            [chatbot, session_state], send_outputs,
        )
        qa_cheap.click(
            lambda h, s: respond("Show me budget-friendly products under $20", h, s),
            [chatbot, session_state], send_outputs,
        )
        qa_cats.click(
            lambda h, s: respond("What categories of products do you have?", h, s),
            [chatbot, session_state], send_outputs,
        )
        qa_install.click(
            lambda h, s: respond("What installment plans do you offer?", h, s),
            [chatbot, session_state], send_outputs,
        )

        handoff_btn.click(handle_handoff, [chatbot, session_state], [chatbot, product_display, session_state])
        clear_btn.click(clear_chat, [], [chatbot, product_display, session_state])

    return app


# ──────────────────────────────────────────────────────────────────────────────
#  10.  Launch
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
