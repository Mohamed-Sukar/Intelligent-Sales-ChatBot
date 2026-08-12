"""
p2_retrieval_engine.py — P2: Search & RAG Engineer
====================================================

Deterministic retrieval + business logic layer for the Intelligent Sales
Chatbot. This module is intentionally API-free: it is the internal engine
that the async Dual-API LLM manager (OpenRouter / Gemini) sits on top of.

Pipeline
--------
    1. build_indexes(products)   -> embed + FAISS (IndexFlatL2) + BM25 (BM25Okapi)
    2. hybrid_search(query)      -> FAISS top-15   ⊕   BM25 top-15
                                     merged via Reciprocal Rank Fusion (RRF)
    3. get_recommendations(id)   -> cross-sell within ±30% price, rated-sorted
    4. calculate_installment()   -> pricing plans
    5. should_handoff()          -> human escalation rules
    6. build_llm_context()       -> pretty JSON context for the external LLM

Language support: English + Arabic. `_normalize()` strips Arabic diacritics and
harmonizes letter variants so BM25 tokenization works well in both scripts.

Usage
-----
    from p2_retrieval_engine import SalesRetrievalEngine

    engine = SalesRetrievalEngine()
    engine.build_indexes(products)
    hits = engine.hybrid_search("Do you have a cheap gaming laptop?")
    cross_sell = engine.get_recommendations("ASUS-G15")
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import unicodedata
from typing import Any, Callable, Dict, List, Optional, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Defaults for the on-disk index files (README convention) ────────────
DEFAULT_DB_PATH = "database/chatbot.db"
DEFAULT_FAISS_PATH = "database/faiss.index"
DEFAULT_BM25_PATH = "database/bm25_index.pkl"

# ── RRF constant (score = 1 / (rank + RRF_K)) ────────────────────────────
RRF_K = 60

# ── Instalment interest-rate table: months -> rate ───────────────────────
INSTALMENT_RATES: Dict[int, float] = {3: 0.10, 6: 0.15, 12: 0.20}

# ── Keywords that trigger a human-agent handoff ──────────────────────────
HANDOFF_KEYWORDS: Tuple[str, ...] = (
    "talk to human",
    "agent",
    "representative",
    "مشكلة",
    "خدمة العملاء",
)

# Unicode ranges: Arabic letters (normalized set) and letter hamza forms.
_ARABIC_ALEF = ("\u0627", "\u0623", "\u0625", "\u0622")  # ا أ إ آ
_ARABIC_DIACRITICS = "\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u0659\u065A\u065B\u065C\u065D\u065E\u065F"
_ARABIC_TATWEEL = "\u0640"

# Tokenizer: keeps hyphenated alphanumeric tokens (SKUs like "G-15") whole,
# and treats contiguous Arabic letters/digits as tokens.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*|[\u0621-\u064A0-9]+")


class SalesRetrievalEngine:
    """Deterministic hybrid retrieval + business logic engine (English/Arabic)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None

        # Dual indexes
        self.faiss_index: Optional[faiss.IndexFlatL2] = None
        self.bm25: Optional[BM25Okapi] = None

        # Corpus storage
        self.products: List[Dict[str, Any]] = []
        self.rich_texts: List[str] = []
        self._vectors: Optional[np.ndarray] = None
        self._is_built: bool = False

    # ────────────────────────────────────────────────────────────────────
    # Private helpers
    # ────────────────────────────────────────────────────────────────────

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load the sentence-transformer model (cached for repeated calls)."""
        if self._model is None:
            logger.info("Loading embedding model '%s' ...", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for both English and Arabic.

        - Lowercase (handles Arabic once letters are de-diacriticised).
        - NFKC unicode normalization.
        - Strip Arabic diacritics and tatweel (ـ).
        - Collapse Arabic letter variants onto canonical bases
          (أ/إ/آ -> ا , ة -> ه , ى -> ي).
        - Collapse runs of whitespace to a single space.
        """
        if not text:
            return ""

        text = unicodedata.normalize("NFKC", str(text))
        text = text.lower()

        # strip Arabic diacritics and elongation marks
        text = re.sub(f"[{_ARABIC_DIACRITICS}{_ARABIC_TATWEEL}]", "", text)

        # harmonise Arabic letter variants (single-char replacements)
        text = text.replace("\u0623", "\u0627").replace("\u0625", "\u0627").replace("\u0622", "\u0627")
        text = text.replace("\u0629", "\u0647")  # ة -> ه
        text = text.replace("\u0649", "\u064A")  # ى -> ي

        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        """SKU-aware + Arabic-aware tokenization.

        Uses the normalized text so tokens are already diacritic-free,
        lower-cased and letter-harmonized. Hyphenated alnum tokens (G-15)
        are kept as a single token for exact SKU matching.
        """
        return _TOKEN_RE.findall(cls._normalize(text))

    @staticmethod
    def _build_rich_text(product: Dict[str, Any]) -> str:
        """Combine the most semantically useful product fields into one string."""
        parts = [
            str(product.get("title", "")),
            str(product.get("category", "")),
            str(product.get("product_description", "")),
            str(product.get("product_specifications", "")),
        ]
        return " ".join(p for p in parts if p).strip()

    def _require_built(self) -> None:
        if not self._is_built:
            raise RuntimeError(
                "Indexes are not built yet. Call build_indexes() or load_indexes() first."
            )

    def _require_query(self, query: str) -> str:
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        return query.strip()

    # ────────────────────────────────────────────────────────────────────
    # A1-A3. Initialisation & index management
    # ────────────────────────────────────────────────────────────────────

    def build_indexes(
        self,
        products: List[Dict[str, Any]],
        faiss_path: str = DEFAULT_FAISS_PATH,
        bm25_path: str = DEFAULT_BM25_PATH,
    ) -> Tuple[str, str]:
        """Preprocess products, embed them and build FAISS + BM25 indexes.

        Args:
            products: list of product dicts (as produced by ProductDatabase).
            faiss_path: where to persist the FAISS index.
            bm25_path: where to persist the pickled BM25 index.

        Returns:
            Tuple of (faiss_path, bm25_path) for the saved artifacts.

        Raises:
            ValueError: if `products` is empty.
        """
        if not products:
            raise ValueError("Cannot build indexes from an empty product list.")

        # Ensure the parent directory for on-disk artifacts exists.
        for path in (faiss_path, bm25_path):
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)

        self.products = list(products)
        self.rich_texts = [self._build_rich_text(p) for p in self.products]

        # -- Dense: FAISS (IndexFlatL2 -> Euclidean distance) --------------
        model = self._get_model()
        vectors = model.encode(
            self.rich_texts,
            normalize_embeddings=False,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        dim = self._vectors.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dim)
        self.faiss_index.add(self._vectors)
        faiss.write_index(self.faiss_index, faiss_path)
        logger.info("FAISS index saved to %s (%d vectors, dim=%d)", faiss_path, len(self._vectors), dim)

        # -- Sparse: BM25Okapi over normalized tokens -----------------------
        tokenized = [self._tokenize(t) for t in self.rich_texts]
        self.bm25 = BM25Okapi(tokenized)
        with open(bm25_path, "wb") as fh:
            pickle.dump(self.bm25, fh)
        logger.info("BM25 index saved to %s", bm25_path)

        self._is_built = True
        return faiss_path, bm25_path

    def load_indexes(
        self,
        faiss_path: str,
        bm25_path: str,
        products: List[Dict[str, Any]],
    ) -> None:
        """Load pre-computed indexes from disk to save memory & startup time.

        Args:
            faiss_path: path to a previously saved FAISS index.
            bm25_path: path to a previously saved (pickled) BM25 index.
            products: the same corpus used to build those indexes.

        Note:
            Vector embeddings are NOT recomputed here; the FAISS index file
            already contains them. Only the product dicts are restored, so a
            future encode() (e.g. for recommendations) is still available.

        Raises:
            ValueError / RuntimeError: if paths are missing or unreadable.
        """
        if not os.path.exists(faiss_path) or not os.path.exists(bm25_path):
            raise ValueError(f"Index paths missing: {faiss_path!r} / {bm25_path!r}")

        self.faiss_index = faiss.read_index(faiss_path)
        with open(bm25_path, "rb") as fh:
            self.bm25 = pickle.load(fh)

        self.products = list(products)
        self.rich_texts = [self._build_rich_text(p) for p in self.products]

        # Reconstruct vectors for get_recommendations() without an extra
        # forward pass over the whole corpus: read them straight back from
        # the FAISS index (cheap, deterministic).
        self._vectors = np.array(self.faiss_index.reconstruct_n(0, self.faiss_index.ntotal))

        self._is_built = True
        logger.info("Loaded FAISS (%d vectors) + BM25 from disk.", self.faiss_index.ntotal)

    def build_indexes_from_database(
        self,
        db_path: str = DEFAULT_DB_PATH,
        faiss_path: str = DEFAULT_FAISS_PATH,
        bm25_path: str = DEFAULT_BM25_PATH,
        error_on_empty: bool = True,
    ) -> Tuple[str, str]:
        """Bridge to the existing SQLite pipeline: build indexes from the DB.

        Pulls all active products through ``ProductDatabase.get_all_active_products()``
        (from ``core.data_pipeline``) and delegates to :meth:`build_indexes`.

        Args:
            db_path: path to the SQLite database produced by data_pipeline.
            faiss_path: where to persist the FAISS index.
            bm25_path: where to persist the pickled BM25 index.
            error_on_empty: if True, raise when the DB has no active products.

        Returns:
            Tuple of (faiss_path, bm25_path).

        Raises:
            FileNotFoundError: if the database file does not exist.
            RuntimeError: if the DB has no active products and error_on_empty=True.
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"Database not found at {db_path!r}. Run core.data_pipeline first."
            )

        from core.data_pipeline import ProductDatabase  # local import to avoid hard coupling

        db = ProductDatabase(db_path)
        try:
            products = db.get_all_active_products()
        finally:
            db.close()

        if not products:
            if error_on_empty:
                raise RuntimeError(
                    f"Database {db_path!r} has no active products to index."
                )
            logger.warning("Database %s has no active products; nothing indexed.", db_path)
            return faiss_path, bm25_path

        logger.info("Loaded %d active products from %s", len(products), db_path)
        return self.build_indexes(products, faiss_path=faiss_path, bm25_path=bm25_path)

    # ────────────────────────────────────────────────────────────────────
    # B4. Core hybrid retrieval (RRF fusion)
    # ────────────────────────────────────────────────────────────────────

    @staticmethod
    def _rrf_add(score_map: Dict[int, float], positions: List[int]) -> None:
        """Accumulate RRF scores for an ordering of corpus positions.

        rank is the 0-indexed position in `positions`; a document appearing
        early in *either* list gets a high score, and appearing in both lists
        stacks its contribution — which is how RRF rewards agreement.
        """
        for rank, pos in enumerate(positions):
            score_map[pos] = score_map.get(pos, 0.0) + 1.0 / (rank + RRF_K)

    def _dense_search_positions(self, query_vec: np.ndarray, top_n: int) -> List[int]:
        """FAISS search -> ordered list of corpus positions (best-first)."""
        assert self.faiss_index is not None
        k = min(top_n, self.faiss_index.ntotal)
        if k <= 0:
            return []
        _d, idx = self.faiss_index.search(query_vec, k)
        return [int(i) for i in idx[0] if i != -1]

    def _sparse_search_positions(self, query_tokens: List[str], top_n: int) -> List[int]:
        """BM25 -> ordered list of corpus positions (best-first)."""
        assert self.bm25 is not None
        scores = np.asarray(self.bm25.get_scores(query_tokens))
        if scores.size == 0:
            return []
        k = min(top_n, scores.size)
        # np.argpartition is O(n) — we only fully sort the top-k slice.
        part = np.argpartition(-scores, k - 1)[:k]
        order = part[np.argsort(-scores[part], kind="stable")]
        return [int(i) for i in order]

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Run dense+sparse retrieval and fuse results with Reciprocal Rank Fusion.

        Args:
            query: the user's natural-language / keyword query (EN or AR).
            top_k: number of final results to return.

        Returns:
            A list of the top-k product dicts, best first.

        Raises:
            RuntimeError: if indexes are not built.
            ValueError: if `query` is empty.
        """
        self._require_built()
        query = self._require_query(query)
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        # vectorised query for the dense side
        dense_query_vec = self._get_model().encode(
            [self._normalize(query)],
            normalize_embeddings=False,
            convert_to_numpy=True,
        )

        # independent top-15 lists from each index
        dense_top15 = self._dense_search_positions(dense_query_vec, 15)
        sparse_top15 = self._sparse_search_positions(self._tokenize(query), 15)

        # RRF fusion — accumulate scores keyed by corpus position
        score_map: Dict[int, float] = {}
        self._rrf_add(score_map, dense_top15)
        self._rrf_add(score_map, sparse_top15)

        # sort by descending RRF score, ties broken by position for stability
        ranked = sorted(score_map.items(), key=lambda kv: (-kv[1], kv[0]))

        return [dict(self.products[pos]) for pos, _score in ranked[:top_k]]

    # ────────────────────────────────────────────────────────────────────
    # C5-C7. Business logic & cross-selling
    # ────────────────────────────────────────────────────────────────────

    def get_recommendations(self, product_id: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Recommend similar products for cross-selling / up-selling.

        Uses semantic similarity (FAISS top-20) as the candidate pool, then
        enforces a strict ±30% price-band constraint and sorts by rating.

        Args:
            product_id: the id of the product the user is viewing.
            top_k: number of recommendations to return.

        Returns:
            Recommended product dicts (original product excluded).

        Raises:
            RuntimeError: indexes not built.
            LookupError: product_id not found in the corpus.
        """
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

        # filter out the original and enforce the ±30% price band
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

        # sort by rating descending (ties broken by ratings_count desc, then pos)
        candidates.sort(
            key=lambda t: (
                -float(self.products[t[0]].get("rating", 0.0)),
                -float(self.products[t[0]].get("ratings_count", 0.0)),
                t[0],
            )
        )

        return [dict(self.products[pos]) for pos, _price in candidates[:top_k]]

    def calculate_installment(self, price: float, months: int = 6) -> Dict[str, Any]:
        """Compute monthly instalment for a given price.

        Rates: 3 -> 10%, 6 -> 15%, 12 -> 20%.

        Args:
            price: product price.
            months: one of {3, 6, 12}.

        Returns:
            dict with keys: monthly_payment, total_with_interest, months.

        Raises:
            ValueError: on non-positive price or unsupported months.
        """
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
        """Decide whether to escalate to a human agent.

        Escalates when the VADER sentiment is very negative (< -0.5) OR when
        the user text contains any escalation keyword (English or Arabic).

        Args:
            sentiment_score: VADER-style sentiment in [-1, 1].
            user_text: raw user message.

        Returns:
            True if the conversation should be handed off to a human.
        """
        if sentiment_score < -0.5:
            return True
        lowered = (user_text or "").lower()
        return any(kw in lowered for kw in HANDOFF_KEYWORDS)

    # ────────────────────────────────────────────────────────────────────
    # D8. LLM pipeline context builder
    # ────────────────────────────────────────────────────────────────────

    def build_llm_context(
        self,
        user_message: str,
        intent: str,
        retrieved_products: List[Dict[str, Any]],
        last_5_messages: List[Dict[str, Any]],
        sentiment: float,
    ) -> str:
        """Build a clean, structured prompt context for the external LLM.

        All data is injected as beautifully-formatted JSON (Arabic-safe), so
        the Dual-API manager (OpenRouter / Gemini) and the downstream LLM can
        parse it reliably.

        Args:
            user_message: the latest user utterance.
            intent: classified intent label (e.g. 'product_inquiry').
            retrieved_products: output of hybrid_search (or []).
            last_5_messages: list of {role, content} dicts (chat history).
            sentiment: numeric sentiment in [-1, 1].

        Returns:
            A single formatted string to include in the LLM prompt.
        """

        def _serialize(obj: Any) -> Any:
            """JSON-safe fallback for numpy types / datetimes."""
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        context = {
            "user_message": user_message,
            "intent": intent,
            "sentiment": sentiment,
            "conversation_history": last_5_messages[-5:] if last_5_messages else [],
            "retrieved_products": retrieved_products,
        }

        payload = json.dumps(context, indent=2, ensure_ascii=False, default=_serialize)
        return (
            "You are the sales representative for our e-commerce store. "
            "Use ONLY the retrieved product data below to answer, never invent offers.\n\n"
            "=== CONTEXT ===\n"
            f"{payload}\n"
            "=== INSTRUCTIONS ===\n"
            "Formulate a friendly sales pitch: highlight the most relevant product(s), "
            "mention price, and suggest an instalment plan or complementary product when useful."
        )


# ════════════════════════════════════════════════════════════════════════
#  Test block — mock mixed EN/AR dataset and a quick end-to-end run
# ════════════════════════════════════════════════════════════════════════

def _mock_products() -> List[Dict[str, Any]]:
    """Return a small mock catalogue (English + Arabic products)."""
    return [
        {
            "product_id": "ASUS-G15",
            "title": "ASUS ROG Strix G-15 Gaming Laptop",
            "category": "laptops",
            "product_description": "Cheap yet powerful gaming laptop with RTX 4060 and 165Hz display.",
            "product_specifications": "Ryzen 9, 16GB RAM, 1TB SSD, G-15, 1440p",
            "final_price": 1199.00,
            "rating": 4.6,
            "ratings_count": 812,
            "currency": "USD",
        },
        {
            "product_id": "LENOVO-LEGION-5",
            "title": "Lenovo Legion 5 Gaming Laptop",
            "category": "laptops",
            "product_description": "Affordable gaming machine for esports and AAA titles.",
            "product_specifications": "Ryzen 7, 16GB RAM, 512GB SSD, 144Hz",
            "final_price": 1049.00,
            "rating": 4.4,
            "ratings_count": 1240,
            "currency": "USD",
        },
        {
            "product_id": "DELL-XPS-13",
            "title": "Dell XPS 13 Ultrabook",
            "category": "laptops",
            "product_description": "Premium ultra-thin business ultrabook, not for gaming.",
            "product_specifications": "Core i7, 16GB RAM, 512GB SSD",
            "final_price": 1799.00,
            "rating": 4.7,
            "ratings_count": 560,
            "currency": "USD",
        },
        {
            "product_id": "SAMSUNG-MONITOR-27",
            "title": "شاشة سامسونج 27 بوصة للألعاب",
            "category": "monitors",
            "product_description": "شاشة ألعاب بدقة 2K ومعدل تحديث 144 هرتز مع أسعار مناسبة.",
            "product_specifications": "OLED، 2K، 144Hz، G-Sync",
            "final_price": 329.00,
            "rating": 4.3,
            "ratings_count": 431,
            "currency": "USD",
        },
    ]


if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 62)
    print("  SalesRetrievalEngine — local smoke test")
    print("=" * 62)

    catalogue = _mock_products()

    # Run the whole test in a temp dir so the run is side-effect free.
    with tempfile.TemporaryDirectory() as tmp:
        engine = SalesRetrievalEngine()

        print("\n[1/5] build_indexes() ...")
        fpath, bpath = engine.build_indexes(
            catalogue,
            faiss_path=os.path.join(tmp, "products_faiss.index"),
            bm25_path=os.path.join(tmp, "bm25_index.pkl"),
        )
        print(f"      saved -> {os.path.basename(fpath)}, {os.path.basename(bpath)}")

        print("\n[2/5] hybrid_search('Do you have a cheap gaming laptop like the Asus G-15?')")
        hits = engine.hybrid_search("Do you have a cheap gaming laptop like the Asus G-15?", top_k=3)
        for rank, hit in enumerate(hits, 1):
            print(f"      {rank}. {hit['product_id']:<22} ${hit['final_price']:<8} {hit['title']}")

        print("\n[2b] Arabic query: 'أحتاج شَاشَة سَامسونج' (diacritics stripped by _normalize)")
        ar_hits = engine.hybrid_search("أحتاج شَاشَة سَامسونج", top_k=2)
        for rank, hit in enumerate(ar_hits, 1):
            print(f"      {rank}. {hit['product_id']:<22} {hit['title']}")

        print("\n[3/5] get_recommendations('ASUS-G15')  (±30% price band, rated-sorted)")
        recs = engine.get_recommendations("ASUS-G15", top_k=3)
        for rank, rec in enumerate(recs, 1):
            ratio = rec["final_price"] / 1199.00
            print(f"      {rank}. {rec['product_id']:<22} ${rec['final_price']:<8} "
                  f"rating={rec['rating']}  price_ratio={ratio:.2f}")

        print("\n[4/5] calculate_installment()")
        for months in (3, 6, 12):
            plan = engine.calculate_installment(1199.00, months)
            print(f"      {months:>2}m -> {plan}")

        print("\n[5/5] should_handoff()")
        cases = [
            (-0.8, "this product is broken"),          # very negative sentiment
            (0.4, "I want to talk to human please"),   # keyword 'talk to human'
            (0.2, "عندي مشكلة في الطلب"),               # Arabic keyword 'مشكلة'
            (0.9, "thanks, looks great"),              # happy customer -> no handoff
        ]
        for score, text in cases:
            print(f"      sentiment={score:+.1f} text={text!r:<40} -> handoff={engine.should_handoff(score, text)}")

    print("\n" + "=" * 62)
    print("  Smoke test complete.")
    print("=" * 62)
