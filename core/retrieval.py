import os
import pickle
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from core.config import (
    DEFAULT_BM25_PATH,
    DEFAULT_FAISS_PATH,
    RRF_K,
    logger,
    _ARABIC_DIACRITICS,
    _ARABIC_TATWEEL,
    _TOKEN_RE,
    INSTALMENT_RATES,
)

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

    def load_indexes(
        self,
        products: List[Dict[str, Any]],
        faiss_path: str = DEFAULT_FAISS_PATH,
        bm25_path: str = DEFAULT_BM25_PATH,
    ) -> None:
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"FAISS index not found at {faiss_path}")
        if not os.path.exists(bm25_path):
            raise FileNotFoundError(f"BM25 index not found at {bm25_path}")
            
        self.products = list(products)
        self.rich_texts = [self._build_rich_text(p) for p in self.products]
        self.categories = sorted({
            str(p.get("category", "")).strip()
            for p in self.products
            if str(p.get("category", "")).strip()
        })
        
        self.faiss_index = faiss.read_index(faiss_path)
        with open(bm25_path, "rb") as fh:
            self.bm25 = pickle.load(fh)
        self._is_built = True

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
