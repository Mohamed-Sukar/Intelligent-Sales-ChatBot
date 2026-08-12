"""
main.py — End-to-end retrieval runner (P1 -> P2)
==================================================
Wires the existing SQLite data pipeline (core.data_pipeline) to the hybrid
retrieval engine (p2_retrieval_engine) and runs a live demo on the real
913-product catalogue.

Steps
-----
    1. Build `database/chatbot.db` from data/products_clean.csv if missing.
    2. Build FAISS + BM25 indexes from the DB (or load them if already built).
    3. Run hybrid_search demo queries + get_recommendations on real products.

Usage
-----
    python main.py            # build-or-load indexes, run demo
    python main.py --rebuild  # force a full index rebuild
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List

# ── Paths (aligned with the README / data_pipeline conventions) ─────────
DATA_CSV = "data/products_clean.csv"
DB_PATH = "database/chatbot.db"
FAISS_PATH = "database/faiss.index"
BM25_PATH = "database/bm25_index.pkl"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("main")


def ensure_database(exists_ok: bool = True) -> None:
    """Build the SQLite DB from the clean CSV if it does not exist yet."""
    if os.path.exists(DB_PATH):
        logger.info("Database already exists at %s (skipping build).", DB_PATH)
        return
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(
            f"Clean dataset not found at {DATA_CSV!r}; run core.data_pipeline first."
        )
    from core.data_pipeline import setup_data_pipeline

    logger.info("Building database from %s ...", DATA_CSV)
    setup_data_pipeline(csv_path=DATA_CSV, db_path=DB_PATH)
    logger.info("Database ready at %s", DB_PATH)


def build_or_load_engine(rebuild: bool = False) -> Any:
    """Return a ready SalesRetrievalEngine, (re)building indexes as needed."""
    from p2_retrieval_engine import SalesRetrievalEngine

    engine = SalesRetrievalEngine()
    indexes_present = os.path.exists(FAISS_PATH) and os.path.exists(BM25_PATH)

    if rebuild or not indexes_present:
        logger.info("Building FAISS + BM25 indexes ...")
        t0 = time.perf_counter()
        engine.build_indexes_from_database(
            db_path=DB_PATH, faiss_path=FAISS_PATH, bm25_path=BM25_PATH
        )
        logger.info("Indexes built in %.2fs.", time.perf_counter() - t0)
    else:
        logger.info("Loading pre-computed indexes from disk (fast path) ...")
        from core.data_pipeline import ProductDatabase

        db = ProductDatabase(DB_PATH)
        try:
            products = db.get_all_active_products()
        finally:
            db.close()
        engine.load_indexes(FAISS_PATH, BM25_PATH, products)
        logger.info("Loaded %d products + indexes.", len(engine.products))

    return engine


def _fmt(product: Dict[str, Any]) -> str:
    """Compact one-line render of a product for the console."""
    price = product.get("final_price", 0.0)
    rating = product.get("rating", 0.0)
    title = product.get("title", "")
    category = product.get("category", "")
    return (
        f"    • {title.strip()}\n"
        f"      [{product.get('product_id')} | {category} | ${price:,.2f} | "
        f"rating {rating}]"
    )


def _demo_hybrid(engine: Any, query: str, top_k: int = 5) -> None:
    print(f"\n  Q: {query}")
    t0 = time.perf_counter()
    results = engine.hybrid_search(query, top_k=top_k)
    ms = (time.perf_counter() - t0) * 1000
    print(f"  ({len(results)} hits in {ms:,.0f} ms)")
    for r in results:
        print(_fmt(r))


def _demo_recommendations(engine: Any, product_id: str, top_k: int = 3) -> None:
    target = next(
        (p for p in engine.products if str(p.get("product_id")) == str(product_id)),
        None,
    )
    if target is None:
        print(f"\n  get_recommendations({product_id}) -> product not found.")
        return
    print(f"\n  Cross-sell for: {target.get('title')!r} (${target.get('final_price', 0):,.2f})")
    for r in engine.get_recommendations(product_id, top_k=top_k):
        print(_fmt(r))


def pick_demo_product_id(engine: Any) -> str:
    """Return a product id whose price band (+/-30%) admits real neighbours."""
    for p in engine.products:
        pid = str(p.get("product_id"))
        try:
            if engine.get_recommendations(pid, top_k=1):
                return pid
        except Exception:
            continue
    return str(engine.products[0]["product_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end retrieval demo.")
    parser.add_argument("--rebuild", action="store_true", help="force index rebuild")
    args = parser.parse_args()

    print("=" * 64)
    print("  Intelligent Sales ChatBot — Retrieval Layer (P1 -> P2)")
    print("=" * 64)

    # Step 1: ensure DB exists
    ensure_database()

    # Step 2: build or load indexes
    engine = build_or_load_engine(rebuild=args.rebuild)
    print(f"  Indexed corpus: {len(engine.products)} products")

    # Step 3: demo queries on the real catalogue
    print("\n[Hybrid Search]")
    _demo_hybrid(engine, "wireless bluetooth headphones for music", top_k=4)
    _demo_hybrid(engine, "white formal shirt with slim fit", top_k=4)
    _demo_hybrid(engine, "winter jacket for men waterproof", top_k=4)

    print("\n[Cross-sell Recommendations]")
    demo_id = pick_demo_product_id(engine)
    _demo_recommendations(engine, demo_id, top_k=3)

    print("\n" + "=" * 64)
    print("  Done. Index artifacts:")
    print(f"    - {FAISS_PATH}")
    print(f"    - {BM25_PATH}")
    print(f"    - {DB_PATH}")
    print("=" * 64)


if __name__ == "__main__":
    sys.exit(main())
