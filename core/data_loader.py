import os
from typing import Any, Dict, List, Optional
import pandas as pd

from core.config import DEFAULT_CSV_CANDIDATES, TEXT_COLUMNS, NUMERIC_COLUMNS

def _find_csv_path(explicit_path: Optional[str] = None) -> str:
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path
    for candidate in DEFAULT_CSV_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "products_clean.csv not found. Place it in the data/ directory."
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
    if "product_id" not in df.columns:
        df["product_id"] = df.index.astype(str)
    df["product_id"] = df["product_id"].astype(str)
    products = df.to_dict(orient="records")
    print(f"Loaded {len(products)} active products from {path!r}")
    return products
