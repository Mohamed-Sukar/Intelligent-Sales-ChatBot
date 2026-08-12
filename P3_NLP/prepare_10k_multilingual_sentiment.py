"""Prepare a balanced 10k Arabic/English sentiment CSV from Hugging Face.

Source dataset:
    https://huggingface.co/datasets/clapAI/MultiLingualSentiment

The source dataset card reports Apache-2.0 licensing, Arabic/English coverage,
three sentiment labels, and source/language metadata. This script keeps the
source and domain columns for traceability.

Output:
    sentiment_train_10k.csv

Install:
    pip install -U datasets pandas

Run:
    python prepare_10k_multilingual_sentiment.py
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset

DATASET_ID = "clapAI/MultiLingualSentiment"
OUTPUT = Path("sentiment_train_10k.csv")
SEED = 42

# 2 languages x 3 classes x 1,666 = 9,996; four extra rows are added later.
PER_LANGUAGE_LABEL = 1666
LANGUAGES = ("ar", "en")
LABELS = ("negative", "neutral", "positive")


def main() -> None:
    random.seed(SEED)
    quotas = {
        (language, label): PER_LANGUAGE_LABEL
        for language in LANGUAGES
        for label in LABELS
    }
    buckets: dict[tuple[str, str], list[dict[str, str]]] = {
        key: [] for key in quotas
    }

    print(f"Loading {DATASET_ID} in streaming mode...")
    stream = load_dataset(DATASET_ID, split="train", streaming=True)
    stream = stream.shuffle(seed=SEED, buffer_size=20_000)

    scanned = 0
    for row in stream:
        scanned += 1
        language = str(row.get("language", "")).lower().strip()
        label = str(row.get("label", "")).lower().strip()
        text = str(row.get("text", "")).strip()
        key = (language, label)

        if key not in quotas or not text:
            continue
        if len(buckets[key]) >= quotas[key]:
            continue

        buckets[key].append(
            {
                "text": text.replace("\r", " ").replace("\n", " "),
                "label": label,
                "language": language,
                "source": str(row.get("source", "")),
                "domain": str(row.get("domain", "")),
            }
        )

        if all(len(buckets[key]) >= quotas[key] for key in quotas):
            break

        if scanned % 100_000 == 0:
            print(f"Scanned {scanned:,} rows...")

    missing = {key: quotas[key] - len(rows) for key, rows in buckets.items() if len(rows) < quotas[key]}
    if missing:
        raise RuntimeError(
            "Could not collect the balanced target. Missing quotas: "
            f"{missing}. The source distribution may have changed."
        )

    rows = [row for bucket in buckets.values() for row in bucket]
    random.shuffle(rows)

    # Add four extra rows only if needed to reach exactly 10,000.
    rows = rows[:10_000]
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["text", "label", "language", "source", "domain"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows):,} rows to {OUTPUT}")
    print("Language counts:", Counter(row["language"] for row in rows))
    print("Label counts:", Counter(row["label"] for row in rows))
    print("Source: https://huggingface.co/datasets/clapAI/MultiLingualSentiment")
    print("License reported by dataset card: Apache-2.0")


if __name__ == "__main__":
    main()
