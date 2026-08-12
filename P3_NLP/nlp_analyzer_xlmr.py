"""P3 NLP analyzer for Arabic/multilingual sales conversations.

Features:
- Rule-based intent classification with Arabic and English keywords.
- Optional Hugging Face sentiment model, configured with SENTIMENT_MODEL_NAME.
- Handoff detection for severe negativity, explicit human requests, abuse,
  and repeated complaints.

Install for XLM-R inference:
    pip install transformers torch

Important:
    bert-base-multilingual-cased is a base encoder, not a sentiment classifier.
    SENTIMENT_MODEL_NAME must point to a fine-tuned sentiment checkpoint.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Iterable


# Set this to a fine-tuned multilingual XLM-R sentiment checkpoint.
# Do not use the plain base mBERT checkpoint unless you fine-tuned it first.
SENTIMENT_MODEL_NAME = os.getenv(
    "SENTIMENT_MODEL_NAME",
    "models/xlmr-arabic-english-sentiment",
)


INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "handoff": (
        "موظف", "خدمة العملاء", "خدمه العملاء", "مندوب", "شخص حقيقي",
        "اكلم حد", "عايز اكلم حد", "اريد موظف", "أريد موظف",
        "human", "agent", "representative", "talk to someone",
    ),
    "complaint": (
        "شكوى", "شكوي", "سيء", "سيئ", "وحش", "زفت", "نصاب", "غبي",
        "مش راضي", "غير راضي", "خدمة سيئة", "خدمه سيئه", "مشكلة", "مشكله",
        "terrible", "worst", "hate", "angry", "bad", "complaint",
    ),
    "installment": (
        "تقسيط", "قسط", "أقساط", "اقساط", "قسط شهري", "دفع شهري",
        "تمويل", "installment", "installments", "monthly", "payment plan", "emi",
    ),
    "price": (
        "سعر", "بكام", "بكم", "تكلفة", "رخيص", "غالي", "اغلى", "أغلى",
        "price", "cost", "cheap", "expensive", "how much",
    ),
    "reviews": (
        "تقييم", "تقييمات", "آراء العملاء", "رأي العملاء", "الناس بتقول",
        "يستاهل", "مراجعة", "مراجعات", "review", "reviews", "worth",
    ),
    "cross_sell": (
        "إكسسوارات", "اكسسوارات", "ملحقات", "حاجات معاه", "منتجات تانية",
        "accessories", "related products", "what else", "cross sell",
    ),
    "browse": (
        "عايز", "أريد", "اريد", "ابحث", "دور على", "دورلي على", "وريني", "اعرض",
        "منتجات", "موبايل", "هاتف", "لابتوب", "سماعة", "show me", "browse", "find", "search",
    ),
}

ABUSE_TERMS: tuple[str, ...] = (
    "غبي", "حمار", "نصاب", "نصّاب", "وسخ", "حقير", "يلعن", "شتيمة",
    "idiot", "stupid", "scam", "fraud", "shut up",
)


def normalize_text(message: Any) -> str:
    """Return a safe, lowercase string for matching."""
    if not isinstance(message, str):
        return ""
    return " ".join(message.strip().lower().split())


def has_match(words: Iterable[str], text: str) -> bool:
    """Substring matching works more reliably than \b for Arabic text."""
    return any(word in text for word in words)


def classify_intent(message: Any) -> str:
    """Classify the highest-priority intent without calling an LLM."""
    text = normalize_text(message)
    if not text:
        return "general"

    # Priority matters: a request for a human must override a product intent.
    priority = (
        "handoff",
        "complaint",
        "installment",
        "price",
        "cross_sell",
        "browse",
        "reviews",
    )
    for intent in priority:
        if has_match(INTENT_KEYWORDS[intent], text):
            return intent
    return "general"


def contains_abuse(message: Any) -> bool:
    return has_match(ABUSE_TERMS, normalize_text(message))


@lru_cache(maxsize=1)
def _get_sentiment_pipeline():
    """Load the sentiment model only when it is first needed."""
    # The default path is created after fine-tuning. Do not load raw mBERT.
    if not SENTIMENT_MODEL_NAME or (
        not SENTIMENT_MODEL_NAME.startswith(("http://", "https://"))
        and not os.path.isdir(SENTIMENT_MODEL_NAME)
    ):
        return None

    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Install transformers and torch, or unset SENTIMENT_MODEL_NAME."
        ) from exc

    return pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL_NAME,
        tokenizer=SENTIMENT_MODEL_NAME,
        truncation=True,
        max_length=256,
    )


def _label_to_score(label: str, confidence: float) -> tuple[str, float]:
    """Normalize common checkpoint labels to a score in [-1, 1]."""
    normalized = label.lower().strip()

    if normalized in {"negative", "neg", "very_negative", "label_0"}:
        return "negative", -confidence
    if normalized in {"positive", "pos", "very_positive", "label_2"}:
        return "positive", confidence
    if normalized in {"neutral", "neu", "label_1"}:
        return "neutral", 0.0

    # Unknown labels should not silently be treated as positive.
    return "neutral", 0.0


def analyze_sentiment(message: Any) -> dict[str, Any]:
    """Analyze Arabic/multilingual sentiment using the configured HF model."""
    text = normalize_text(message)
    if not text:
        return {"score": 0.0, "label": "neutral", "confidence": 0.0, "source": "empty"}

    model = _get_sentiment_pipeline()
    if model is None:
        # Safe development fallback. Production should configure a checkpoint.
        return {
            "score": 0.0,
            "label": "neutral",
            "confidence": 0.0,
            "source": "mbert_checkpoint_not_found",
        }

    result = model(text[:1000])[0]
    confidence = float(result.get("score", 0.0))
    label, score = _label_to_score(str(result.get("label", "neutral")), confidence)
    return {
        "score": round(max(-1.0, min(1.0, score)), 4),
        "label": label,
        "confidence": round(confidence, 4),
        "source": "huggingface_sentiment_model",
    }


def count_complaints(session_history: Any) -> int:
    """Count complaint messages in strings or common chat dict formats."""
    if not session_history:
        return 0

    count = 0
    for item in session_history:
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = item.get("text") or item.get("content") or item.get("message") or ""
        else:
            text = ""

        if classify_intent(text) == "complaint":
            count += 1
    return count


def should_handoff(
    message: Any,
    sentiment_score: float,
    intent: str,
    complaint_count: int,
) -> tuple[bool, str | None]:
    """Apply deterministic escalation rules."""
    try:
        score = float(sentiment_score)
    except (TypeError, ValueError):
        score = 0.0

    if contains_abuse(message):
        return True, "إساءة أو تهديد — تحويل فوري لموظف"
    if score < -0.5:
        return True, "غضب شديد — تحويل تلقائي"
    if intent == "handoff":
        return True, "المستخدم طلب التحدث مع موظف"
    if complaint_count >= 3 and (intent == "complaint" or score < 0):
        return True, "تكرار الشكوى — المستخدم غير راضٍ"
    return False, None


def analyze_message(message: Any, session_history: Any = None) -> dict[str, Any]:
    """Run the complete P3 understanding pipeline."""
    history = session_history or []
    intent = classify_intent(message)
    sentiment = analyze_sentiment(message)
    complaint_count = count_complaints(history)
    handoff, reason = should_handoff(
        message=message,
        sentiment_score=sentiment["score"],
        intent=intent,
        complaint_count=complaint_count,
    )

    return {
        "intent": intent,
        "sentiment_score": sentiment["score"],
        "sentiment_label": sentiment["label"],
        "sentiment_confidence": sentiment["confidence"],
        "sentiment_source": sentiment["source"],
        "complaint_count": complaint_count,
        "contains_abuse": contains_abuse(message),
        "should_handoff": handoff,
        "handoff_reason": reason,
    }


if __name__ == "__main__":
    examples = [
        "عايز موبايل كويس تحت 500 دولار",
        "بكام السماعة وهل ينفع تقسيط؟",
        "الخدمة سيئة جداً وعايز أكلم موظف",
        "إيه رأي العملاء في المنتج؟",
    ]
    for example in examples:
        print(example)
        print(analyze_message(example))
        print("-" * 60)
