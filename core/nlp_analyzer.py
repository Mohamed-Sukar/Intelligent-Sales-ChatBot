import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure vader_lexicon is downloaded
try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()

def classify_intent(message):
    """
    Classifies the user intent using rule-based keywords and regex boundaries.
    Handles None and unexpected types gracefully.
    """
    if not isinstance(message, str) or not message.strip():
        return "general"
        
    message = message.lower()
    
    def has_match(words, text):
        pattern = r'\b(?:' + '|'.join(words) + r')\b'
        return re.search(pattern, text) is not None

    # Priority 1: Handoff
    if has_match(["human", "agent", "representative", "real person", "talk to someone"], message):
        return "handoff"
        
    # Priority 2: Complaint (excluding negations like "don't hate")
    msg_no_negation = re.sub(r"\b(don't|do not|not)\s+(hate|terrible|worst|angry|complaint)\b", "", message)
    if has_match(["terrible", "worst", "hate", "angry", "complaint", "bad"], msg_no_negation):
        return "complaint"
        
    # Priority 3: Installment
    if has_match(["installment", "monthly", "payment plan", "emi", "installments"], message):
        return "installment"
        
    # Priority 4: Price
    if has_match(["how much", "price", "cost", "cheap", "expensive"], message):
        return "price"
        
    # Priority 5: Reviews
    if has_match(["review", "customers say", "good product", "worth", "reviews"], message):
        return "reviews"
        
    # Priority 6: Browse
    if has_match(["show me", "browse", "looking for", "find", "search"], message):
        return "browse"
    
    return "general"

def analyze_sentiment(message):
    """
    Analyzes sentiment of the message using VADER.
    Returns a compound score from -1.0 to +1.0.
    """
    if not isinstance(message, str) or not message.strip():
        return 0.0
    scores = sia.polarity_scores(message)
    return scores['compound']

def count_complaints(session_history):
    """
    Helper function to count the number of complaints in the session history.
    Safely ignores non-string messages or parses dict structures if needed.
    """
    if not session_history:
        return 0
        
    count = 0
    for msg in session_history:
        if isinstance(msg, str):
            if classify_intent(msg) == "complaint":
                count += 1
        elif isinstance(msg, dict):
            text = msg.get("text", "")
            if isinstance(text, str) and classify_intent(text) == "complaint":
                count += 1
    return count

def should_handoff(message, sentiment_score, intent, complaint_count):
    """
    Determines whether to hand off to a human agent based on:
    - Extreme negative sentiment (<-0.5)
    - Explicit handoff request
    - Repeated complaints (3 or more, and current message is negative/complaint)
    """
    # 1. Negative sentiment
    if sentiment_score < -0.5:
        return True, "غضب شديد — تحويل تلقائي"
    
    # 2. Explicit request
    if intent == "handoff":
        return True, "المستخدم طلب التحدث مع موظف"
    
    # 3. Repeated complaints (only trigger if currently complaining or unhappy)
    if complaint_count >= 3 and (intent == "complaint" or sentiment_score < 0):
        return True, "تكرار الشكوى — المستخدم غير راضٍ"
    
    # 4. No reason for handoff
    return False, None

def analyze_message(message, session_history=None):
    """
    Main pipeline to analyze user message.
    """
    if session_history is None:
        session_history = []
        
    intent = classify_intent(message)
    sentiment = analyze_sentiment(message)
    complaint_count = count_complaints(session_history)
    
    handoff, reason = should_handoff(message, sentiment, intent, complaint_count)
    
    return {
        "intent": intent,           # browse, price, complaint, ...
        "sentiment_score": sentiment, # -1 to +1
        "should_handoff": handoff,   # True / False
        "handoff_reason": reason     # Handoff reason
    }
