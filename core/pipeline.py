import re
from typing import Any, Dict, List, Optional

from core.llm import LLMManager
from core.retrieval import SalesRetrievalEngine

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
8. Reply in the same language the user wrote in (Arabic or English).
9. IMPORTANT: ALWAYS format your responses using Markdown. When listing products, YOU MUST use a main bullet point for the product name, and NESTED bullet points (indented) for the price and rating below it. For example:
- Product Name
  * Price: $X (Y% off)
  * Rating: Z/5 (W reviews)
NEVER output all product details on a single squished line."""

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


class ConversationMemory:
    def __init__(self) -> None:
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def _session(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self._sessions:
            if len(self._sessions) >= 100:
                oldest = next(iter(self._sessions))
                del self._sessions[oldest]
            self._sessions[session_id] = {"messages": [], "viewed_products": []}
        return self._sessions[session_id]

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


class HumanHandoffPolicy:
    def __init__(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.analyzer = SentimentIntensityAnalyzer()
        except ImportError:
            print("[WARNING] vaderSentiment not installed. Sentiment analysis will be disabled.")
            self.analyzer = None

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
    
    _FRUSTRATION_PHRASES = (
        "sucks", "shit", "terrible", "worst", "garbage", "trash", "fuck", 
        "stupid", "idiot", "hate", "awful", "scam", "bs", "bullshit",
        "زفت", "خرا", "نصب", "حرامية", "سيء", "أسوأ", "زبالة", "غبي", "نصابين", "قرف"
    )

    def evaluate(self, user_message: str, request_handoff: bool = False) -> Optional[str]:
        if request_handoff:
            return "frontend_request"
        normalized = " ".join((user_message or "").casefold().split())
        
        if any(phrase in normalized for phrase in self._EXPLICIT_REQUEST_PHRASES):
            return "explicit_customer_request"
            
        import re
        for phrase in self._FRUSTRATION_PHRASES:
            # For English words, use word boundaries to avoid false positives (e.g. 'shit' in 'shitsu')
            # For Arabic, simple inclusion is generally fine or we can rely on standard \b
            pattern = r'\b' + re.escape(phrase) + r'\b' if phrase.isascii() else re.escape(phrase)
            if re.search(pattern, normalized, re.IGNORECASE):
                return "customer_frustration_keyword"
                
        # VADER Sentiment Analysis
        if self.analyzer and user_message.strip():
            scores = self.analyzer.polarity_scores(user_message)
            if scores.get("compound", 0) <= -0.5:
                return "customer_frustration_vader"

        return None


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
        self._quick_action_cache: Dict[str, Dict[str, Any]] = {}
        self.STATIC_PROMPTS = {
            "What are the best trending products right now?",
            "Show me budget-friendly products under $20",
            "What categories of products do you have?",
            "What installment plans do you offer?",
        }

    def process_message(
        self,
        user_message: str,
        session_id: str,
        request_handoff: bool = False,
        installment_months: Optional[int] = None,
        suppress_auto_handoff: bool = False,
    ) -> Dict[str, Any]:
        handoff_reason = self.handoff_policy.evaluate(user_message, request_handoff)
        
        # UI Logic Support: Ignore automatic frustration handoff if suppressed (for the 2-strike system)
        if handoff_reason and suppress_auto_handoff and not request_handoff:
            handoff_reason = None

        self.memory.save_message(session_id, "user", user_message)

        if handoff_reason:
            summary = self._generate_handoff_summary(session_id, handoff_reason)
            return {
                "type": "handoff",
                "message": "I\u2019ll connect you with a human representative.",
                "summary_for_agent": summary,
                "handoff_reason": handoff_reason,
            }
            
        clean_msg = user_message.strip() if user_message else ""
        if clean_msg in self.STATIC_PROMPTS:
            if clean_msg in self._quick_action_cache:
                cached_resp = self._quick_action_cache[clean_msg]
                self.memory.save_message(session_id, "bot", cached_resp["message"])
                if cached_resp.get("products"):
                    self.memory.add_viewed_products(session_id, cached_resp["products"][:1])
                return cached_resp

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

        response_dict = {
            "type": "response",
            "message": llm_response["content"],
            "products": surfaced_products,
            "installment": installment_info,
            "llm_source": llm_response["source"],
        }
        
        if clean_msg in self.STATIC_PROMPTS:
            self._quick_action_cache[clean_msg] = response_dict
            
        return response_dict

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
            if len(first_word) >= 3:
                pattern = r"\b" + re.escape(first_word) + r"\b"
                if re.search(pattern, llm_content, re.IGNORECASE):
                    matched = True

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
