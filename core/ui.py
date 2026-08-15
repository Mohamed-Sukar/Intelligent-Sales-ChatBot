import uuid
from typing import Any, Dict, List
import gradio as gr

import os
import re
from core.config import DEFAULT_FAISS_PATH, DEFAULT_BM25_PATH
from core.data_loader import load_products_from_csv
from core.retrieval import SalesRetrievalEngine
from core.pipeline import RAGPipeline

print("⏳ Loading product catalogue …")
products = load_products_from_csv()
engine = SalesRetrievalEngine()

if os.path.exists(DEFAULT_FAISS_PATH) and os.path.exists(DEFAULT_BM25_PATH):
    print("⏳ Loading search indexes from disk …")
    engine.load_indexes(products)
else:
    print("⏳ Building search indexes …")
    engine.build_indexes(products)

pipeline = RAGPipeline(engine)
print("✅ Pipeline ready — launching Gradio …")

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
        session_state = gr.State(value=lambda: str(uuid.uuid4()))

        gr.HTML("""
        <div class="app-header">
            <h1>SmartSales Bot</h1>
            <p>Your AI-powered shopping assistant. Ask about products, prices,<br>
            installment plans, or categories — in English or Arabic.</p>
        </div>
        """)

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

        product_display = gr.HTML(value="", visible=False)

        with gr.Row(elem_classes=["quick-actions"]):
            qa_trending = gr.Button("Trending products", elem_classes=["quick-chip"], size="sm")
            qa_cheap    = gr.Button("Budget-friendly picks", elem_classes=["quick-chip"], size="sm")
            qa_cats     = gr.Button("Browse categories", elem_classes=["quick-chip"], size="sm")
            qa_install  = gr.Button("Installment plans", elem_classes=["quick-chip"], size="sm")

        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Type your message…",
                show_label=False,
                container=False,
                scale=7,
                autofocus=True,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)

        with gr.Row():
            handoff_btn = gr.Button("Contact a Representative", elem_classes=["handoff-btn"], size="sm")
            clear_btn   = gr.Button("Clear Chat", elem_classes=["handoff-btn"], size="sm")

        status_html = gr.HTML('<div class="status-bar">Powered by Hybrid RAG — FAISS + BM25 · Dual-API LLM (OpenRouter / Gemini)</div>')

        def respond(user_message: str, chat_history: list, session_id: str):
            if not user_message or not user_message.strip():
                return "", chat_history, gr.update(visible=False, value=""), session_id

            chat_history = chat_history + [{"role": "user", "content": user_message}]

            installment_months = None
            match = re.search(r'\b(\d+)\s*(?:months?|mo|شهر|شهور|أشهر)\b', user_message, re.IGNORECASE)
            if match:
                installment_months = int(match.group(1))

            result = pipeline.process_message(
                user_message,
                session_id=session_id,
                installment_months=installment_months
            )

            bot_reply = result["message"]
            chat_history = chat_history + [{"role": "assistant", "content": bot_reply}]

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
