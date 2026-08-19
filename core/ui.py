from __future__ import annotations

import html as html_lib
import re
import uuid
from typing import Any, Dict, List, Optional

import streamlit as st
from st_copy import copy_button

from core.config import DEFAULT_FAISS_PATH, DEFAULT_BM25_PATH
from core.data_loader import load_products_from_csv
from core.retrieval import SalesRetrievalEngine
from core.pipeline import RAGPipeline

import os


# ──────────────────────────────────────────────────────────────────────────────
#  Page config  (MUST be first Streamlit command)
# ──────────────────────────────────────────────────────────────────────────────
def setup_page():
    st.set_page_config(
        page_title="SmartSales | AI Shopping Concierge",
        page_icon="icon0.png",
        layout="wide",
        initial_sidebar_state="expanded",
    )


    # ──────────────────────────────────────────────────────────────────────────────
    #  Design system — CSS
    # ──────────────────────────────────────────────────────────────────────────────
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
      --ink: #f7f7fb;
      --muted: #a4a6ba;
      --muted-2: #777b94;
      --line: rgba(255,255,255,.10);
      --panel: rgba(21, 24, 45, .78);
      --panel-strong: #181b35;
      --accent: #aa8cff;
      --accent-2: #54d6c5;
      --warning: #f5bf67;
      --danger: #ff7d92;
    }

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp {
      background:
        radial-gradient(circle at 8% 3%, rgba(126, 92, 255, .22), transparent 29rem),
        radial-gradient(circle at 94% 19%, rgba(56, 209, 194, .13), transparent 25rem),
        linear-gradient(145deg, #0b0d1c 0%, #10132a 48%, #0a0d1c 100%);
      color: var(--ink);
    }
    .block-container { max-width: 1450px; padding: 2.2rem 3.2rem 3.5rem; }
    section[data-testid="stSidebar"] {
      background: rgba(9, 11, 26, .84);
      border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] > div { padding: 1.6rem 1.15rem; }

    .brand { display:flex; align-items:center; gap:.7rem; margin: .15rem 0 2.8rem; }
    .brand-mark {
      width: 48px; height: 48px; border-radius: 13px; display:flex; align-items:center; justify-content:center;
      background: linear-gradient(135deg, var(--accent), #6d63ff); color:#fff; font-size: 1.22rem; box-shadow: 0 8px 24px rgba(121,93,255,.32);
    }
    .brand-name { font: 700 1rem 'Space Grotesk', sans-serif; letter-spacing: .14em; }
    .brand-sub { color: var(--muted-2); font-size: .67rem; letter-spacing: .12em; text-transform: uppercase; margin-top: .15rem; }
    .hero-title { font: 700 clamp(2rem, 4vw, 3.6rem)/1.04 'Space Grotesk', sans-serif; letter-spacing:-.055em; margin:0; max-width: 780px; }
    .hero-title span { color: var(--accent); }
    .hero-copy { color: var(--muted); font-size:1.02rem; line-height:1.7; max-width: 720px; margin:1rem 0 1.7rem; }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:24px; padding:1.35rem; box-shadow: 0 24px 70px rgba(0,0,0,.18); backdrop-filter: blur(16px); }
    .panel-heading { display:flex; align-items:center; justify-content:space-between; gap:1rem; padding-bottom:1rem; border-bottom:1px solid var(--line); margin-bottom:1rem; }
    .panel-title { font:600 1rem 'Space Grotesk',sans-serif; letter-spacing:-.02em; }
    .panel-note { color:var(--muted-2); font-size:.74rem; }
    .stButton > button { border-radius:13px; border:1px solid var(--line); background:rgba(255,255,255,.045); color:var(--ink); min-height:2.65rem; transition: all .2s ease; }
    .stButton > button:hover { border-color:rgba(170,140,255,.7); color:#fff; background:rgba(170,140,255,.12); transform:translateY(-1px); }
    .handoff-button > button { border-color:rgba(255,125,146,.45); color:#ffb6c3; background:rgba(255,125,146,.08); }
    .handoff-button > button:hover { background:rgba(255,125,146,.17); border-color:var(--danger); }
    div[data-testid="stChatMessage"] { background:transparent; padding: .75rem 0; }
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { line-height:1.65; }
    .chat-empty { text-align:center; padding: 3.4rem 1.5rem; color:var(--muted); }
    .chat-empty-icon { font-size:2.4rem; color:var(--accent); margin-bottom:.8rem; }
    .chat-empty h3 { color:var(--ink); font:600 1.25rem 'Space Grotesk',sans-serif; margin:.3rem 0 .5rem; }
    .chat-empty p { max-width: 390px; margin:auto; font-size:.9rem; line-height:1.6; }
    .product-shelf { margin-top: .1rem; }
    .shelf-kicker { color: var(--accent-2); font-size: .66rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
    .shelf-title { color: var(--ink); font: 600 1.08rem 'Space Grotesk', sans-serif; margin: .25rem 0 .2rem; }
    .shelf-copy { color: var(--muted); font-size: .76rem; line-height: 1.5; margin-bottom: .85rem; }
    .product-card { position: relative; background: rgba(255,255,255,.035); border: 1px solid rgba(255,255,255,.09); border-radius: 15px; padding: .88rem .95rem; margin: .55rem 0; transition: border-color .2s ease, background .2s ease, transform .2s ease; cursor: pointer; display: flex; gap: 1rem; align-items: stretch; }
    .product-card:hover { background: rgba(255,255,255,.065); border-color: rgba(170,140,255,.45); transform: translateX(2px); }
    .product-card--featured { background: linear-gradient(110deg, rgba(170,140,255,.13), rgba(255,255,255,.035) 70%); border-color: rgba(170,140,255,.3); }
    .product-card--selected { background: rgba(170,140,255,.12); border-color: rgba(170,140,255,.6); box-shadow: 0 0 20px rgba(170,140,255,.15); }
    .product-image { width: 90px; height: 90px; flex-shrink: 0; border-radius: 12px; overflow: hidden; background: #fff; display: flex; align-items: center; justify-content: center; }
    .product-image img { width: 100%; height: 100%; object-fit: cover; }
    .product-body { flex-grow: 1; display: flex; flex-direction: column; justify-content: space-between; }
    .cart-icon-btn { background: #3b5bdb; color: white; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1rem; box-shadow: 0 4px 12px rgba(59,91,219,.3); }
    .product-category { color: var(--accent-2); font-size: .62rem; letter-spacing: .12em; text-transform: uppercase; font-weight: 700; }
    .product-title { font: 600 .92rem 'Space Grotesk', sans-serif; color: var(--ink); margin: .25rem 0 .28rem; line-height: 1.25; }
    .product-description { color: var(--muted); font-size: .73rem; line-height: 1.45; margin-bottom: .55rem; }
    .product-bottom { display:flex; align-items: baseline; justify-content: space-between; gap:.7rem; }
    .product-price { font-size: 1.08rem; font-weight: 700; color: var(--ink); white-space: nowrap; }
    .product-price-old { color: var(--muted-2); text-decoration: line-through; font-size: .7rem; margin-left: .3rem; }
    .product-save { color: var(--accent-2); font-size: .68rem; white-space: nowrap; }
    .product-meta { color: var(--muted); font-size: .68rem; white-space: nowrap; }
    .product-footnote { color: var(--muted-2); font-size: .68rem; padding-top: .65rem; margin-top: .7rem; border-top: 1px solid var(--line); }
    .product-available { color: var(--accent-2); font-size: .62rem; font-weight: 700; letter-spacing: .1em; background: rgba(84,214,197,.08); border: 1px solid rgba(84,214,197,.2); padding: .15rem .45rem; border-radius: 20px; white-space: nowrap; }
    .sidebar-label { color:var(--muted-2); font-size:.7rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; margin:1.3rem 0 .6rem; }
    hr { border-color:var(--line); }
    [data-testid="stChatInput"] { background:rgba(16,19,42,.9); }
    [data-testid="stChatInput"] textarea { color:var(--ink); }

    /* Selection bar */
    .selection-bar { background:rgba(170,140,255,.08); border:1px solid rgba(170,140,255,.25); border-radius:12px; padding:.7rem 1rem; margin-bottom:.5rem; display:flex; align-items:center; justify-content:space-between; gap:.5rem; }
    .selection-text { font-size:.78rem; color:var(--accent); line-height:1.5; }

    /* Chat message bubbles */
    [data-testid="stChatMessageContent"] {
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: .8rem 1rem;
      background: rgba(255,255,255,.045);
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
      background: linear-gradient(135deg, rgba(170,140,255,.20), rgba(170,140,255,.08));
      border-color: rgba(170,140,255,.28);
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
      background: rgba(21, 24, 45, .72);
      border-color: var(--line) !important;
      border-radius: 24px !important;
      padding: 1.15rem 1.25rem !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div { gap: .85rem; }
    [data-testid="stSidebar"] .stButton > button { text-align: left; }

    /* Interaction and responsive polish */
    *:focus-visible { outline: 2px solid var(--accent-2) !important; outline-offset: 3px; }
    .stButton > button, [data-testid="stChatInput"] { min-height: 2.85rem; }
    .stButton > button p { font-size: .82rem; }
    [data-testid="stChatMessage"] { gap: .7rem; }

    /* Make the whole card clickable via invisible Streamlit button */
    div[data-testid="stVerticalBlock"]:has(.is-product-card-container) {
        position: relative;
        gap: 0 !important;
    }
    div[data-testid="stVerticalBlock"]:has(.is-product-card-container) div[data-testid="stButton"] {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: 10;
    }
    div[data-testid="stVerticalBlock"]:has(.is-product-card-container) div[data-testid="stButton"] button {
        width: 100%;
        height: 100%;
        opacity: 0;
        cursor: pointer;
    }
    
    @media (max-width: 900px) {
      .block-container { padding: 1.35rem 1rem 2.6rem; }
      .hero-title { font-size: 2.35rem; }
      .hero-copy { font-size: .94rem; }
      [data-testid="stVerticalBlockBorderWrapper"] { padding: .95rem !important; border-radius: 19px !important; }
    }
    @media (max-width: 640px) {
      .brand { margin-bottom: 1.4rem; }
      .hero-title { font-size: 2rem; }
      [data-testid="stChatMessageContent"] { padding: .7rem .8rem; }
      .product-card { padding: .85rem; }
    }

    </style>
    """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Pipeline loader
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    print("[INFO] Loading product catalogue ...")
    products = load_products_from_csv()
    engine = SalesRetrievalEngine()

    if os.path.exists(DEFAULT_FAISS_PATH) and os.path.exists(DEFAULT_BM25_PATH):
        print("[INFO] Loading search indexes from disk ...")
        engine.load_indexes(products)
    else:
        print("[INFO] Building search indexes ...")
        engine.build_indexes(products)

    pipeline = RAGPipeline(engine)
    print("[SUCCESS] Pipeline ready")
    return products, pipeline


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────
def safe_text(value: Any) -> str:
    return html_lib.escape(str(value or ""))


def clean_product_name_formatting(message: str, products: List[Dict[str, Any]]) -> str:
    """Remove accidental Markdown emphasis around known catalogue names."""
    cleaned = str(message or "")
    for product in products or []:
        title = str(product.get("title", "") or "").strip()
        if not title:
            continue
        escaped_title = re.escape(title)
        patterns = (
            rf"\*{{1,3}}\s*{escaped_title}\s*\*{{1,3}}",
            rf"_{{1,3}}\s*{escaped_title}\s*_{{1,3}}",
            rf"`\s*{escaped_title}\s*`",
        )
        for pattern in patterns:
            cleaned = re.sub(pattern, title, cleaned, flags=re.IGNORECASE)
    return cleaned


def render_product_cards(products: List[Dict[str, Any]], selected_ids: set, unique_prefix: str = "") -> None:
    """Render product cards in the Shopping Brief panel."""
    if not products:
        return

    for index, product in enumerate(products[:5], start=1):
        pid = str(product.get("product_id", ""))
        title = safe_text(product.get("title", "Product"))
        category = safe_text(product.get("category", "Store pick"))
        description = safe_text(product.get("product_description", ""))
        if len(description) > 150:
            description = description[:147].rsplit(" ", 1)[0] + "…"
        price = float(product.get("final_price", 0) or 0)
        initial_price = float(product.get("initial_price", 0) or 0)
        rating = product.get("rating", "—")
        count = product.get("ratings_count", "—")
        savings = max(0, initial_price - price) if initial_price > price else 0

        price_line = f'<span class="product-price">${price:,.2f}</span>'
        if initial_price > price:
            price_line += f'<span class="product-price-old">${initial_price:,.2f}</span>'
        save_line = f'<span class="product-save">Save ${savings:,.2f}</span>' if savings else ''

        images_str = product.get("images", "")
        image_url = images_str.split(",")[0].strip('"\'') if images_str else ""
        image_html = f'<div class="product-image"><img src="{image_url}" onerror="this.style.display=\'none\'" /></div>' if image_url else '<div class="product-image" style="background: rgba(255,255,255,0.05);"></div>'

        is_selected = pid in selected_ids
        card_class = "product-card"
        if index == 1:
            card_class += " product-card--featured"
        if is_selected:
            card_class += " product-card--selected"
            
        btn_label = "✅ Selected" if is_selected else "💬 Ask More"
        label_color = "#4ade80" if is_selected else "#3b5bdb"
        label_bg = "rgba(74,222,128,0.1)" if is_selected else "rgba(59,91,219,0.1)"

        with st.container():
            st.markdown('<div class="is-product-card-container"></div>', unsafe_allow_html=True)
            st.markdown(
                f'''<div class="{card_class}">
                    {image_html}
                    <div class="product-body">
                        <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                            <div class="product-title" style="margin:0;">{title}</div>
                            <span class="product-available">Available</span>
                        </div>
                        <div class="product-description" style="margin: 0.2rem 0 0.5rem 0;">{description or "A relevant pick from the current catalogue."}</div>
                        <div class="product-bottom" style="align-items:center; display:flex; justify-content:space-between;">
                            <div style="color: #3b5bdb; font-weight: 700; font-size: 1.1rem;">{price_line}</div>
                            <div style="font-weight: 600; color: {label_color}; background: {label_bg}; padding: 0.25rem 0.8rem; border-radius: 20px; font-size: 0.85rem;">
                                {btn_label}
                            </div>
                        </div>
                    </div>
                </div>''',
                unsafe_allow_html=True,
            )
            # Invisible Ask More button overlaid on the card
            btn_key = f"ask_{unique_prefix}_{pid}_{index}" if unique_prefix else f"ask_{pid}_{index}"
            if st.button("invisible", key=btn_key, use_container_width=True):
                if is_selected:
                    st.session_state.selected_products.discard(pid)
                else:
                    st.session_state.selected_products.add(pid)
                st.rerun()

    st.markdown(
        '<div class="product-footnote">Prices and ratings are taken from the current catalogue.</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  State
# ──────────────────────────────────────────────────────────────────────────────
def reset_conversation() -> None:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.current_products = []
    st.session_state.selected_products = set()
    st.session_state.frustration_count = 0
    st.session_state.last_result = None


def initialize_state() -> None:
    if "session_id" not in st.session_state:
        reset_conversation()
    if "product_lookup" not in st.session_state:
        st.session_state.product_lookup = {}


# ──────────────────────────────────────────────────────────────────────────────
#  Message processing
# ──────────────────────────────────────────────────────────────────────────────
def process_user_message(text: str, pipeline) -> None:
    text = (text or "").strip()
    if not text:
        return

    st.session_state.messages.append({"id": str(uuid.uuid4()), "role": "user", "content": text})

    # Build backend prompt with product context injection
    backend_prompt = text
    if st.session_state.selected_products:
        context_parts = []
        for pid in st.session_state.selected_products:
            p = st.session_state.product_lookup.get(pid)
            if p:
                context_parts.append(f"'{p['title']}' (ID: {pid})")
        if context_parts:
            context_prefix = f"[Context: User is asking about these selected products: {', '.join(context_parts)}]\n\n"
            backend_prompt = context_prefix + text

    # Dual-strike frustration logic
    suppress_auto = False
    handoff_reason = pipeline.handoff_policy.evaluate(text)
    if handoff_reason:
        if st.session_state.frustration_count == 0:
            st.session_state.frustration_count += 1
            suppress_auto = True  # First strike — let LLM answer normally
        # else: second strike — let handoff happen

    with st.spinner("SmartSales is thinking…"):
        result = pipeline.process_message(
            backend_prompt,
            session_id=st.session_state.session_id,
            suppress_auto_handoff=suppress_auto,
        )

    reply = result.get("message", "")

    # Second strike: append suggestion
    if result.get("type") == "handoff" and st.session_state.frustration_count >= 1:
        reply = (
            "I notice you might be having a difficult experience. "
            "Would you like me to connect you with a human representative?\n\n"
            + reply
        )
        st.session_state.frustration_count = 0

    reply = clean_product_name_formatting(reply, result.get("products", []))

    st.session_state.messages.append({
        "id": str(uuid.uuid4()),
        "role": "assistant", 
        "content": reply,
        "products": result.get("products", [])
    })
    st.session_state.current_products = result.get("products", [])
    st.session_state.last_result = result

    # Update product lookup with any new products
    for p in st.session_state.current_products:
        pid = str(p.get("product_id", ""))
        if pid:
            st.session_state.product_lookup[pid] = p


def process_handoff(pipeline) -> None:
    st.session_state.messages.append(
        {"id": str(uuid.uuid4()), "role": "user", "content": "I want to talk to a human representative"}
    )
    with st.spinner("Preparing your conversation for a representative…"):
        result = pipeline.process_message(
            "I want to talk to a human representative",
            session_id=st.session_state.session_id,
            request_handoff=True,
        )
    st.session_state.messages.append(
        {"id": str(uuid.uuid4()), "role": "assistant", "content": result.get("message", ""), "products": []}
    )
    st.session_state.current_products = []
    st.session_state.last_result = result


# ──────────────────────────────────────────────────────────────────────────────
#  Main render
# ──────────────────────────────────────────────────────────────────────────────
def render_app():
    setup_page()
    initialize_state()
    products_db, pipeline = load_pipeline()

    # Build product lookup from DB (once)
    if not st.session_state.product_lookup:
        for p in products_db:
            st.session_state.product_lookup[str(p.get("product_id"))] = p

    # ── Sidebar ──────────────────────────────────────────────────────────────
    import base64
    import os
    icon_path = os.path.join(os.path.dirname(__file__), "..", "icon.png")
    try:
        with open(icon_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        img_html = f'<img src="data:image/png;base64,{encoded_string}" style="width:100%; height:100%; object-fit:contain; border-radius:13px;" />'
    except Exception:
        img_html = '✦'

    with st.sidebar:
        st.markdown(
            f'<div class="brand"><div class="brand-mark">{img_html}</div><div>'
            '<div class="brand-name">SMARTSALES</div>'
            '<div class="brand-sub">AI shopping concierge</div>'
            "</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-label">Start with a shortcut</div>', unsafe_allow_html=True)
        quick_prompts = {
            "Explore trending picks": "What are the best trending products right now?",
            "Find budget-friendly products": "Show me budget-friendly products under $20",
            "Browse categories": "What categories of products do you have?",
            "Explore installment plans": "What installment plans do you offer?",
        }
        for label, prompt in quick_prompts.items():
            if st.button(label, use_container_width=True, key=f"quick_{label}"):
                process_user_message(prompt, pipeline)
                st.rerun()

        st.markdown('<div class="sidebar-label">Need a person?</div>', unsafe_allow_html=True)
        st.markdown('<div class="handoff-button">', unsafe_allow_html=True)
        if st.button("Connect me with a representative", use_container_width=True):
            process_handoff(pipeline)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="sidebar-label">Session</div>', unsafe_allow_html=True)
        if st.button("Start a fresh conversation", use_container_width=True):
            reset_conversation()
            st.rerun()

    # ── Main workspace ────────────────────────────────────────────────────────
    st.markdown(
        '<h1 class="hero-title">RAG-based sales representative <span>Chatbot</span></h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-copy">Ask about products, compare options, or explore payment plans.</p>',
        unsafe_allow_html=True,
    )

    # Selection indicator
    if st.session_state.selected_products:
        selected_titles = []
        for pid in list(st.session_state.selected_products):
            p = st.session_state.product_lookup.get(pid)
            if p:
                selected_titles.append(p.get("title", "Unknown"))
        if selected_titles:
            sel_col, clear_col = st.columns([8, 2])
            with sel_col:
                st.markdown(
                    f'<div class="selection-bar"><div class="selection-text">'
                    f'🔗 <strong>Asking about:</strong> {", ".join(selected_titles)}</div></div>',
                    unsafe_allow_html=True,
                )
            with clear_col:
                if st.button("Clear Selection", use_container_width=True):
                    st.session_state.selected_products = set()
                    st.rerun()

    # Two-column layout: Chat | Shopping Brief
    chat_col, insight_col = st.columns([1.55, 1], gap="large")

    with chat_col:
        with st.container(border=True):
            if not st.session_state.messages:
                empty_img = img_html.replace('width:100%; height:100%', 'width:100px; height:80px') if '<img' in img_html else '✦'
                st.markdown(
                    f'<div class="chat-empty"><div class="chat-empty-icon" style="display:flex; justify-content:center; align-items:center;">{empty_img}</div>'
                    "<h3>What can I help you discover?</h3>"
                    "<p>Ask for a recommendation, a price comparison, "
                    "or the exact monthly cost of a product.</p></div>",
                    unsafe_allow_html=True,
                )
            else:
                for i, message in enumerate(st.session_state.messages):
                    with st.chat_message(message["role"]):
                        col1, col2 = st.columns([0.9, 0.1])
                        with col1:
                            if message["role"] == "assistant":
                                msg_id = message.get("id", "")
                                st.markdown(f'<div id="resp-{msg_id}" class="sync-resp"></div>', unsafe_allow_html=True)
                            st.markdown(message["content"])
                        with col2:
                            copy_button(message["content"], key=f"copy_btn_{i}")

        user_prompt = st.chat_input("Ask about a product, price, or installment plan…")
        if user_prompt:
            process_user_message(user_prompt, pipeline)
            st.rerun()

    with insight_col:
        with st.container(border=True):
            has_cards = any(msg.get("products") for msg in st.session_state.messages if msg["role"] == "assistant")
            if has_cards:
                for i, msg in enumerate(st.session_state.messages):
                    if msg["role"] == "assistant" and msg.get("products"):
                        msg_id = msg.get("id", "")
                        with st.container():
                            st.markdown(f'<div id="card-{msg_id}" class="sync-card"></div>', unsafe_allow_html=True)
                            render_product_cards(msg["products"], st.session_state.selected_products, unique_prefix=msg_id)
            else:
                st.markdown(
                    '<div class="panel-heading"><div>'
                    '<div class="panel-title">Your shopping brief</div>'
                    '<div class="panel-note">A quieter view of the details</div>'
                    '</div><div style="color:var(--accent);font-size:1.2rem">◌</div></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="product-shelf"><div class="shelf-kicker">SHORTLIST</div>'
                    '<div class="shelf-title">Your shopping brief</div>'
                    '<div class="shelf-copy">Products mentioned in the conversation</div></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div style="text-align:center;padding:2rem;color:var(--muted)">'
                    "<p>I will display relevant products here as we chat.</p></div>",
                    unsafe_allow_html=True,
                )

    import streamlit.components.v1 as components
    import time
    js_code = f"""
    <script>
    // Execution ID: {time.time()}
    const parent = window.parent.document;
    
    // --- 1. Card Alignment Logic ---
    function alignCards() {{
        try {{
            const resps = parent.querySelectorAll('.sync-resp');
            if (resps.length === 0) return;
            
            resps.forEach(resp => {{
                const id = resp.id.replace('resp-', '');
                const cardMarker = parent.getElementById('card-' + id);
                if (resp && cardMarker) {{
                    const respRect = resp.getBoundingClientRect();
                    const block = cardMarker.closest('[data-testid="stVerticalBlock"]');
                    if (!block) return;
                    
                    const blockRect = block.getBoundingClientRect();
                    if (blockRect.top < respRect.top) {{
                        const diff = respRect.top - blockRect.top;
                        const currentMargin = parseFloat(window.getComputedStyle(block).marginTop) || 0;
                        block.style.marginTop = (currentMargin + diff) + 'px';
                    }}
                }}
            }});
        }} catch(e) {{}}
    }}
    
    setTimeout(alignCards, 50);
    setTimeout(alignCards, 300);
    setTimeout(alignCards, 1000);
    setTimeout(alignCards, 2000);
    
    // --- 2. Immediate Card Selection UI ---
    // Ensure we only attach the listener once to avoid multiple toggles
    if (!parent.window.smartSalesUIInitialized) {{
        parent.addEventListener('click', function(e) {{
            // Find if the click is on an invisible button over a card
            const block = e.target.closest('[data-testid="stVerticalBlock"]');
            if (!block) return;
            
            const isCardBlock = block.querySelector('.is-product-card-container');
            const isButton = e.target.closest('button');
            
            if (isCardBlock && isButton) {{
                const card = block.querySelector('.product-card');
                if (card) {{
                    // Immediately toggle CSS class for instant visual feedback
                    card.classList.toggle('product-card--selected');
                    
                    // Immediately update the label
                    const labelBtn = card.querySelector('.product-bottom > div:last-child');
                    if (labelBtn) {{
                        if (card.classList.contains('product-card--selected')) {{
                            labelBtn.innerHTML = "✅ Selected";
                            labelBtn.style.color = "#4ade80";
                            labelBtn.style.background = "rgba(74,222,128,0.1)";
                        }} else {{
                            labelBtn.innerHTML = "💬 Ask More";
                            labelBtn.style.color = "#3b5bdb";
                            labelBtn.style.background = "rgba(59,91,219,0.1)";
                        }}
                    }}
                }}
            }}
        }});
        parent.window.smartSalesUIInitialized = true;
    }}
    </script>
    """
    components.html(js_code, height=0)


# Alias for backward compatibility
create_app = render_app
