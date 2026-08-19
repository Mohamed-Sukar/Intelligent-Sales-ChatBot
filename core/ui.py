from __future__ import annotations

import html as html_lib
import re
import uuid
from typing import Any, Dict, List

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
    icon0_path = os.path.join(os.path.dirname(__file__), "..", "data", "icon0.png")
    st.set_page_config(
        page_title="SmartSales | AI Shopping Concierge",
        page_icon=icon0_path if os.path.exists(icon0_path) else "icon0.png",
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

    .brand { display:flex; align-items:center; gap:1.8rem; margin: .15rem 0 2.8rem; }
    .brand-mark {
      width: 48px; height: 48px; display:flex; align-items:center; justify-content:center;
    }
    .brand-mark img {
      transform: scale(1.6);
    }
    .brand-name { font: 700 1rem 'Space Grotesk', sans-serif; letter-spacing: .14em; }
    .brand-sub { color: var(--muted-2); font-size: .67rem; letter-spacing: .12em; text-transform: uppercase; margin-top: .15rem; }
    .hero-title { font: 700 clamp(2rem, 4vw, 3.6rem)/1.04 'Space Grotesk', sans-serif; letter-spacing:-.055em; margin:0; max-width: 780px; }
    .hero-title span { color: var(--accent); }
    .hero-copy { color: var(--muted); font-size:1.02rem; line-height:1.7; max-width: 720px; margin:1rem 0 1.7rem; }
    .panel-heading { display:flex; align-items:center; justify-content:space-between; gap:1rem; padding-bottom:1rem; border-bottom:1px solid var(--line); margin-bottom:1rem; }
    .panel-title { font:600 1rem 'Space Grotesk',sans-serif; letter-spacing:-.02em; }
    .panel-note { color:var(--muted-2); font-size:.74rem; }
    .stButton > button { border-radius:13px; border:1px solid var(--line); background:rgba(255,255,255,.045); color:var(--ink); min-height:2.65rem; transition: all .2s ease; }
    .stButton > button:hover { border-color:rgba(170,140,255,.7); color:#fff; background:rgba(170,140,255,.12); transform:translateY(-1px); }
    .handoff-button > button { border-color:rgba(255,125,146,.45); color:#ffb6c3; background:rgba(255,125,146,.08); }
    .handoff-button > button:hover { background:rgba(255,125,146,.17); border-color:var(--danger); }
    div[data-testid="stChatMessage"] { background:transparent; padding: .75rem 0; }
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { line-height: 1.7; }
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] ul { margin-bottom: 1rem; }
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li { margin-bottom: 0.6rem; }
    .chat-empty { text-align:center; padding: 3.4rem 1.5rem; color:var(--muted); }
    .chat-empty-icon { font-size:2.4rem; color:var(--accent); margin-bottom:.8rem; }
    .chat-empty h3 { color:var(--ink); font:600 1.25rem 'Space Grotesk',sans-serif; margin:.3rem 0 .5rem; }
    .chat-empty p { max-width: 390px; margin:auto; font-size:.9rem; line-height:1.6; }
    .product-shelf { margin-top: .1rem; }
    .shelf-kicker { color: var(--accent-2); font-size: .66rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
    .shelf-title { color: var(--ink); font: 600 1.08rem 'Space Grotesk', sans-serif; margin: .25rem 0 .2rem; }
    .shelf-copy { color: var(--muted); font-size: .76rem; line-height: 1.5; margin-bottom: .85rem; }
    .product-card { position: relative; background: rgba(255,255,255,.035); border: 1px solid rgba(255,255,255,.09); border-radius: 15px; padding: .88rem .95rem .88rem .95rem; margin: .55rem 0 0 0; transition: border-color .2s ease, background .2s ease, transform .2s ease; cursor: pointer; display: flex; gap: 1rem; align-items: stretch; }
    .product-card:hover { background: rgba(255,255,255,.065); border-color: rgba(170,140,255,.45); transform: translateX(2px); }
    .product-card--featured { background: linear-gradient(110deg, rgba(170,140,255,.13), rgba(255,255,255,.035) 70%); border-color: rgba(170,140,255,.3); }
    .product-card--selected { background: rgba(170,140,255,.12); border-color: rgba(170,140,255,.6); box-shadow: 0 0 20px rgba(170,140,255,.15); }
    .product-image { width: 90px; height: 90px; flex-shrink: 0; border-radius: 12px; overflow: hidden; background: #fff; display: flex; align-items: center; justify-content: center; }
    .product-image img { width: 100%; height: 100%; object-fit: cover; }
    .product-body { flex-grow: 1; display: flex; flex-direction: column; justify-content: space-between; }
    .product-title { font: 600 .92rem 'Space Grotesk', sans-serif; color: var(--ink); margin: .25rem 0 .28rem; line-height: 1.25; }
    .product-description { color: var(--muted); font-size: .73rem; line-height: 1.45; margin-bottom: .55rem; }
    .product-bottom { display:flex; align-items: center; justify-content: space-between; gap:.7rem; }
    .product-price { font-size: 1.08rem; font-weight: 700; color: var(--ink); white-space: nowrap; }
    .product-price-old { color: var(--muted-2); text-decoration: line-through; font-size: .7rem; margin-left: .3rem; }
    .product-save { color: var(--accent-2); font-size: .68rem; white-space: nowrap; }
    .product-footnote { color: var(--muted-2); font-size: .68rem; padding-top: .65rem; margin-top: .7rem; border-top: 1px solid var(--line); }
    .product-available { color: var(--accent-2); font-size: .62rem; font-weight: 700; letter-spacing: .1em; background: rgba(84,214,197,.08); border: 1px solid rgba(84,214,197,.2); padding: .15rem .45rem; border-radius: 20px; white-space: nowrap; }
    .sidebar-label { color:var(--muted-2); font-size:.7rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; margin:1.3rem 0 .6rem; }
    hr { border-color:var(--line); }
    [data-testid="stChatInput"] { background:rgba(16,19,42,.9); }
    [data-testid="stChatInput"] textarea { color:var(--ink); }

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

    /* Product card with overlapping Ask More button */
    .product-card-wrapper {
        margin-bottom: .55rem;
    }
    .product-card-wrapper > div[data-testid="element-container"]:has(.pc-marker) {
        display: none !important;
    }
    .product-card-wrapper div[data-testid="stButton"] {
        margin-top: -22px;
        display: flex !important;
        width: 100% !important;
        justify-content: flex-end !important;
        padding-right: 14px;
        position: relative;
        z-index: 10;
    }
    .product-card-wrapper div[data-testid="stButton"] button {
        background: rgba(30, 34, 60, .92) !important;
        border: 1px solid rgba(255,255,255,.12) !important;
        border-radius: 25px !important;
        padding: .45rem 1.4rem !important;
        color: var(--ink) !important;
        font-size: .82rem !important;
        min-height: 2.3rem !important;
        width: auto !important;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 15px rgba(0,0,0,.3);
        transition: all .25s ease !important;
        white-space: nowrap;
    }
    .product-card-wrapper div[data-testid="stButton"] button:hover {
        background: rgba(170,140,255,.18) !important;
        border-color: rgba(170,140,255,.5) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(170,140,255,.2) !important;
    }
    /* Selected state for overlap button */
    .product-card-wrapper.card-selected div[data-testid="stButton"] button {
        background: rgba(170,140,255,.22) !important;
        border-color: rgba(170,140,255,.6) !important;
        color: #fff !important;
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


def toggle_selection(product_id, product_title):
    if product_id in st.session_state.selected_products:
        del st.session_state.selected_products[product_id]
    else:
        st.session_state.selected_products[product_id] = product_title


def clear_all_selected_products() -> None:
    """Clear all currently selected products."""
    if "selected_products" in st.session_state:
        st.session_state.selected_products = {}


def render_product_cards(products: List[Dict[str, Any]], unique_prefix: str = "") -> None:
    """Render product cards in the Shopping Brief panel."""
    if not products:
        return

    for index, product in enumerate(products[:5], start=1):
        pid = str(product.get("product_id", ""))
        title = safe_text(product.get("title", "Product"))
        description = safe_text(product.get("product_description", ""))
        if len(description) > 150:
            description = description[:147].rsplit(" ", 1)[0] + "…"
        price = float(product.get("final_price", 0) or 0)
        initial_price = float(product.get("initial_price", 0) or 0)
        price_line = f'<span class="product-price">${price:,.2f}</span>'
        if initial_price > price:
            price_line += f'<span class="product-price-old">${initial_price:,.2f}</span>'
        images_str = product.get("images", "")
        image_url = images_str.split(",")[0].strip("\"'") if images_str else ""
        image_html = f'<div class="product-image"><img src="{image_url}" onerror="this.style.display=\'none\'" /></div>' if image_url else '<div class="product-image" style="background: rgba(255,255,255,0.05);"></div>'

        is_selected = pid in st.session_state.selected_products
        card_class = "product-card"
        if index == 1:
            card_class += " product-card--featured"
        if is_selected:
            card_class += " product-card--selected"
            
        btn_label = "Selected ✅" if is_selected else "Ask More 💬"
        btn_type = "primary" if is_selected else "secondary"
        wrapper_class = "product-card-wrapper card-selected" if is_selected else "product-card-wrapper"

        with st.container():
            # Marker div for JS to identify and wrap this container
            st.markdown(
                f'<div class="pc-marker" data-wrapper-class="{wrapper_class}" style="display:none;"></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""<div class="{card_class}" style="padding-bottom: 1.6rem;">
                    {image_html}
                    <div class="product-body">
                        <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                            <div class="product-title" style="margin:0;">{title}</div>
                            <span class="product-available">Available</span>
                        </div>
                        <div class="product-description" style="margin: 0.2rem 0 0.5rem 0;">{description or "A relevant pick from the current catalogue."}</div>
                        <div class="product-bottom" style="align-items:center; display:flex; justify-content:space-between;">
                            <div style="color: #3b5bdb; font-weight: 700; font-size: 1.1rem;">{price_line}</div>
                        </div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
            btn_key = f"ask_{unique_prefix}_{pid}_{index}" if unique_prefix else f"ask_{pid}_{index}"
            st.button(
                label=btn_label,
                key=btn_key,
                on_click=toggle_selection,
                args=(pid, title),
                type=btn_type,
                use_container_width=False
            )

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
    st.session_state.selected_products = {}
    st.session_state.frustration_count = 0


def initialize_state() -> None:
    if "session_id" not in st.session_state:
        reset_conversation()


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
        selected_titles_str = ", ".join(st.session_state.selected_products.values())
        injected_context = f"\n\n[System Note: The user has selected these products from the UI for context: {selected_titles_str}]\n"
        backend_prompt = text + injected_context

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


# ──────────────────────────────────────────────────────────────────────────────
#  Main render
# ──────────────────────────────────────────────────────────────────────────────
def render_app():
    setup_page()
    initialize_state()
    products_db, pipeline = load_pipeline()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    import base64
    icon_path = os.path.join(os.path.dirname(__file__), "..", "data", "icon.png")
    if not os.path.exists(icon_path):
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
            '<div class="brand-name">  SMARTSALES</div>'
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

        if st.session_state.selected_products:
            col_text, col_btn = st.columns([0.83, 0.17])
            with col_text:
                selected_titles = list(st.session_state.selected_products.values())
                st.markdown(
                    f"<div style='padding-top: 5px; font-size: 0.85rem; color: var(--muted);'>"
                    f"<strong style='color: var(--accent);'>Selected:</strong> {', '.join(selected_titles)}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with col_btn:
                st.markdown('<div class="unselect-btn">', unsafe_allow_html=True)
                st.button(
                    "Unselect All",
                    key="clear_all_selected",
                    on_click=clear_all_selected_products,
                    help="Remove all selected products",
                    use_container_width=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)

        user_prompt = st.chat_input("Ask about a product, price, or installment plan…")
        if user_prompt:
            process_user_message(user_prompt, pipeline)
            st.rerun()

    with insight_col:
        with st.container(border=True):
            st.markdown(
                '<div class="panel-heading"><div>'
                '<div class="panel-title">Your shopping brief</div>'
                '<div class="panel-note">A quieter view of the details</div>'
                '</div><div style="color:var(--accent);font-size:1.2rem">◌</div></div>',
                unsafe_allow_html=True,
            )
            
            has_cards = any(msg.get("products") for msg in st.session_state.messages if msg["role"] == "assistant")
            if has_cards:
                for i, msg in enumerate(st.session_state.messages):
                    if msg["role"] == "assistant" and msg.get("products"):
                        msg_id = msg.get("id", "")
                        with st.container():
                            st.markdown(f'<div id="card-{msg_id}" class="sync-card"></div>', unsafe_allow_html=True)
                            render_product_cards(msg["products"], unique_prefix=msg_id)
            else:
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
    function wrapCards() {{
        try {{
            const parent = window.parent.document;
            const markers = parent.querySelectorAll('.pc-marker');
            markers.forEach(marker => {{
                // The marker is inside a stMarkdownContainer -> stMarkdown -> element -> stVerticalBlock
                // We need to find the st.container() wrapper that holds the marker, card HTML, and button
                const container = marker.closest('[data-testid="stVerticalBlock"]');
                if (!container) return;
                const wrapperClass = marker.getAttribute('data-wrapper-class') || 'product-card-wrapper';
                if (!container.classList.contains('product-card-wrapper')) {{
                    wrapperClass.split(' ').forEach(cls => container.classList.add(cls));
                    // Remove gap from the container so card and button are tight together
                    container.style.gap = '0';
                }}
            }});
        }} catch(e) {{}}
    }}

    function alignCards() {{
        try {{
            const parent = window.parent.document;
            const resps = parent.querySelectorAll('.sync-resp');
            if (resps.length === 0) return;
            
            resps.forEach(resp => {{
                const id = resp.id.replace('resp-', '');
                const cardMarker = parent.getElementById('card-' + id);
                if (resp && cardMarker) {{
                    const respRect = resp.getBoundingClientRect();
                    
                    // Find the container that holds both the marker and the rendered cards
                    const block = cardMarker.closest('[data-testid="stVerticalBlock"]');
                    if (!block) return;
                    
                    const blockRect = block.getBoundingClientRect();
                    
                    // We only push down. If it's already lower (due to previous cards), we leave it.
                    if (blockRect.top < respRect.top) {{
                        const diff = respRect.top - blockRect.top;
                        const currentMargin = parseFloat(window.getComputedStyle(block).marginTop) || 0;
                        block.style.marginTop = (currentMargin + diff) + 'px';
                    }}
                }}
            }});
        }} catch(e) {{}}
    }}
    
    function runAll() {{ wrapCards(); alignCards(); }}
    setTimeout(runAll, 50);
    setTimeout(runAll, 300);
    setTimeout(runAll, 800);
    setTimeout(runAll, 1500);
    </script>
    """
    components.html(js_code, height=0)

