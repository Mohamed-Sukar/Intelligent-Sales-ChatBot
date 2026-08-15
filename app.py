"""
SmartSales Bot — Gradio Deployment
===================================
A modern, fully-styled Gradio chatbot UI wrapping the RAG pipeline. Run with:

    python app.py

Requires:
    pip install gradio faiss-cpu sentence-transformers rank-bm25 pandas \
                torch transformers langchain-openai langchain-core \
                requests python-dotenv
"""

from core.ui import create_app

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
