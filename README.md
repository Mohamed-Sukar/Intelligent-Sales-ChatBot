# Intelligent Sales ChatBot

An end-to-end AI-powered **Sales Representative ChatBot** for an e-commerce company. The bot acts as the company's front-line sales agent: it showcases products & prices, answers product inquiries, calculates installment plans, and seamlessly escalates to a human agent when requested.

## 🚀 Key Updates & Current Architecture

This project has been heavily refactored for efficiency, removing complex databases and multi-file structures in favor of a fast, single-file orchestrator design.

### Features

- **Modular Orchestrator**: The entire backend logic (Retrieval, LLM Management, Memory, Business Logic) and frontend UI are elegantly structured in a professional `core/` package, with a simple `app.py` entry point.
- **Dual-API LLM Architecture**: Uses Google Gemini API (Primary) via direct POST request, and automatically falls back to OpenRouter (Backup) via LangChain ensuring maximum uptime.
- **Hybrid Product Search (RAG)**: Combines Dense Semantic Search (FAISS + `all-MiniLM-L6-v2`) and Sparse Keyword Search (BM25) to retrieve the most relevant products directly from an in-memory dataset, merged via Reciprocal Rank Fusion (RRF).
- **Ghost-Card Filtering**: The system actively parses the LLM's response to ensure only products explicitly mentioned by the AI are rendered as HTML cards in the UI.
- **Smart Context & Memory**: The bot injects the user's recent chat history and previously viewed products into the LLM prompt to maintain a natural, context-aware conversation.
- **Installment Calculator**: Dynamic calculations for 3, 6, and 12-month payment plans, seamlessly injected into the prompt.
- **Rule-Based Human Handoff**: Detects specific keywords (e.g., "talk to human", "خدمة العملاء") to seamlessly transition the chat to a human agent, generating an automatic "Agent Briefing Summary".
- **Modern Streamlit UI**: Interactive chat interface with custom dark-theme CSS, HTML product cards (showing ratings, discounts, and prices), and quick-action chips.

## 📂 Project Structure

```text
Intelligent Sales ChatBot/
│
├── .env                       # API Keys (OPEN_ROUTER_KEY, GEMINI_API_KEY)
├── .env.example               # Example environment variables file
├── app.py                     # Entry point for Streamlit Frontend
│
├── core/                      # Backend package
│   ├── config.py              # Environment variables & constants
│   ├── data_loader.py         # CSV loading logic
│   ├── retrieval.py           # FAISS + BM25 search engine
│   ├── llm.py                 # LLM API connections
│   ├── pipeline.py            # RAG orchestration, memory, and handoff
│   └── ui.py                  # Streamlit frontend components
│
├── data/                      # Dataset directory
│   └── products_clean.csv     # Cleaned e-commerce dataset
│
├── database/                  # Auto-generated Search Indexes
│   ├── faiss.index            # FAISS dense search index
│   └── bm25_index.pkl         # Pickled BM25 keyword search index
│
├── requirements.txt           # Project dependencies
└── notebooks/                 # Jupyter notebooks for data exploration and testing
    ├── 01_data_exploration.ipynb
    └── 02_full_pipeline.ipynb
```

## ⚙️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd "Intelligent Sales ChatBot"
   ```

2. **Install dependencies:**
   Make sure you have Python installed, then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables:**
   Create a `.env` file in the root directory (or rename `.env.example`) and add your API keys. Make sure to use the exact variable names expected by `app.py`:
   ```env
   OPEN_ROUTER_KEY=your_openrouter_key
   GEMINI_API_KEY=your_gemini_key
   ```

4. **Run the Application:**
   Unlike previous versions, there is no need to manually run data pipelines. The script will automatically load `products_clean.csv` and build the FAISS and BM25 indexes on startup.
   ```bash
   streamlit run app.py
   ```

5. **Access the ChatBot:**
   Open your browser and navigate to the local Streamlit server URL provided in the terminal (usually `http://localhost:8501`).

## 📊 Dataset

This project is built on top of a highly cleaned and preprocessed version of the [Shopping Dataset](https://www.kaggle.com/datasets/anvitkumar/shopping-dataset) from Kaggle. The raw data has been filtered for duplicates, prices have been normalized to USD, and HTML noise has been stripped from product descriptions to optimize the embeddings and LLM performance. The cleaned version resides in `products_clean.csv`.
