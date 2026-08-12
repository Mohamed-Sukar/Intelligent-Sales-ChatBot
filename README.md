# Intelligent Sales ChatBot

An end-to-end AI-powered **Sales Representative ChatBot** for an e-commerce company. The bot acts as the company's front-line sales agent: it showcases products & prices, answers product inquiries, suggests installment plans, recommends complementary products (cross-sell/up-sell), and seamlessly escalates to a human agent when needed.

## Features

- **Conversational AI with Dual-API Architecture**: Uses OpenRouter (Primary) and Google Gemini (Backup) via LangChain for 100% uptime and cost optimization.
- **Hybrid Product Search (RAG)**: Combines Semantic Search (FAISS) and Keyword Search (BM25) to retrieve the most relevant products from an SQLite database.
- **Intent & Sentiment Analysis**: Built-in rule-based intent classification and VADER sentiment analysis to detect user frustration and automatically escalate to a human agent.
- **Smart Recommendations**: Cross-sell and up-sell suggestions based on product categories, prices, and ratings.
- **Installment Calculator**: Built-in calculator to suggest payment plans (3, 6, 12 months) with interest rates.
- **Gradio UI**: Interactive chat interface with product cards, filters, and product summaries.
- **Admin Panel**: Password-protected dashboard to add, edit, and delete products, seamlessly auto-updating the FAISS/BM25 indexes.

## System Architecture

- **Frontend**: Gradio (Chat UI, Product Cards, Admin Panel)
- **Backend**: Python
- **Database**: SQLite (Products, Conversations, Admin logs)
- **Vector Search**: FAISS + rank-bm25 (Semantic + keyword search)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLMs**: OpenRouter API + LangChain ChatOpenAI (Primary), Google Gemini API (Backup)

## Project Structure

```text
Intelligent Sales ChatBot/
│
├── .env                       # API Keys (OPENROUTER_API_KEY, GEMINI_API_KEY, ADMIN_PASSWORD)
├── .gitignore                 
├── requirements.txt           
├── app.py                     # Entry point (Gradio Frontend)
│
├── data/                      # Raw dataset
│   └── shopping_data.csv      
│
├── database/                  # SQLite DB & Search Indexes
│   ├── chatbot.db             
│   ├── faiss.index            
│   └── bm25_index.pkl         
│
├── core/                      # Backend & AI Logic
│   ├── data_pipeline.py       # Data cleaning & SQLite setup
│   ├── search_engine.py       # FAISS, BM25 & Hybrid Search
│   ├── nlp_analyzer.py        # Intent, Sentiment & Handoff logic
│   ├── llm_manager.py         # Prompt Builder & Dual-API manager
│   └── business_logic.py      # Memory, Installments & Admin ops
│
└── notebooks/                 # Jupyter notebooks for testing
```

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd "Intelligent Sales ChatBot"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables:**
   Create a `.env` file in the root directory and add your API keys:
   ```env
   OPENROUTER_API_KEY= your_openrouter_key
   GEMINI_API_KEY=     your_gemini_key
   ADMIN_PASSWORD=     your_secure_password  --> to update the products in database.
   ```

4. **Prepare the Data & Indexes:**
   Run the data pipeline and search engine scripts to initialize the database and FAISS/BM25 indexes from your raw CSV.
   ```bash
   python core/data_pipeline.py
   python core/search_engine.py
   ```

5. **Run the Application:**
   ```bash
   python app.py
   ```

6. **Access the ChatBot:**
   Open your browser and navigate to the local Gradio server URL provided in the terminal (usually `http://localhost:7860`).

## Team Roles

- **Data Engineer**: Data preparation, cleaning, and SQLite database management.
- **Search & RAG Engineer**: Hybrid search implementation (FAISS + BM25) and recommendations.
- **NLP Engineer**: Intent classification, sentiment analysis (VADER), and handoff logic.
- **LLM & RAG Pipeline Engineer**: Dual-API LLM integration and prompt engineering.
- **Business Logic & Memory Engineer**: Conversational memory, installment calculations, and admin backend.
- **Frontend (Collaborative)**: Gradio UI implementation.

## Dataset

This project uses the [Ecommerce Dataset (Products & Sizes Included)](https://www.kaggle.com/datasets/anvitkumar/shopping-dataset) from Kaggle, containing 1,000+ real e-commerce products with rich metadata including prices, ratings, and customer reviews.
