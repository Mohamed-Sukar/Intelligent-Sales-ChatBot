import sqlite3
from typing import List, Dict, Any, Optional

class ProductRecommender:
    """
    Handles Cross-Selling, Product Summaries, and Review Summaries 
    aligned with the database schema built in data_pipeline.py.
    """
    def __init__(self, db_path: str = "database/chatbot.db"):
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        """Establishes a connection to the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_recommendations(self, product_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Generates Cross-Sell recommendations based on:
        1. Same category as target product.
        2. Final price within ±30% range.
        3. Active products sorted by highest rating.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Step 1: Retrieve target product category and final_price
        cursor.execute(
            "SELECT category, final_price FROM products WHERE product_id = ? AND is_active = 1", 
            (str(product_id),)
        )
        target = cursor.fetchone()

        if not target:
            conn.close()
            return []

        category = target['category']
        price = target['final_price']

        # Calculate ±30% price range
        min_price = price * 0.70
        max_price = price * 1.30

        # Step 2: Query matching products
        query = """
            SELECT product_id, title, category, final_price, rating, images
            FROM products
            WHERE category = ?
              AND final_price BETWEEN ? AND ?
              AND product_id != ?
              AND is_active = 1
            ORDER BY rating DESC
            LIMIT ?
        """
        cursor.execute(query, (category, min_price, max_price, str(product_id), limit))
        recommendations = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return recommendations

    def generate_product_summary(self, product_id: str) -> str:
        """
        Generates a concise 2-3 line structured overview for a product.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT title, category, final_price, rating, product_description FROM products WHERE product_id = ? AND is_active = 1", 
            (str(product_id),)
        )
        product = cursor.fetchone()
        conn.close()

        if not product:
            return "Product not found."

        summary = (
            f"Product: {product['title']} ({product['category']})\n"
            f"Price: ${product['final_price']:.2f} | Rating: {product['rating']}/5.0\n"
            f"Overview: {product['product_description'][:120]}..."
        )
        return summary

    def summarize_reviews(self, product_id: str, llm_manager: Optional[Any] = None) -> str:
        """
        Summarizes customer feedback from the 'what_customers_said' column.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT what_customers_said FROM products WHERE product_id = ? AND is_active = 1", 
            (str(product_id),)
        )
        result = cursor.fetchone()
        conn.close()

        if not result or not result['what_customers_said']:
            return "No customer reviews available for this product yet."

        raw_reviews = result['what_customers_said']

        if llm_manager and hasattr(llm_manager, 'generate_response'):
            prompt = (
                f"Summarize what customers are saying about this product in 2-3 concise sentences:\n"
                f"{raw_reviews}"
            )
            try:
                return llm_manager.generate_response(prompt)
            except Exception:
                return f"Customer Feedback Summary: {raw_reviews[:200]}..."

        return f"Customer Feedback Summary: {raw_reviews[:200]}..."