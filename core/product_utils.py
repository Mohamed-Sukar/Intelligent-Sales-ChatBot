import sqlite3
from typing import List, Dict, Any, Optional

class ProductUtils:
    """
    Utility class for fetching, formatting, and preparing product data 
    matching the ProductDatabase schema from data_pipeline.py.
    """
    def __init__(self, db_path: str = "database/chatbot.db"):
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        """Establishes a connection to the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetches full details of a single active product using product_id.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM products WHERE product_id = ? AND is_active = 1", 
            (str(product_id),)
        )
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def format_product_card(self, product: Dict[str, Any]) -> str:
        """
        Formats a raw product dictionary into Markdown for Gradio UI product cards.
        Matches schema: product_id, title, final_price, rating, category, product_description, images.
        """
        if not product:
            return "Product details unavailable."

        title = product.get("title", "Unknown Product")
        price = product.get("final_price", 0.0)
        initial_price = product.get("initial_price", price)
        discount = product.get("discount", 0.0)
        rating = product.get("rating", 0.0)
        category = product.get("category", "Uncategorized")
        description = product.get("product_description", "No description available.")
        images = product.get("images", "")

        card_md = f"###  {title}\n"
        
        # Extract first image if multiple images exist
        if images:
            first_img = images.split(',')[0].strip()
            card_md += f"![{title}]({first_img})\n\n"

        # Pricing and Discount line
        price_line = f"**Price:** ${price:.2f}"
        if discount > 0 and initial_price > price:
            price_line += f" ~(${initial_price:.2f})~  *{discount}% OFF*"

        card_md += (
            f"**Category:** {category}\n"
            f"{price_line}\n"
            f"**Rating:** {rating} / 5.0\n\n"
            f"**Description:**\n{description[:250]}...\n"
        )
        return card_md

    def format_product_list_for_llm(self, products: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved products into structured prompt context for LLM injection.
        """
        if not products:
            return "No matching products found."

        formatted_items = []
        for idx, item in enumerate(products, start=1):
            text_block = (
                f"{idx}. ID: {item.get('product_id')} | Name: {item.get('title')}\n"
                f"   Category: {item.get('category')} | Price: ${item.get('final_price', 0):.2f} | Rating: {item.get('rating')}\n"
                f"   Description: {str(item.get('product_description', ''))[:150]}...\n"
                f"   Reviews Summary: {str(item.get('what_customers_said', ''))[:100]}..."
            )
            formatted_items.append(text_block)

        return "\n\n".join(formatted_items)