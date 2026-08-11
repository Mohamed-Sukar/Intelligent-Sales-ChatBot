"""
data_pipeline.py — P1: Data Engineer
=====================================
This module handles 3 Classes:
  1. DataCleaner      → Downloading & cleaning the raw dataset
  2. DatabaseBuilder  → Building SQLite database from clean data
  3. ProductDatabase  → Query & Filter API for the rest of the team

and 8 ready-to-use functions:
  1- get_product_by_id() → Fetch a single product by ID
  2- filter_by_category() → Get all products in a category (partial match)
  3- filter_by_price_range() → Get products within a price range
  4- filter_by_rating() → Get products with rating >= min_rating
  5- advanced_filter() → Advanced multi-filter query
  6- get_all_active_products() → Fetch ALL active products
  7- get_all_categories() → Get list of all categories with product counts
  8- get_stats() → Quick stats about the database
-->  setup_data_pipeline() → Run the full pipeline once to build the database
  - close() → Close the database connection
  
Usage:
    from core.data_pipeline import ProductDatabase

    db = ProductDatabase("database/chatbot.db")
    product = db.get_product_by_id("8376765")
    results = db.advanced_filter(category="shirts", max_price=50)
"""

import os
import re
import sqlite3
import pandas as pd


# ══════════════════════════════════════════════════════════════
# 1. DataCleaner — Download & Clean the Raw Dataset
# ══════════════════════════════════════════════════════════════

class DataCleaner:

    CURRENCY_RATES = {
        'USD': 1.0,
        '$': 1.0,
        'INR': 0.012,
        '₹': 0.012,
        'EUR': 1.08,
        '€': 1.08,
        'GBP': 1.27,
        '£': 1.27,
    }

    def clean_dataset(self, df):
        """Main cleaning pipeline — runs all steps in order."""

        print(f"🔄 Starting cleaning... ({len(df)} rows)")

        # Step 1: Remove duplicates
        df = self._remove_duplicates(df)

        # Step 2: Handle missing values
        df = self._handle_missing_values(df)

        # Step 3: Normalize prices to USD
        df = self._normalize_prices(df)

        # Step 4: Clean text columns
        df = self._clean_text_columns(df)

        # Step 5: Validate ratings
        df = self._clean_ratings(df)

        # Step 6: Add soft-delete flag
        df['is_active'] = 1

        print(f"✅ Cleaning done! ({len(df)} rows remaining)")
        return df

    def _remove_duplicates(self, df):
        before = len(df)
        df = df.drop_duplicates(subset=['title', 'final_price'], keep='first')
        removed = before - len(df)
        if removed > 0:
            print(f"  🗑️ Removed {removed} duplicate products")
        return df

    def _handle_missing_values(self, df):
        # Critical: drop rows without title or price
        df = df.dropna(subset=['title'])
        df = df.dropna(subset=['final_price'])

        # Fill descriptions with title as fallback
        df['product_description'] = df['product_description'].fillna(df['title'])

        # Fill numeric columns
        df['initial_price'] = df['initial_price'].fillna(df['final_price'])
        df['discount'] = df['discount'].fillna(0)
        df['rating'] = df['rating'].fillna(0)
        df['ratings_count'] = df['ratings_count'].fillna(0)

        # Fill text columns
        df['category'] = df['category'].fillna('Uncategorized')
        df['product_specifications'] = df['product_specifications'].fillna(
            'No specifications available'
        )
        df['what_customers_said'] = df['what_customers_said'].fillna(
            'No reviews available'
        )
        df['images'] = df['images'].fillna(
            'https://via.placeholder.com/300x300?text=No+Image'
        )
        df['seller_name'] = df['seller_name'].fillna('Unknown Seller')
        df['variations'] = df['variations'].fillna('Standard')

        print("  🔧 Missing values handled")
        return df

    def _normalize_prices(self, df):
        def convert(row, col):
            price = row[col]
            currency = str(row.get('currency', 'USD')).strip()
            if isinstance(price, str):
                price = re.sub(r'[^\d.]', '', price)
                price = float(price) if price else 0
            rate = self.CURRENCY_RATES.get(currency, 1.0)
            return round(float(price) * rate, 2)

        df['final_price'] = df.apply(lambda r: convert(r, 'final_price'), axis=1)
        df['initial_price'] = df.apply(lambda r: convert(r, 'initial_price'), axis=1)
        df['currency'] = 'USD'

        # Drop zero-price products
        df = df[df['final_price'] > 0]

        print("  💰 Prices normalized to USD")
        return df

    def _clean_text_columns(self, df):
        text_columns = [
            'title', 'product_description', 'product_specifications',
            'what_customers_said', 'category'
        ]
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
                df[col] = df[col].apply(lambda x: re.sub(r'<[^>]+>', '', x))
                df[col] = df[col].apply(lambda x: ' '.join(x.split()))
                df[col] = df[col].apply(
                    lambda x: re.sub(r'[^\w\s.,!?;:\-\'\"()/&%$#@+★☆●]', '', x)
                )
        print("  📝 Text columns cleaned")
        return df

    def _clean_ratings(self, df):
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
        df['rating'] = df['rating'].clip(0, 5)
        df['ratings_count'] = pd.to_numeric(
            df['ratings_count'], errors='coerce'
        ).fillna(0).astype(int)
        print("  ⭐ Ratings cleaned")
        return df


# ══════════════════════════════════════════════════════════════
# 2. DatabaseBuilder — Build SQLite from Clean DataFrame
# ══════════════════════════════════════════════════════════════

class DatabaseBuilder:

    def __init__(self, db_path="database/chatbot.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def build_database(self, df):
        """Build the full database from a clean DataFrame."""
        self._create_tables()
        self._insert_products(df)
        print(f"✅ Database built: {self.db_path}")
        print(f"   📦 Products: {len(df)}")

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS products (
                product_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                product_description TEXT,
                rating REAL DEFAULT 0,
                ratings_count INTEGER DEFAULT 0,
                initial_price REAL,
                discount REAL DEFAULT 0,
                final_price REAL NOT NULL,
                currency TEXT DEFAULT 'USD',
                category TEXT,
                breadcrumbs TEXT,
                product_specifications TEXT,
                what_customers_said TEXT,
                images TEXT,
                seller_name TEXT,
                variations TEXT,
                is_active INTEGER DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                intent TEXT,
                sentiment REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS viewed_products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                product_id TEXT NOT NULL,
                product_title TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS admin_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                product_id TEXT NOT NULL,
                details TEXT,
                admin_user TEXT DEFAULT 'admin',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Performance indexes
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_products_category ON products(category)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_products_price ON products(final_price)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_products_rating ON products(rating)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_products_active ON products(is_active)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id)"
        )

        self.conn.commit()
        print("  📋 Tables created with indexes")

    def _insert_products(self, df):
        for _, row in df.iterrows():
            self.conn.execute("""
                INSERT OR REPLACE INTO products
                (product_id, title, product_description, rating, ratings_count,
                 initial_price, discount, final_price, currency, category,
                 breadcrumbs, product_specifications, what_customers_said,
                 images, seller_name, variations, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(row.get('product_id', '')),
                row['title'],
                row['product_description'],
                row['rating'],
                row['ratings_count'],
                row['initial_price'],
                row['discount'],
                row['final_price'],
                row['currency'],
                row['category'],
                row.get('breadcrumbs', ''),
                row['product_specifications'],
                row['what_customers_said'],
                row.get('images', ''),
                row.get('seller_name', 'Unknown Seller'),
                row.get('variations', 'Standard'),
                row.get('is_active', 1)
            ))

        self.conn.commit()
        print(f"  📦 {len(df)} products inserted")

    def close(self):
        self.conn.close()


# ══════════════════════════════════════════════════════════════
# 3. ProductDatabase — Query & Filter API (for P2, P4, P5)
# ══════════════════════════════════════════════════════════════

class ProductDatabase:
    """
    The Query API that the rest of the team uses to access product data.

    Usage:
        db = ProductDatabase("database/chatbot.db")
        product = db.get_product_by_id("8376765")
        laptops = db.filter_by_category("laptops")
        cheap = db.filter_by_price_range(0, 50)
    """

    def __init__(self, db_path="database/chatbot.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    # ─── Get Single Product by ID ────────────────────────────
    def get_product_by_id(self, product_id):
        """Fetch a single product by its ID. Returns dict or None."""
        cursor = self.conn.execute(
            "SELECT * FROM products WHERE product_id = ? AND is_active = 1",
            (str(product_id),)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    # ─── Filter by Category ─────────────────────────────────
    def filter_by_category(self, category):
        """Get all products in a category (partial match). Sorted by rating."""
        cursor = self.conn.execute(
            """SELECT * FROM products
               WHERE category LIKE ? AND is_active = 1
               ORDER BY rating DESC""",
            (f"%{category}%",)
        )
        return [dict(row) for row in cursor]

    # ─── Filter by Price Range ───────────────────────────────
    def filter_by_price_range(self, min_price=0, max_price=float('inf')):
        """Get products within a price range. Sorted by price ascending."""
        cursor = self.conn.execute(
            """SELECT * FROM products
               WHERE final_price >= ? AND final_price <= ? AND is_active = 1
               ORDER BY final_price ASC""",
            (min_price, max_price)
        )
        return [dict(row) for row in cursor]

    # ─── Filter by Rating ────────────────────────────────────
    def filter_by_rating(self, min_rating=4.0):
        """Get products with rating >= min_rating. Sorted by rating descending."""
        cursor = self.conn.execute(
            """SELECT * FROM products
               WHERE rating >= ? AND is_active = 1
               ORDER BY rating DESC""",
            (min_rating,)
        )
        return [dict(row) for row in cursor]

    # ─── Advanced Filter (Category + Price + Rating + Sort) ──
    def advanced_filter(self, category=None, min_price=0,
                        max_price=float('inf'), min_rating=0,
                        sort_by='rating', limit=10):
        """
        Advanced multi-filter query.

        sort_by options: 'rating', 'price_low', 'price_high', 'reviews', 'discount'
        """
        query = "SELECT * FROM products WHERE is_active = 1"
        params = []

        if category:
            query += " AND category LIKE ?"
            params.append(f"%{category}%")

        query += " AND final_price >= ? AND final_price <= ?"
        params.extend([min_price, max_price])

        if min_rating > 0:
            query += " AND rating >= ?"
            params.append(min_rating)

        sort_options = {
            'rating': 'rating DESC',
            'price_low': 'final_price ASC',
            'price_high': 'final_price DESC',
            'reviews': 'ratings_count DESC',
            'discount': 'discount DESC'
        }
        query += f" ORDER BY {sort_options.get(sort_by, 'rating DESC')}"
        query += f" LIMIT {limit}"

        cursor = self.conn.execute(query, params)
        return [dict(row) for row in cursor]

    # ─── Get All Active Products (for P2: FAISS + BM25) ─────
    def get_all_active_products(self):
        """
        Fetch ALL active products.
        P2 uses this to build FAISS and BM25 search indexes.
        """
        cursor = self.conn.execute(
            "SELECT * FROM products WHERE is_active = 1"
        )
        return [dict(row) for row in cursor]

    # ─── Get All Categories ──────────────────────────────────
    def get_all_categories(self):
        """Get list of all categories with product counts. For UI buttons."""
        cursor = self.conn.execute(
            """SELECT DISTINCT category, COUNT(*) as count
               FROM products WHERE is_active = 1
               GROUP BY category ORDER BY count DESC"""
        )
        return [{"category": row[0], "count": row[1]} for row in cursor]

    # ─── Get Database Stats ──────────────────────────────────
    def get_stats(self):
        """Quick stats about the database. For verification."""
        total = self.conn.execute(
            "SELECT COUNT(*) FROM products WHERE is_active = 1"
        ).fetchone()[0]

        avg_price = self.conn.execute(
            "SELECT AVG(final_price) FROM products WHERE is_active = 1"
        ).fetchone()[0]

        avg_rating = self.conn.execute(
            "SELECT AVG(rating) FROM products WHERE is_active = 1 AND rating > 0"
        ).fetchone()[0]

        categories = self.conn.execute(
            "SELECT COUNT(DISTINCT category) FROM products WHERE is_active = 1"
        ).fetchone()[0]

        return {
            "total_products": total,
            "avg_price": round(avg_price, 2) if avg_price else 0,
            "avg_rating": round(avg_rating, 2) if avg_rating else 0,
            "total_categories": categories
        }

    def close(self):
        self.conn.close()


# ══════════════════════════════════════════════════════════════
# 4. Main Pipeline — Run Everything Once
# ══════════════════════════════════════════════════════════════

def setup_data_pipeline(csv_path="data/products_clean.csv",
                        db_path="database/chatbot.db"):
    """
    Main entry point — Run this once to build the database from clean CSV.

    Usage:
        python -m core.data_pipeline
    """
    print("=" * 50)
    print("🚀 Starting Data Pipeline...")
    print("=" * 50)

    # Step 1: Load clean data
    print("\n📥 Step 1: Loading clean dataset...")
    df = pd.read_csv(csv_path)
    print(f"   📊 Loaded {len(df)} products")

    # Step 2: Build database
    print("\n🗃️ Step 2: Building SQLite database...")
    builder = DatabaseBuilder(db_path)
    builder.build_database(df)
    builder.close()

    # Step 3: Verify
    print("\n✅ Step 3: Verification...")
    db = ProductDatabase(db_path)
    stats = db.get_stats()
    print(f"   📦 Total Products: {stats['total_products']}")
    print(f"   💰 Avg Price: ${stats['avg_price']}")
    print(f"   ⭐ Avg Rating: {stats['avg_rating']}")
    print(f"   📂 Categories: {stats['total_categories']}")
    db.close()

    print("\n" + "=" * 50)
    print("✅ Data Pipeline Complete!")
    print("=" * 50)


if __name__ == "__main__":
    setup_data_pipeline()