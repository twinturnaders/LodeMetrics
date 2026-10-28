import sqlite3
import numpy as np
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LodeDB:
    def __init__(self, db_path: str = "lodemetrics.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite schema. Uses WAL mode for better concurrent read/write performance."""
        with self.get_conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")

            conn.executescript("""
                               CREATE TABLE IF NOT EXISTS listings
                               (
                                   id
                                   INTEGER
                                   PRIMARY
                                   KEY
                                   AUTOINCREMENT,
                                   platform
                                   TEXT
                                   NOT
                                   NULL,
                                   platform_id
                                   TEXT
                                   NOT
                                   NULL,
                                   title
                                   TEXT,
                                   description
                                   TEXT,
                                   category
                                   TEXT,
                                   brand
                                   TEXT,
                                   size
                                   TEXT,
                                   condition
                                   TEXT,
                                   original_price
                                   REAL,
                                   sold_price
                                   REAL,
                                   currency
                                   TEXT
                                   DEFAULT
                                   'USD',
                                   listed_at
                                   TEXT,
                                   sold_at
                                   TEXT,
                                   dom
                                   INTEGER, -- Days on Market
                                   url
                                   TEXT,
                                   scraped_at
                                   TEXT
                                   NOT
                                   NULL,
                                   UNIQUE
                               (
                                   platform,
                                   platform_id
                               )
                                   );

                               CREATE TABLE IF NOT EXISTS photos
                               (
                                   id
                                   INTEGER
                                   PRIMARY
                                   KEY
                                   AUTOINCREMENT,
                                   listing_id
                                   INTEGER
                                   NOT
                                   NULL,
                                   url
                                   TEXT,
                                   local_path
                                   TEXT,
                                   embedding
                                   BLOB, -- Stores numpy array as bytes
                                   scraped_at
                                   TEXT
                                   NOT
                                   NULL,
                                   FOREIGN
                                   KEY
                               (
                                   listing_id
                               ) REFERENCES listings
                               (
                                   id
                               ) ON DELETE CASCADE
                                   );

                               CREATE INDEX IF NOT EXISTS idx_listings_platform ON listings(platform);
                               CREATE INDEX IF NOT EXISTS idx_listings_sold_price ON listings(sold_price);
                               CREATE INDEX IF NOT EXISTS idx_photos_listing_id ON photos(listing_id);
                               """)
        logger.info(f"Database initialized at {self.db_path}")

    @contextmanager
    def get_conn(self):
        """Context manager for DB connections to ensure clean commits/rollbacks."""
        conn = sqlite3.connect(self.db_path, timeout=20.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"DB Error: {e}")
            raise
        finally:
            conn.close()

    def insert_listing(self, data: dict) -> int:
        """Inserts a listing. Returns the row ID. Ignores if platform_id already exists."""
        with self.get_conn() as conn:
            cursor = conn.execute("""
                                  INSERT
                                  OR IGNORE INTO listings 
                (platform, platform_id, title, description, category, brand, size, condition, 
                 original_price, sold_price, listed_at, sold_at, dom, url, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                  """, (
                                      data['platform'], data['platform_id'], data.get('title'), data.get('description'),
                                      data.get('category'), data.get('brand'), data.get('size'), data.get('condition'),
                                      data.get('original_price'), data.get('sold_price'), data.get('listed_at'),
                                      data.get('sold_at'), data.get('dom'), data.get('url'),
                                      datetime.utcnow().isoformat()
                                  ))
            if cursor.rowcount == 0:
                # Fetch existing ID if ignored
                row = conn.execute("SELECT id FROM listings WHERE platform=? AND platform_id=?",
                                   (data['platform'], data['platform_id'])).fetchone()
                return row['id']
            return cursor.lastrowid

    def insert_photo(self, listing_id: int, url: str, local_path: str = None, embedding: np.ndarray = None):
        """Inserts a photo record. Converts numpy embedding to BLOB."""
        emb_blob = embedding.tobytes() if embedding is not None else None
        with self.get_conn() as conn:
            conn.execute("""
                         INSERT INTO photos (listing_id, url, local_path, embedding, scraped_at)
                         VALUES (?, ?, ?, ?, ?)
                         """, (listing_id, url, local_path, emb_blob, datetime.utcnow().isoformat()))

    def get_embedding(self, photo_id: int, dim: int = 512) -> np.ndarray:
        """Retrieves and reconstructs a numpy embedding from a BLOB."""
        with self.get_conn() as conn:
            row = conn.execute("SELECT embedding FROM photos WHERE id=?", (photo_id,)).fetchone()
            if row and row['embedding']:
                return np.frombuffer(row['embedding'], dtype=np.float32).reshape(dim)
        return None