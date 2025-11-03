import sqlite3
import threading
from typing import Any, Optional
import json
import time

lock = threading.Lock()
DB_FILE = "shared_state.db"

# Initialize the database
with lock:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS state (
            key TEXT PRIMARY KEY,
            value TEXT,
            expire_at REAL
        )
    """)
    conn.commit()
    conn.close()

def update_state(key: str, value: Any, ttl: int = 300):
    """Set a key-value pair with TTL (seconds)."""
    expire_at = time.time() + ttl
    value_json = json.dumps(value)
    with lock:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("""
            INSERT INTO state (key, value, expire_at) 
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, expire_at=excluded.expire_at
        """, (key, value_json, expire_at))
        conn.commit()
        conn.close()

def get_state(key: str) -> Optional[Any]:
    """Get a value from the shared state, respecting TTL."""
    with lock:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT value, expire_at FROM state WHERE key=?", (key,))
        row = c.fetchone()
        conn.close()
        if row:
            value_json, expire_at = row
            if time.time() > expire_at:
                return None
            return json.loads(value_json)
        return None