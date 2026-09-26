"""SQLite-backed cache of finished annotations and a daily spend ledger."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
import threading
import time


class Store:
    def __init__(self, path: str):
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, events TEXT, created REAL)")
            self._db.execute("CREATE TABLE IF NOT EXISTS spend (day TEXT PRIMARY KEY, usd REAL)")
            self._db.commit()

    @staticmethod
    def _today() -> str:
        return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")

    def get_cached(self, key: str):
        with self._lock:
            row = self._db.execute("SELECT events FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put_cached(self, key: str, events: list):
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?)", (key, json.dumps(events), time.time()))
            self._db.commit()

    def add_spend(self, usd: float):
        if usd <= 0:
            return
        day = self._today()
        with self._lock:
            self._db.execute("INSERT INTO spend VALUES (?, ?) ON CONFLICT(day) DO UPDATE SET usd = usd + ?",
                             (day, usd, usd))
            self._db.commit()

    def spent_today(self) -> float:
        with self._lock:
            row = self._db.execute("SELECT usd FROM spend WHERE day = ?", (self._today(),)).fetchone()
        return row[0] if row else 0.0
