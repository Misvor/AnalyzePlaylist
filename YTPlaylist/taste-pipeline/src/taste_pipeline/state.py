"""SQLite-backed pipeline state: the seen-history shared by all passes.

The store is a thin stdlib ``sqlite3`` wrapper over ``<data_dir>/state.db``.
The ``seen`` table records every video id the pipeline has listed, together
with the stage it reached (``listed`` -> ``metadata`` -> ``downloaded`` ...),
which makes re-runs idempotent: pass A drops ids already present here instead
of relying on yt-dlp's ``--download-archive`` (read-only during listing).
"""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

_DB_FILENAME: Final[str] = "state.db"
_CREATE_SEEN: Final[str] = """
CREATE TABLE IF NOT EXISTS seen (
    video_id TEXT PRIMARY KEY,
    first_seen TEXT NOT NULL,
    stage TEXT NOT NULL
)
"""
_INSERT_SEEN: Final[str] = "INSERT OR IGNORE INTO seen (video_id, first_seen, stage) VALUES (?, ?, ?)"
_SELECT_SEEN: Final[str] = "SELECT 1 FROM seen WHERE video_id = ?"


class StateStore:
    """Small wrapper over the pipeline's SQLite state database."""

    def __init__(self, data_dir: Path) -> None:
        """Open (creating if needed) ``state.db`` under ``data_dir`` and ensure the schema exists."""
        data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path: Final = data_dir / _DB_FILENAME
        self._conn: sqlite3.Connection = sqlite3.connect(self.db_path)
        self.create_tables()

    def create_tables(self) -> None:
        """Create the ``seen`` table when missing (safe to call repeatedly)."""
        with self._conn:
            _ = self._conn.execute(_CREATE_SEEN)

    def record_seen(self, video_id: str, stage: str) -> bool:
        """Insert ``video_id`` with ``stage`` and the current UTC time; existing rows are untouched.

        Returns:
            True when the id was newly inserted, False when it was already present.
        """
        first_seen = datetime.now(UTC).isoformat()
        with self._conn:
            cursor = self._conn.execute(_INSERT_SEEN, (video_id, first_seen, stage))
        return cursor.rowcount > 0

    def already_seen(self, video_id: str) -> bool:
        """Return True when ``video_id`` is present in the seen-history."""
        cursor = self._conn.execute(_SELECT_SEEN, (video_id,))
        return cursor.fetchone() is not None

    def close(self) -> None:
        """Close the underlying connection (Windows releases the file lock only on close)."""
        self._conn.close()
