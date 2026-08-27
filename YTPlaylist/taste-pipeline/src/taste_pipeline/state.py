"""SQLite-backed pipeline state: the seen-history shared by all passes.

The store is a thin stdlib ``sqlite3`` wrapper over ``<data_dir>/state.db``.
The ``seen`` table records every video id the pipeline has listed, together
with the stage it reached (``listed`` -> ``metadata`` -> ``downloaded`` ...),
which makes re-runs idempotent: pass A drops ids already present here instead
of relying on yt-dlp's ``--download-archive`` (read-only during listing).

The ``downloads`` table records pass-C results: for every downloaded video the
FLAC's path on disk and the stage it reached, so later passes (triage, web UI)
can locate the audio without re-walking the inbox.
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
_CREATE_DOWNLOADS: Final[str] = """
CREATE TABLE IF NOT EXISTS downloads (
    video_id TEXT PRIMARY KEY,
    stage TEXT NOT NULL,
    audio_path TEXT NOT NULL,
    downloaded_at TEXT NOT NULL
)
"""
_INSERT_SEEN: Final[str] = "INSERT OR IGNORE INTO seen (video_id, first_seen, stage) VALUES (?, ?, ?)"
_INSERT_DOWNLOAD: Final[str] = (
    "INSERT OR REPLACE INTO downloads (video_id, stage, audio_path, downloaded_at) VALUES (?, ?, ?, ?)"
)
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
        """Create the ``seen`` and ``downloads`` tables when missing (safe to call repeatedly)."""
        with self._conn:
            _ = self._conn.execute(_CREATE_SEEN)
            _ = self._conn.execute(_CREATE_DOWNLOADS)

    def record_seen(self, video_id: str, stage: str) -> bool:
        """Insert ``video_id`` with ``stage`` and the current UTC time; existing rows are untouched.

        Returns:
            True when the id was newly inserted, False when it was already present.
        """
        first_seen = datetime.now(UTC).isoformat()
        with self._conn:
            cursor = self._conn.execute(_INSERT_SEEN, (video_id, first_seen, stage))
        return cursor.rowcount > 0

    def record_download(self, video_id: str, audio_path: Path, stage: str = "downloaded") -> bool:
        """Record ``video_id`` as downloaded to ``audio_path`` with the current UTC time.

        Unlike :meth:`record_seen` this uses REPLACE semantics: a re-download of
        the same id updates the row, so it always points at the current file.

        Returns:
            True when the row was written.
        """
        downloaded_at = datetime.now(UTC).isoformat()
        with self._conn:
            cursor = self._conn.execute(_INSERT_DOWNLOAD, (video_id, stage, str(audio_path), downloaded_at))
        return cursor.rowcount > 0

    def already_seen(self, video_id: str) -> bool:
        """Return True when ``video_id`` is present in the seen-history."""
        cursor = self._conn.execute(_SELECT_SEEN, (video_id,))
        return cursor.fetchone() is not None

    def state_counts(self) -> dict[str, dict[str, int]]:
        """Return row counts per stage for both the seen and downloads tables.

        Only stages with at least one row appear in the inner dict; an empty
        table yields an empty inner dict (e.g. ``{"seen": {}, "downloads": {}}``
        for a freshly-opened store).
        """
        seen_cursor = self._conn.execute("SELECT stage, COUNT(*) FROM seen GROUP BY stage")
        downloads_cursor = self._conn.execute("SELECT stage, COUNT(*) FROM downloads GROUP BY stage")
        return {
            "seen": {str(stage): int(count) for stage, count in seen_cursor.fetchall()},
            "downloads": {str(stage): int(count) for stage, count in downloads_cursor.fetchall()},
        }

    def close(self) -> None:
        """Close the underlying connection (Windows releases the file lock only on close)."""
        self._conn.close()
