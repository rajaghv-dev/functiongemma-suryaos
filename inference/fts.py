"""
FTS5 retrieval over the tool catalog.

Why FTS5?
    SQLite FTS5 is always present (zero extra deps), handles exact and
    prefix matches, and is fast enough for catalogs up to ~10,000 tools.
    It's the always-on fallback when vector retrieval is unavailable.

Data model:
    A virtual FTS5 table in the same SQLite DB used by the audit logger:

        CREATE VIRTUAL TABLE tool_fts USING fts5(
            tool_name UNINDEXED,    -- exact lookup key
            searchable              -- all text we want to match
        );

    The `searchable` column is a single concatenated blob:
        "<name>  <domain>  <description>  <example1>  <example2>  <alias1>"

    FTS5 tokenises and indexes this. At query time we use:
        SELECT tool_name, bm25(tool_fts) as score
        FROM tool_fts
        WHERE tool_fts MATCH ?
        ORDER BY score
        LIMIT ?

    bm25() returns a negative value (higher = more negative = less relevant);
    we negate it for a positive relevance score.

Usage:
    fts = FTSIndex(db_path)
    fts.build(registry)                     # (re)build from tool catalog
    results = fts.search("volume up", k=5)  # list[FTSResult]
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from surya_agent.tool_registry import ToolRegistry


FTS_TABLE = "tool_fts"


@dataclass
class FTSResult:
    tool_name: str
    score: float            # higher = more relevant


class FTSIndex:
    """SQLite FTS5 index over the tool catalog."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build(self, registry: "ToolRegistry") -> int:
        """(Re)build the FTS index from the current tool registry.

        Drops and recreates the FTS table so removed tools are cleaned up.
        Returns the number of tools indexed.
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        conn.execute(f"DROP TABLE IF EXISTS {FTS_TABLE}")
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE {FTS_TABLE} USING fts5(
                tool_name UNINDEXED,
                searchable
            )
            """
        )
        rows = []
        for tool in registry.tools.values():
            parts = [
                tool.name,
                tool.domain,
                tool.description,
                " ".join(tool.examples),
                " ".join(tool.aliases),
                " ".join(tool.tags),
            ]
            searchable = "  ".join(p for p in parts if p)
            rows.append((tool.name, searchable))
        conn.executemany(f"INSERT INTO {FTS_TABLE}(tool_name, searchable) VALUES (?, ?)", rows)
        conn.commit()
        conn.close()
        return len(rows)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 12) -> list[FTSResult]:
        """Return up to k tools ranked by FTS5 BM25 relevance.

        Returns an empty list (not an exception) when the index doesn't
        exist yet — callers should check for that and fall back.
        """
        if not self._table_exists():
            return []

        # FTS5 MATCH syntax rejects punctuation (?, !, -, etc.).
        # Strip everything except alphanumerics and spaces before querying.
        import re
        clean = re.sub(r"[^\w\s]", " ", query).strip()
        if not clean:
            return []

        # FTS5 bm25() returns negative numbers; negate for ascending relevance.
        conn = self._connect()
        try:
            cur = conn.execute(
                f"""
                SELECT tool_name, -bm25({FTS_TABLE}) AS score
                FROM {FTS_TABLE}
                WHERE {FTS_TABLE} MATCH ?
                ORDER BY score DESC
                LIMIT ?
                """,
                (clean, k),
            )
            results = [FTSResult(tool_name=row[0], score=float(row[1])) for row in cur]
        except Exception:
            results = []
        conn.close()
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _table_exists(self) -> bool:
        if not self.db_path.exists():
            return False
        conn = self._connect()
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (FTS_TABLE,),
        )
        exists = cur.fetchone() is not None
        conn.close()
        return exists
