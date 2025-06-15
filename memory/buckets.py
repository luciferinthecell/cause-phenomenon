"""
Light-weight, JSON-bucket memory layer.

*   Keeps fast append/read for recent vectors (T/C/I/E/M).
*   Persists as a JSON-Lines file (`harin_memory.jsonl`).
*   Intended for quick similarity lookup; heavyweight graph
    reasoning lives in `harin.memory.graph.PalantirGraph`.

Back-compat:
    `HarinMemoryBuckets` exports an alias
    `HarinMemoryPalantir` so legacy imports keep working.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

__all__ = ["MemoryRecord", "HarinMemoryBuckets", "HarinMemoryPalantir"]


DATA_FILE = Path(__file__).with_suffix(".jsonl")  # same folder → buckets.jsonl


@dataclass
class MemoryRecord:
    """
    Generic record used by the bucket store.
    Vectors follow the (T,C,I,E,M) convention but are fully optional;
    additional metadata can be attached in `meta`.
    """
    t: float  # timestamp (epoch seconds)
    content: str
    vectors: Dict[str, float]
    meta: Dict[str, Any]

    @classmethod
    def create(
        cls,
        content: str,
        vectors: Dict[str, float] | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> "MemoryRecord":
        return cls(
            t=time.time(),
            content=content,
            vectors=vectors or {},
            meta=meta or {},
        )


class HarinMemoryBuckets:
    """
    Append-only JSONL bucket memory.

    Basic API:
        • add(content, vectors, meta)=store & return MemoryRecord
        • recent(n, vector_key)=get last n contents for a bucket type
        • find(keyword, limit)=simple keyword scan (no embedding)
        • iterate()=yield MemoryRecord in order
    """

    def __init__(self, file_path: Path | str | None = None) -> None:
        self.file_path: Path = Path(file_path) if file_path else DATA_FILE
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────── Internal helpers ─────────────────────────────

    def _write_line(self, rec: MemoryRecord) -> None:
        with self.file_path.open("a", encoding="utf-8") as f:
            json.dump(asdict(rec), f, ensure_ascii=False)
            f.write("\n")

    def _read_lines(self) -> Sequence[MemoryRecord]:
        if not self.file_path.exists():
            return []
        with self.file_path.open(encoding="utf-8") as f:
            return [MemoryRecord(**json.loads(line)) for line in f]

    # ───────────────────────────── Public API ──────────────────────────────────

    def add(
        self,
        content: str,
        vectors: Dict[str, float] | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> MemoryRecord:
        rec = MemoryRecord.create(content, vectors, meta)
        self._write_line(rec)
        return rec

    def recent(
        self,
        n: int = 5,
        vector_key: str | None = None,
    ) -> List[str]:
        """
        Return latest `n` contents.
        If `vector_key` is given (e.g., "C"), filter to those records whose
        vectors include that key (>0).
        """
        items = reversed(self._read_lines())
        if vector_key:
            items = (r for r in items if r.vectors.get(vector_key, 0) > 0)
        return [r.content for r in items][:n]

    def find(self, keyword: str, limit: int = 10) -> List[MemoryRecord]:
        keyword_lower = keyword.lower()
        hits: List[MemoryRecord] = []
        for rec in reversed(self._read_lines()):
            if keyword_lower in rec.content.lower():
                hits.append(rec)
                if len(hits) >= limit:
                    break
        return hits

    # Simple iterator access for external modules
    def iterate(self) -> Sequence[MemoryRecord]:
        return self._read_lines()


# ───── Legacy alias ───────────────────────────────────────────────────────────
# To keep old imports functioning while the codebase migrates.
HarinMemoryPalantir = HarinMemoryBuckets
