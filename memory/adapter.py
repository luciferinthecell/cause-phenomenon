"""
MemoryEngine  ════════════════════════════════════════════════════════════════
Bridges two complementary memory layers:

  1) 📒  BucketStore      – append-only JSON-Lines file (quick log & retrieval)
  2) 🔮  PalantirGraph    – structured long-term graph with ThoughtNode + edges

Public API (used by HarinAgent / ThoughtProcessor)
────────────────────────────────────────────────────
• store(text, *, node_type="misc", vectors=None, meta=None) → ThoughtNode
• similarity(query, *, top_k=3) → List[ThoughtNode]
• recent(n=5)                → List[ThoughtNode]
• save() / load()

Design goals
------------
• **Speed**: small in-memory indices for cosine similarity (cheap bag-of-words).
• **Transparency**: each store() returns the created node reference (id).
• **Safety**: atomic writes (tmp → rename) to avoid corruption.
"""

from __future__ import annotations

import json
import math
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Light import – PalantirGraph & ThoughtNode live next to this adapter
from harin.memory.palantir_graph import PalantirGraph, ThoughtNode  # type: ignore


# ╭─────────────────────────────────────────────────────────────────────────╮
# │ Utility – Bag-of-words cosine for quick similarity                     │
# ╰─────────────────────────────────────────────────────────────────────────╯
def _vectorise(text: str) -> Dict[str, float]:
    words = [w.lower() for w in text.split() if len(w) > 1]
    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    # l2-normalize
    norm = math.sqrt(sum(v * v for v in freq.values())) or 1.0
    return {k: v / norm for k, v in freq.items()}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    common = set(a).intersection(b)
    return sum(a[k] * b[k] for k in common)


# ╭─────────────────────────────────────────────────────────────────────────╮
# │ Flat-bucket layer                                                      │
# ╰─────────────────────────────────────────────────────────────────────────╯
@dataclass
class BucketEntry:
    id: str
    text: str
    node_type: str
    vectors: Dict[str, float]
    meta: Dict[str, any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class BucketStore:
    """Simple JSONL append log."""

    def __init__(self, path: Path):
        self.path = path
        self.entries: List[BucketEntry] = []
        self.load()

    # ------------------------------------------------------------------ #
    def append(self, entry: BucketEntry) -> None:
        self.entries.append(entry)
        self._atomic_write(entry.to_json())

    def last(self, n: int) -> Sequence[BucketEntry]:
        return self.entries[-n:]

    def load(self) -> None:
        if not self.path.exists():
            self.entries = []
            return
        with self.path.open(encoding="utf-8") as f:
            self.entries = [BucketEntry(**json.loads(line)) for line in f]

    # internal – safe write
    def _atomic_write(self, line: str) -> None:
        tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8")
        try:
            tmp.write(line + "\n")
            tmp.flush()
            tmp.close()
            # append to main file
            with self.path.open("a", encoding="utf-8") as dst:
                with open(tmp.name, encoding="utf-8") as src:
                    dst.write(src.read())
        finally:
            Path(tmp.name).unlink(missing_ok=True)


# ╭─────────────────────────────────────────────────────────────────────────╮
# │ MemoryEngine                                                           │
# ╰─────────────────────────────────────────────────────────────────────────╯
class MemoryEngine:
    """
    Quick bucket logging  +  semantic PalantirGraph storage
    """

    # rough default weights for T/C/I/E/M vectors – consumer can tweak
    _V_DEFAULT = {"T": 0.1, "C": 0.1, "I": 0.1, "E": 0.1, "M": 0.6}

    def __init__(
        self,
        *,
        bucket_path: Path,
        graph_path: Path,
    ) -> None:
        self.bucket = BucketStore(bucket_path)
        self.graph = PalantirGraph(path=graph_path)
        self._bow_index: Dict[str, Dict[str, float]] = {}  # node_id → bow vector

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def store(
        self,
        text: str,
        *,
        node_type: str = "misc",
        vectors: Optional[Dict[str, float]] = None,
        meta: Optional[Dict[str, any]] = None,
    ) -> ThoughtNode:
        vectors = vectors or self._V_DEFAULT.copy()
        meta = meta or {}

        node = ThoughtNode.new(text, node_type=node_type, vectors=vectors, meta=meta)
        self.graph.add_node(node)
        self._bow_index[node.id] = _vectorise(text)

        entry = BucketEntry(
            id=node.id,
            text=text,
            node_type=node_type,
            vectors=vectors,
            meta=meta,
        )
        self.bucket.append(entry)
        return node

    # ------------------------------------------------------------------ #
    def similarity(self, query: str, *, top_k: int = 3) -> List[ThoughtNode]:
        if not self._bow_index:
            self._rebuild_bow()
        q_vec = _vectorise(query)
        scored = [
            (nid, _cosine(q_vec, bow))
            for nid, bow in self._bow_index.items()
            if bow
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_ids = [nid for nid, score in scored[:top_k] if score > 0.2]
        return [self.graph[nid] for nid in top_ids if nid in self.graph]

    def recent(self, n: int = 5) -> List[ThoughtNode]:
        return [self.graph[e.id] for e in self.bucket.last(n)]

    # ------------------------------------------------------------------ #
    def save(self) -> None:
        self.graph.save()

    def load(self) -> None:
        self.graph.load()
        self._rebuild_bow()

    # internal
    def _rebuild_bow(self) -> None:
        self._bow_index = {nid: _vectorise(node.text) for nid, node in self.graph.nodes.items()}
