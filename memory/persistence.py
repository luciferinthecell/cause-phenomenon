"""
harin.memory.persistence
~~~~~~~~~~~~~~~~~~~~~~~~

JSON 직렬화를 이용해 **PalantirGraph**(장기 기억)를 디스크에 영속화/복원한다.
추가로 전체 Harin 세션(그래프 + 부가 데이터)을 간단히 스냅샷/로드 하는
`MemoryPersistenceManager` 도 함께 제공한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional

from .palantir import PalantirGraph


# --------------------------------------------------------------------------- #
#  1) Low-level helper – GraphPersistence
# --------------------------------------------------------------------------- #


class GraphPersistence:
    """Save / load PalantirGraph → *.json on disk."""

    DEFAULT_PATH = Path("harin_memory.json")

    # ── public ────────────────────────────────────────────────────────────
    @staticmethod
    def save(graph: PalantirGraph, path: str | Path | None = None) -> Path:
        """Serialize *graph* to <path>.json (defaults to ./harin_memory.json)."""
        target = Path(path) if path else GraphPersistence.DEFAULT_PATH
        target.write_text(graph.to_json(), encoding="utf-8")
        return target

    @staticmethod
    def load(path: str | Path | None = None) -> PalantirGraph:
        """Read <path>.json and reconstruct a PalantirGraph; returns empty graph if file missing."""
        source = Path(path) if path else GraphPersistence.DEFAULT_PATH
        if not source.exists():
            return PalantirGraph()
        return PalantirGraph.from_json(source.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
#  2) High-level helper – MemoryPersistenceManager
# --------------------------------------------------------------------------- #


class MemoryPersistenceManager:
    """
    Convenience façade that bundles PalantirGraph + extra runtime artefacts
    (session trace, identity-shards …) into a single *save()* / *load()* pair.
    """

    def __init__(
        self,
        graph: PalantirGraph,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.graph = graph
        self.extra: Dict[str, Any] = extra or {}

    # ── save / load ───────────────────────────────────────────────────────
    def save(self, path: str | Path = "harin_snapshot.json") -> Path:
        """
        Dump graph **and** extra payload to one JSON file.

        File schema
        ```json
        {
          "graph": { ... },          # PalantirGraph JSON
          "extra": { ... arbitrary ... }
        }
        ```
        """
        bundle = {
            "graph": json.loads(self.graph.to_json()),
            "extra": self.extra,
        }
        p = Path(path)
        p.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    @staticmethod
    def load(path: str | Path = "harin_snapshot.json") -> "MemoryPersistenceManager":
        """Reverse of *save()* – returns a fully populated manager instance."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        graph = PalantirGraph.from_json(json.dumps(data["graph"], ensure_ascii=False))
        extra = data.get("extra", {})
        return MemoryPersistenceManager(graph, extra)

    # ── convenience helpers ───────────────────────────────────────────────
    def add_extra(self, key: str, value: Any) -> None:
        self.extra[key] = value

    def get_extra(self, key: str, default: Any = None) -> Any:
        return self.extra.get(key, default)

    # debug
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<MemoryPersistenceManager nodes={len(self.graph.nodes())} "
            f"extra_keys={list(self.extra)}>"
        )
