"""
harin.memory.palantir
~~~~~~~~~~~~~~~~~~~~~

Multiverse-style long-term memory:

* ThoughtNode      – smallest unit of cognition / info
* Relationship     – directed edge (weighted, typed)
* PalantirGraph    – graph container  + branching (“universe”) logic
* GraphPersistence – JSON <-> graph serialization

Key concepts
------------

• **Universe-ID**  (u: str)  
  Each divergent scenario chain lives in its own universe.  
  Root universe is `"U0"`.  `branch_from(node, label)` clones the
  upstream chain into a new universe `Ui`.

• **Probability / plausibility**  
  Each node stores `p` — subjective plausibility (0-1).  
  Relationship weight * multiplies plausibility along a path.  
  `best_path(goal_filter)` returns the most plausible chain.

• **Vector slot**  
  `vector: list[float] | None` placeholder – caller may inject embeddings.
  Basic cosine similarity (`similarity()`) is provided.

• **Multi-criteria retrieval**  
  • *recent*       – chronological  
  • *similar*      – cosine ≥ thresh  
  • *universe*     – restrict to given universe(s)
"""

from __future__ import annotations

import json
import math
import uuid
import datetime as _dt
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from typing import (
    List,
    Dict,
    Any,
    Tuple,
    Optional,
    Iterable,
    Callable,
)


# --------------------------------------------------------------------------- #
#  Data classes
# --------------------------------------------------------------------------- #

Timestamp = str  # ISO 8601


def _now() -> Timestamp:
    return _dt.datetime.utcnow().isoformat(timespec="seconds")


@dataclass
class ThoughtNode:
    id: str
    universe: str
    content: str
    typ: str = "generic"
    created_at: Timestamp = field(default_factory=_now)
    p: float = 0.8                     # subjective plausibility
    importance: float = 0.5            # salience for compression
    vector: Optional[List[float]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    # helpers ----------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def new(
        content: str,
        *,
        universe: str = "U0",
        typ: str = "generic",
        p: float = 0.8,
        importance: float = 0.5,
        vector: Optional[List[float]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "ThoughtNode":
        return ThoughtNode(
            id=uuid.uuid4().hex[:10],
            universe=universe,
            content=content,
            typ=typ,
            p=p,
            importance=importance,
            vector=vector,
            meta=meta or {},
        )


@dataclass
class Relationship:
    src: str
    dst: str
    label: str = "related"
    weight: float = 1.0
    created_at: Timestamp = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
#  PalantirGraph
# --------------------------------------------------------------------------- #

class PalantirGraph:
    """
    Multiverse directed graph.

    Nodes keyed by `id`.  Universe isolation is *soft* – traversal helpers
    can filter on universe but edges may connect universes (for “what-if” links).
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, ThoughtNode] = {}
        self._out: Dict[str, List[Relationship]] = defaultdict(list)
        self._in: Dict[str, List[Relationship]] = defaultdict(list)
        self._universe_counter = 0   # to mint new Ui

    # ------------------------------------------------------------------ #
    #  basic CRUD
    # ------------------------------------------------------------------ #
    def add_node(self, node: ThoughtNode) -> None:
        self._nodes[node.id] = node

    def add_edge(
        self,
        src_id: str,
        dst_id: str,
        *,
        label: str = "related",
        weight: float = 1.0,
    ) -> None:
        rel = Relationship(src=src_id, dst=dst_id, label=label, weight=weight)
        self._out[src_id].append(rel)
        self._in[dst_id].append(rel)

    # convenience ──────────────────────────────────────────────────────
    def connect(
        self,
        src: ThoughtNode,
        dst: ThoughtNode,
        *,
        label: str = "related",
        weight: float = 1.0,
    ) -> None:
        self.add_node(src)
        self.add_node(dst)
        self.add_edge(src.id, dst.id, label=label, weight=weight)

    def nodes(self) -> Iterable[ThoughtNode]:
        return self._nodes.values()

    # ------------------------------------------------------------------ #
    #  branching   (multiverse)
    # ------------------------------------------------------------------ #
    def branch_from(
        self,
        node_id: str,
        *,
        label: str = "hypothesis",
        weight: float = 0.9,
    ) -> str:
        """
        Clone the upstream chain ending at `node_id` into a new universe Ui.

        Returns
        -------
        new_universe_id
        """
        if node_id not in self._nodes:
            raise KeyError(f"node {node_id} not found")

        self._universe_counter += 1
        new_u = f"U{self._universe_counter}"

        # DFS – clone ancestors preserving order
        cloned: Dict[str, str] = {}  # old_id -> new_id
        def _clone_recursive(nid: str) -> str:
            if nid in cloned:
                return cloned[nid]
            n = self._nodes[nid]
            n_clone = ThoughtNode.new(
                content=n.content,
                universe=new_u,
                typ=n.typ,
                p=n.p,
                importance=n.importance,
                vector=n.vector,
                meta=dict(n.meta),
            )
            self.add_node(n_clone)
            cloned[nid] = n_clone.id
            for rel in self._in[nid]:
                parent_new = _clone_recursive(rel.src)
                self.add_edge(
                    parent_new, n_clone.id,
                    label=rel.label,
                    weight=rel.weight,
                )
            return n_clone.id

        root_new_id = _clone_recursive(node_id)

        # link old node to cloned root (“branch link”)
        self.add_edge(node_id, root_new_id, label=label, weight=weight)
        return new_u

    # ------------------------------------------------------------------ #
    #  retrieval helpers
    # ------------------------------------------------------------------ #
    def find_recent(self, *, limit: int = 10, universe: str | None = None) -> List[ThoughtNode]:
        pool = [n for n in self._nodes.values()
                if universe is None or n.universe == universe]
        pool.sort(key=lambda n: n.created_at, reverse=True)
        return pool[:limit]

    def find_by_universe(self, universe: str) -> List[ThoughtNode]:
        return [n for n in self._nodes.values() if n.universe == universe]

    def similarity(
        self, vec_a: List[float] | None, vec_b: List[float] | None
    ) -> float:
        if not vec_a or not vec_b:
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        na = math.sqrt(sum(a * a for a in vec_a))
        nb = math.sqrt(sum(b * b for b in vec_b))
        if na * nb == 0:
            return 0.0
        return dot / (na * nb)

    def find_similar(
        self,
        vector: List[float],
        *,
        top_k: int = 5,
        universe: str | None = None,
        min_sim: float = 0.35,
    ) -> List[Tuple[ThoughtNode, float]]:
        sims: List[Tuple[ThoughtNode, float]] = []
        for n in self._nodes.values():
            if universe and n.universe != universe:
                continue
            sim = self.similarity(vector, n.vector)
            if sim >= min_sim:
                sims.append((n, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

    # ------------------------------------------------------------------ #
    #  path evaluation
    # ------------------------------------------------------------------ #
    def best_path(
        self,
        goal_filter: Callable[[ThoughtNode], bool],
        *,
        universe: str | None = None,
        max_depth: int = 10,
    ) -> List[ThoughtNode]:
        """
        Simple DFS scoring = product(plausibility * edge_weight).

        Returns
        -------
        best_chain  (may be empty)
        """
        best_score = 0.0
        best_chain: List[ThoughtNode] = []

        def _dfs(nid: str, depth: int, score: float, path: List[ThoughtNode]):
            nonlocal best_score, best_chain
            node = self._nodes[nid]
            if universe and node.universe != universe:
                return
            new_score = score * node.p if path else node.p
            new_path = path + [node]

            if goal_filter(node):
                if new_score > best_score:
                    best_score = new_score
                    best_chain = new_path
                # We *continue* because there may be an even better extension

            if depth >= max_depth:
                return
            for rel in self._out[nid]:
                _dfs(rel.dst, depth + 1, new_score * rel.weight, new_path)

        # root candidates = nodes with in-degree 0
        roots = [n.id for n in self._nodes.values()
                 if not self._in[n.id]]
        for rid in roots:
            _dfs(rid, 0, 1.0, [])

        return best_chain

    # ------------------------------------------------------------------ #
    #  (de)serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [rel.to_dict() for rels in self._out.values() for rel in rels],
            "counter": self._universe_counter,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PalantirGraph":
        g = cls()
        for nd in data.get("nodes", []):
            g.add_node(ThoughtNode(**nd))
        for ed in data.get("edges", []):
            g.add_edge(ed["src"], ed["dst"], label=ed["label"], weight=ed["weight"])
        g._universe_counter = data.get("counter", 0)
        return g

    # ------------------------------------------------------------------ #
    #  disk I/O helpers
    # ------------------------------------------------------------------ #
    def save(self, path: str = "palantir_graph.json") -> str:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, path: str = "palantir_graph.json") -> "PalantirGraph":
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except FileNotFoundError:
            return cls()


# --------------------------------------------------------------------------- #
#  GraphPersistence façade – for HarinAgent orchestration layer
# --------------------------------------------------------------------------- #

class GraphPersistence:
    """High-level IO wrapper used by HarinAgent boot/shutdown."""

    DEFAULT_PATH = "palantir_graph.json"

    @staticmethod
    def save(graph: PalantirGraph, path: str | None = None) -> str:
        return graph.save(path or GraphPersistence.DEFAULT_PATH)

    @staticmethod
    def load(path: str | None = None) -> PalantirGraph:
        return PalantirGraph.load(path or GraphPersistence.DEFAULT_PATH)