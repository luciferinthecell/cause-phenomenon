"""
MemoryEngine 2.0  –  semantic, time-aware vector memory for Harin
══════════════════════════════════════════════════════════════════
Backend-agnostic: pass any embed-func + vector-db (Faiss, Qdrant…).
Adds:
  • time_decay(t)            – older memories fade unless high trust
  • trust_decay(score)       – low-scored items fade faster
  • recall()                 – weighted k-NN + diversity filter
"""

from __future__ import annotations
import math, time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

SECONDS_PER_DAY = 86_400


# ───────────────────────────────────────────────────────────────────
@dataclass
class MemoryItem:
    id: str
    text: str
    vector: List[float]
    meta: Dict
    ts: float = field(default_factory=time.time)   # unix
    trust: float = 0.7                             # 0‥1


# ───────────────────────────────────────────────────────────────────
class MemoryEngine:
    """
    embed_fn : Callable[[str], List[float]]
    vectordb : Any object with `.add(id, vector)` and `.search(vector, k)`
               (returns List[Tuple[id, score]])
    decay_cfg: dict – tweak γ (time) and β (trust)
    """

    def __init__(
        self,
        embed_fn: Callable[[str], List[float]],
        vectordb,
        *,
        decay_cfg: Dict = None,
    ) -> None:
        self.embed = embed_fn
        self.db = vectordb
        self.items: Dict[str, MemoryItem] = {}

        self.γ = (decay_cfg or {}).get("time_gamma", 0.03)   # per-day
        self.β = (decay_cfg or {}).get("trust_beta", 0.4)

    # ========================================================= #
    #  WRITE
    # ========================================================= #
    def add(self, text: str, meta: Dict | None = None, trust: float = 0.7) -> str:
        vec = self.embed(text)
        mid = f"M{len(self.items)+1:06d}"
        item = MemoryItem(id=mid, text=text, vector=vec, meta=meta or {}, trust=trust)
        self.items[mid] = item
        self.db.add(mid, vec)
        return mid

    # ========================================================= #
    #  READ
    # ========================================================= #
    def recall(self, query: str, *, k: int = 6, diversity: int = 4) -> List[MemoryItem]:
        qv = self.embed(query)
        hits = self.db.search(qv, k=16)  # (id, cosine)
        scored: List[Tuple[float, MemoryItem]] = []

        for mid, sim in hits:
            itm = self.items[mid]
            weight = sim * self._time_decay(itm.ts) * self._trust_decay(itm.trust)
            scored.append((weight, itm))

        # sort and diversity pick
        picked: List[MemoryItem] = []
        for _, itm in sorted(scored, key=lambda x: x[0], reverse=True):
            if not self._redundant(itm, picked):
                picked.append(itm)
            if len(picked) >= k or len(picked) >= diversity:
                break
        return picked

    # ========================================================= #
    #  DECAY
    # ========================================================= #
    def _time_decay(self, ts: float) -> float:
        age_days = (time.time() - ts) / SECONDS_PER_DAY
        return math.exp(-self.γ * age_days)

    def _trust_decay(self, trust: float) -> float:
        return math.exp(self.β * (trust - 1))  # high trust → ~1.0, low trust → down-weight

    # ========================================================= #
    #  UTIL
    # ========================================================= #
    def _redundant(self, itm: MemoryItem, pool: List[MemoryItem]) -> bool:
        dup_kw = set(itm.text.lower().split()[:4])
        for p in pool:
            if dup_kw & set(p.text.lower().split()[:4]):
                return True
        return False
