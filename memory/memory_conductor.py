# === memory/memory_conductor.py ===
# MemoryConductor: Recommends memories based on multi-factor ranking

from typing import List, Dict
from datetime import datetime
import math

class MemoryConductor:
    def __init__(self, memory_store):
        self.memory = memory_store

    def rank(self, query_vector: Dict, limit: int = 5) -> List[Dict]:
        candidates = self.memory.get_all_memory().get("M", [])
        scored = []

        for item in candidates:
            meta = item.get("meta", {})
            score = 0.0
            if "importance" in meta:
                score += 0.3 * meta.get("importance", 0.5)
            if "rhythm" in meta and meta["rhythm"] == query_vector.get("rhythm"):
                score += 0.2
            if "timestamp" in item:
                recency = self._time_decay(item["timestamp"])
                score += 0.2 * recency
            if meta.get("trust", 0.0) >= 0.7:
                score += 0.2
            if meta.get("failed", False):
                score -= 0.5  # penalize previously failed memories

            scored.append({
                "memory": item,
                "score": round(score, 3)
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return [s["memory"] for s in scored[:limit]]

    def _time_decay(self, timestamp: str) -> float:
        try:
            delta = (datetime.utcnow() - datetime.fromisoformat(timestamp)).total_seconds()
            return max(0.0, 1.0 - math.log1p(delta) / 100)
        except:
            return 0.5

    def recommend(self, context_vector: Dict, k: int = 5) -> List[Dict]:
        return self.rank(context_vector, limit=k)