# === core/evolution_trigger.py ===
# EvolutionTrigger: Detects failure patterns and signals recursive self-modification

from typing import Dict, List
from datetime import datetime

class EvolutionTrigger:
    def __init__(self):
        self.history: List[Dict] = []

    def log_loop_outcome(self, loop_name: str, score: float, result: Dict):
        entry = {
            "loop": loop_name,
            "score": score,
            "result_summary": result.get("summary", ""),
            "timestamp": datetime.utcnow().isoformat()
        }
        self.history.append(entry)
        return entry

    def analyze_drift(self, window: int = 5) -> Dict:
        recent = self.history[-window:]
        avg_score = sum(e["score"] for e in recent) / max(1, len(recent))
        low_quality_count = sum(1 for e in recent if e["score"] < 0.5)
        drift_detected = low_quality_count >= 3

        return {
            "avg_score": round(avg_score, 3),
            "low_count": low_quality_count,
            "drift_trigger": drift_detected,
            "recommendation": "engage_correction_loop" if drift_detected else "continue"
        }

    def suggest_adjustment(self, context: Dict) -> str:
        mood = context.get("emotion", "neutral")
        if mood == "confused":
            return "invoke_clarify_loop"
        elif context.get("last_loop") == "retrieval_loop" and context.get("score", 0) < 0.4:
            return "reroute_to_triz_loop"
        return "proceed"