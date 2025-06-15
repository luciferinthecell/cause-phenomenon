# === memory/trace_register.py ===
# TraceRegister: Logs reasoning loop events, intent shifts, and re-evaluations

from typing import List, Dict
from datetime import datetime
import uuid

class TraceRegister:
    def __init__(self):
        self.events: List[Dict] = []

    def log_event(self, loop_id: str, phase: str, status: str, intent: str = "", note: str = ""):
        event = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.utcnow().isoformat(),
            "loop": loop_id,
            "phase": phase,
            "status": status,
            "intent": intent,
            "note": note
        }
        self.events.append(event)
        return event

    def get_recent(self, n=5) -> List[Dict]:
        return self.events[-n:]

    def export(self) -> List[Dict]:
        return self.events