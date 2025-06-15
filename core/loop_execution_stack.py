# === harin/core/loop_execution_stack.py ===
# LoopExecutionStack (merged): Structured loop control + evaluation logic for HarinAgent

from typing import List, Dict, Optional
from dataclasses import dataclass, field
import uuid
import datetime as dt


@dataclass
class LoopTrace:
    loop_id: str
    status: str = "entered"  # "entered", "completed", "rerouted", "skipped"
    context: str = ""
    summary: str = ""
    score: float = -1.0
    timestamp: str = field(default_factory=lambda: dt.datetime.utcnow().isoformat())
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


class LoopExecutionStack:
    """
    Maintains the dynamic sequence of loops executed during HarinAgent reasoning.
    Supports:
    - Rerun due to confidence drop
    - Redirection to alternate loop
    - Branch-based parallel tracks (future extension)
    """

    def __init__(self):
        self.stack: List[LoopTrace] = []

    def push(self, loop_id: str, context: str = "") -> None:
        self.stack.append(LoopTrace(loop_id=loop_id, context=context, status="entered"))

    def mark_complete(self, loop_id: str) -> None:
        for lt in reversed(self.stack):
            if lt.loop_id == loop_id and lt.status == "entered":
                lt.status = "completed"
                return

    def reroute(self, from_loop: str, to_loop: str, reason: str = "") -> None:
        self.push(to_loop, context=f"rerouted from {from_loop}: {reason}")
        self._mark(from_loop, "rerouted")

    def skip(self, loop_id: str, reason: str = "") -> None:
        self.stack.append(LoopTrace(loop_id=loop_id, status="skipped", context=reason))

    def record_result(self, loop_id: str, score: float, summary: str = "") -> None:
        for lt in reversed(self.stack):
            if lt.loop_id == loop_id and lt.status == "entered":
                lt.status = "completed"
                lt.score = score
                lt.summary = summary
                return

    def evaluate_transition(self, score: float, threshold: float = 0.75) -> str:
        if score >= threshold:
            return "finalize"
        elif score >= 0.5:
            return "repeat"
        else:
            return "reroute"

    def get_history(self) -> List[Dict]:
        return [lt.__dict__ for lt in self.stack]

    def current_loop(self) -> Optional[str]:
        if self.stack:
            return self.stack[-1].loop_id
        return None