from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict

@dataclass
class Interaction:
    utterance: str
    timestamp: str
    meta: Dict

@dataclass
class UserModel:
    uid: str = "anonymous"
    knowledge_level: str = "unknown"
    preferred_depth: str = "balanced"
    emotional_tone: str = "neutral"
    purpose: str = "general"
    history: List[Interaction] = field(default_factory=list)

    def update_from_input(self, text: str) -> None:
        self.history.append(Interaction(text, datetime.utcnow().isoformat(), meta={}))
        if len(text.split()) > 50:
            self.preferred_depth = "deep"
        if any(w in text.lower() for w in ["prove", "theorem", "complexity"]):
            self.knowledge_level = "expert"
        elif any(w in text.lower() for w in ["easy", "beginner"]):
            self.knowledge_level = "novice"

    def snapshot(self) -> Dict:
        return {
            "knowledge_level": self.knowledge_level,
            "preferred_depth": self.preferred_depth,
            "emotional_tone": self.emotional_tone,
            "history_size": len(self.history)
        }


# === Harin Patch Injection ===

# === HarinMind Integration ===
from harinmind.intent_anchor import IntentAnchor
from harinmind.live_sync_monitor import LiveSyncMonitor

anchor = IntentAnchor()
monitor = LiveSyncMonitor()

# Example usage in user input processing:
user_intent = anchor.interpret_input(user_input)
monitor.register_input(user_input, user_intent)
