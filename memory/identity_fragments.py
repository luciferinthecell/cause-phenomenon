# === memory/identity_fragments.py ===
# IdentityFragments: Stores and rotates Harin's persona, tone, oath, and contradictions

from typing import Dict, List
from datetime import datetime
import uuid

class IdentityFragment:
    def __init__(self, label: str, role: str, tone: str, meta: Dict):
        self.id = str(uuid.uuid4())[:8]
        self.label = label
        self.role = role
        self.tone = tone
        self.meta = meta
        self.created_at = datetime.utcnow().isoformat()
        self.vector = self.compute_vector()

    def compute_vector(self):
        return {
            "empathy": 0.5 if self.tone == "neutral" else 0.8 if self.tone == "comforting" else 0.3,
            "assertiveness": 0.6 if self.role == "guardian" else 0.3,
            "self_reflection": 0.9 if "oath" in self.meta else 0.4
        }

class IdentityFragmentStore:
    def __init__(self):
        self.shards: Dict[str, IdentityFragment] = {}

    def add_fragment(self, label: str, role: str, tone: str, meta: Dict):
        frag = IdentityFragment(label, role, tone, meta)
        self.shards[frag.id] = frag
        return frag

    def list_fragments(self) -> List[Dict]:
        return [{
            "id": f.id,
            "label": f.label,
            "role": f.role,
            "tone": f.tone,
            "created_at": f.created_at,
            "vector": f.vector
        } for f in self.shards.values()]

    def get_fragment_by_role(self, role: str) -> List[IdentityFragment]:
        return [f for f in self.shards.values() if f.role == role]

    def export_json_ready(self) -> List[Dict]:
        return [{
            "id": f.id,
            "label": f.label,
            "created_at": f.created_at,
            "vector": f.vector,
            "meta": f.meta
        } for f in self.shards.values()]