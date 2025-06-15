# === interface/session_archiver.py ===
# SessionArchiver: Saves structured conversation and reasoning steps to session file

import json
from typing import List, Dict
from datetime import datetime
import os

class SessionArchiver:
    def __init__(self, path: str = "harin_session.jsonl"):
        self.path = path
        self.session_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.entries: List[Dict] = []

    def record(self, user_input: str, final_output: str, trace: List[Dict]):
        record = {
            "session": self.session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "input": user_input,
            "output": final_output,
            "trace": trace
        }
        self.entries.append(record)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "
")

    def load_all(self) -> List[Dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f.readlines()]

    def clear(self):
        if os.path.exists(self.path):
            os.remove(self.path)