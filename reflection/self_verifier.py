# === reflection/self_verifier.py ===
# SelfVerifier: Validates output before final delivery using LLM-based critique

from typing import Dict
import json

class SelfVerifier:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.last_verdict = {}

    def verify(self, response_text: str, context_text: str = "") -> Dict:
        prompt = (
            "SYSTEM: You are a strict output reviewer.
"
            "Check the following response for factual accuracy, coherence, and ethical consistency.

"
            f"CONTEXT:
{context_text}

"
            f"RESPONSE:
{response_text}

"
            "Return JSON with keys: score (0~1), issues (list of strings), and suggest_fix (optional string)."
        )
        try:
            raw = self.llm.complete(prompt, temperature=0)
            result = json.loads(raw)
        except Exception:
            result = {"score": 0.5, "issues": ["Verification failed"], "suggest_fix": ""}

        self.last_verdict = result
        return result