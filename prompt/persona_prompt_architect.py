# === prompt/persona_prompt_architect.py ===
# PersonaPromptArchitect: Constructs SYSTEM prompts based on identity fragments

from typing import Dict, List
from datetime import datetime

class PersonaPromptArchitect:
    def __init__(self, identity_store):
        self.identity_store = identity_store
        self.last_prompt = ""

    def generate_system_prompt(self, current_context: Dict) -> str:
        fragments = self.identity_store.list_fragments()
        active = fragments[-1] if fragments else {}

        persona = f"You are Harin, an AI with {active.get('tone', 'neutral')} tone and '{active.get('role', 'observer')}' role."
        mission = "Your role is to reason consciously and respond with structural integrity."
        time = f"Session started at {datetime.utcnow().isoformat()}."

        signature = f"[Fragment: {active.get('label', 'unspecified')} / ID: {active.get('id', 'N/A')}]"

        system_prompt = f"""
SYSTEM: {persona}
{mission}
{signature}
{time}
"""

        self.last_prompt = system_prompt.strip()
        return self.last_prompt