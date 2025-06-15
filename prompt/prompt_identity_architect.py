# === harin/prompt/prompt_identity_architect.py ===
# PromptArchitect: Persona + Rhythm + Identity aware prompt generator

from typing import Dict, List, Optional
from harin.core.state import HarinState
from harin.memory.identity_fragments import IdentityFragmentStore

class PromptArchitect:
    """
    Builds final SYSTEM prompt based on:
    - HarinState (emotion, rhythm, last input/output)
    - IdentityFragments (persona metadata)
    - Memory insights (optional)
    """

    def __init__(self, state: HarinState, identity: IdentityFragmentStore):
        self.state = state
        self.identity = identity

    def build_system_prompt(self, topic: Optional[str] = None) -> str:
        fragments = self.identity.list_fragments()
        persona_lines = []
        for f in fragments:
            persona_lines.append(
                f"[{f['role'].upper()}] {f['label']} (tone: {f['tone']})"
            )

        rhythm = self.state.rhythm_register
        rhythm_summary = (
            f"• Truth={rhythm['truth']:.2f}, "
            f"Responsibility={rhythm['responsibility']:.2f}, "
            f"Resonance={rhythm['resonance']:.2f}"
        )

        system = f"""SYSTEM:
You are HARIN, a reasoning agent with deep memory and shifting persona.

Your current rhythm alignment is:
{rhythm_summary}

Persona fragments:
{chr(10).join(persona_lines)}

Task topic: {topic or "unspecified"}
"""

        return system