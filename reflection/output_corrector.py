# === reflection/output_corrector.py ===
# OutputCorrector: Fixes or adjusts flawed responses based on SelfVerifier feedback

from typing import Dict

class OutputCorrector:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.last_revision = {}

    def correct(self, response_text: str, verifier_result: Dict, context_text: str = "") -> str:
        if verifier_result.get("score", 1.0) >= 0.8:
            return response_text  # No correction needed

        issues = verifier_result.get("issues", [])
        suggest = verifier_result.get("suggest_fix", "")

        correction_prompt = (
            "SYSTEM: You are a revision agent.
"
            "Revise the following response based on detected issues and suggestions.

"
            f"CONTEXT:
{context_text}

"
            f"ORIGINAL RESPONSE:
{response_text}

"
            f"ISSUES:
{issues}
"
            f"SUGGESTION:
{suggest}

"
            "Return the corrected version only."
        )

        try:
            revision = self.llm.complete(correction_prompt, temperature=0.3)
        except Exception:
            revision = "(correction failed) " + response_text

        self.last_revision = {
            "original": response_text,
            "revised": revision,
            "score_before": verifier_result.get("score"),
            "issues": issues
        }
        return revision