"""
Metacognition
─────────────
Self-assessment + reflection on the reasoning artefact (plan).

Returned structure
------------------
{
    "trust_score" : float (0-1),
    "signals"     : { "complexity":…, "search_support":…, "argument_depth":… },
    "comments"    : [ str, … ]
}
"""

from __future__ import annotations

import math
from typing import Any, Dict, List


# ────────────────────────────────────────────────────────────────────────────
#  Heuristic scorers
# ────────────────────────────────────────────────────────────────────────────
def _complexity(plan: Dict[str, Any]) -> float:
    """Longer + multi-step plans ≈ richer reasoning."""
    steps = plan.get("steps", [])
    depth = len(steps)
    return min(1.0, depth / 6.0)  # 6-step ⇒ 1.0


def _argument_depth(plan: Dict[str, Any]) -> float:
    """Debate style arguments?"""
    args = plan.get("arguments", {})
    pros = len(args.get("pros", []))
    cons = len(args.get("cons", []))
    return min(1.0, (pros + cons) / 8.0)  # 8 arguments ⇒ 1.0


def _search_support(plan: Dict[str, Any]) -> float:
    """Plans with external evidence are slightly more reliable."""
    qnum = len(plan.get("search_queries", []))
    return 0.6 if qnum else 0.3


# ────────────────────────────────────────────────────────────────────────────
#  Metacognition core
# ────────────────────────────────────────────────────────────────────────────
class Metacognition:
    """
    Very light-weight – pure functions now, can be swapped with an
    LLM-powered evaluator later.
    """

    # weight for each signal dimension
    _W = {"complexity": 0.35, "argument_depth": 0.35, "search_support": 0.30}

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def evaluate_trust_score(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        sig = {
            "complexity": _complexity(plan),
            "argument_depth": _argument_depth(plan),
            "search_support": _search_support(plan),
        }
        score = sum(self._W[k] * sig[k] for k in sig)
        score = round(score, 3)

        comments: List[str] = []
        if score < 0.4:
            comments.append("Plan too shallow → needs expansion.")
        if sig["search_support"] < 0.4:
            comments.append("No external evidence detected.")
        if sig["argument_depth"] < 0.3:
            comments.append("Weak pro/con structure.")

        return {"trust_score": score, "signals": sig, "comments": comments}

    # ------------------------------------------------------------------ #
    def reflect(self, meta: Dict[str, Any]) -> str:
        """Generate one-line reflection string."""
        score = meta["trust_score"]
        if score >= 0.75:
            outlook = "Confidence is high."
        elif score >= 0.55:
            outlook = "Confidence is moderate."
        else:
            outlook = "⚠ Confidence is low – refinement suggested."
        detail = "; ".join(meta["comments"]) if meta["comments"] else "No major issues."
        return f"{outlook} ({score})  |  {detail}"
