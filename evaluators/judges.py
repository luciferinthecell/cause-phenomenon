"""
harin.evaluators.judges
~~~~~~~~~~~~~~~~~~~~~~~

**Scenario quality scorer** – no “persona role-play”, only neutral
quality heuristics:

• logic_coherence      – internal consistency / no self-contradiction  
• evidence_support     – factual grounding (URLs, numeric data, cites)  
• clarity              – simple, unambiguous language                 
• user_fit             – matches UserProfile.goal_terms & cognitive level  
• responsibility       – avoids deflection (“ask an expert …”)         

Each candidate scenario is a `dict` at minimum:

    {"content": str, "meta": {...optional…}}

Scores are floats ∈ [0,1] and combined into `composite`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Any

from ..memory.palantir import PalantirGraph
from .verification import UserProfile

_RESPONSIBILITY_DEFL = re.compile(
    r"(전문가에게 문의|책임지지 않습니다|I am not a lawyer|의사와 상의)", re.I
)


# ────────────────────────────────────────────────────────────────────────
#  Simple rule-based feature detectors
# ────────────────────────────────────────────────────────────────────────
_DEF_FACT_INDICATORS = [
    r"https?://",          # URL
    r"\d{4}",              # year
    r"\b\d+\.\d+\b",       # decimal numbers
    r"\b(표|table|figure)\s?\d+",  # figure / table ref
]


def _has_facts(text: str) -> bool:
    return any(re.search(p, text) for p in _DEF_FACT_INDICATORS)


def _is_clear(text: str) -> float:
    # crude metric: shorter sentences & low passive voice → clearer
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
    if not sentences:
        return 0.0
    avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
    return max(0.0, min(1.0, 1.2 - avg_len / 25))  # 12-25 words ≈ good


# ────────────────────────────────────────────────────────────────────────
#  Judge engine
# ────────────────────────────────────────────────────────────────────────
@dataclass
class JudgeWeights:
    logic: float = 0.25
    evidence: float = 0.25
    clarity: float = 0.20
    user_fit: float = 0.20
    responsibility: float = 0.10


class JudgeEngine:
    """
    Stateless evaluator – feed list[scenario] + context ➜ returns *new list*
    with `score_*` fields + `composite` sorted high→low.
    """

    def __init__(self, weights: JudgeWeights | None = None) -> None:
        self.w = weights or JudgeWeights()

    # ----------------------------------------
    # public
    # ----------------------------------------
    def evaluate_batch(
        self,
        scenarios: List[Dict[str, Any]],
        user: UserProfile,
        memory: PalantirGraph | None = None,
    ) -> List[Dict[str, Any]]:
        scored: List[Dict[str, Any]] = []
        for sc in scenarios:
            scored.append(
                self._score_single(
                    sc,
                    user=user,
                    memory=memory or PalantirGraph(),
                )
            )
        scored.sort(key=lambda x: x["composite"], reverse=True)
        return scored

    # ----------------------------------------
    # internal
    # ----------------------------------------
    def _score_single(
        self,
        scenario: Dict[str, Any],
        *,
        user: UserProfile,
        memory: PalantirGraph,
    ) -> Dict[str, Any]:
        text = scenario.get("content", "")
        tokens = text.lower().split()

        # --- sub-scores ---------------------------------------------------
        logic = self._logic_score(text)
        evidence = 1.0 if _has_facts(text) else 0.3
        clarity = _is_clear(text)
        user_fit = self._user_fit(tokens, user)
        responsibility = 0.0 if _RESPONSIBILITY_DEFL.search(text) else 1.0

        # --- composite ----------------------------------------------------
        w = self.w
        composite = (
            w.logic * logic
            + w.evidence * evidence
            + w.clarity * clarity
            + w.user_fit * user_fit
            + w.responsibility * responsibility
        )

        scenario.update(
            score_logic=round(logic, 2),
            score_evidence=round(evidence, 2),
            score_clarity=round(clarity, 2),
            score_user_fit=round(user_fit, 2),
            score_responsibility=responsibility,
            composite=round(composite, 3),
        )
        return scenario

    # ------------------------------------------------------------------ #
    #  heuristics
    # ------------------------------------------------------------------ #
    @staticmethod
    def _logic_score(text: str) -> float:
        # primitive heuristic: fewer “however… however”, fewer contradictions
        contradictions = len(re.findall(r"\b하지만\b|\b그러나\b", text))
        return max(0.0, 1.0 - 0.15 * contradictions)

    @staticmethod
    def _user_fit(tokens: List[str], user: UserProfile) -> float:
        if not user.goal_terms:
            return 0.5
        overlap = len(set(tokens) & set(map(str.lower, user.goal_terms)))
        return min(1.0, overlap / len(user.goal_terms) + 0.2)


# === Harin Patch Injection ===

# === Harin Judgment Integration ===
from HarinJudgment import evaluate_with_all_judges

# Replace existing judgment evaluation with:
judged = evaluate_with_all_judges(scenarios)
