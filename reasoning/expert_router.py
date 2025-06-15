"""
Expert Router & Expert Modules
──────────────────────────────
Selects the most suitable domain-expert for a user request,
using lightweight keyword heuristics + user profile hints.

Extendability:
    • Add new expert classes → register in `EXPERT_REGISTRY`.
    • Each expert implements:
        - .name           (str)
        - .handles(keywords, profile) → score (0-1)
        - .generate_plan(text, kw, profile) → dict
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


# ────────────────────────────────────────────────────────────────────────────
#  Base Expert
# ────────────────────────────────────────────────────────────────────────────
class BaseExpert(ABC):
    """Abstract contract every Expert must satisfy."""

    name: str = "base"

    # static convenience: pick‐weight pairs
    KEYWORDS: Tuple[str, ...] = ()

    @classmethod
    def handles(cls, keywords: List[str], profile: Dict[str, Any]) -> float:
        """
        Return suitability score (0-1). Default: keyword overlap proportion.
        Child classes can override with richer logic.
        """
        if not cls.KEYWORDS:
            return 0.1  # always minimally available
        hit = sum(1 for k in keywords if k in cls.KEYWORDS)
        return round(hit / len(cls.KEYWORDS), 2)

    # ------------------------------------------------------------------ #
    #  Mandatory: produce plan dict
    # ------------------------------------------------------------------ #
    @abstractmethod
    def generate_plan(
        self, user_text: str, keywords: List[str], user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        ...


# ────────────────────────────────────────────────────────────────────────────
#  Concrete Experts
# ────────────────────────────────────────────────────────────────────────────
class DebateExpert(BaseExpert):
    """Critical analysis & pro/contra builder."""

    name = "debate"
    KEYWORDS = ("이유", "논증", "찬성", "반대", "why", "pros", "cons")

    def generate_plan(self, user_text, keywords, profile):
        pos = [f"Pro ‣ {kw}" for kw in keywords[:3]]
        neg = [f"Con ‣ not {kw}" for kw in keywords[:3]]
        search_terms = [f"{kw} evidence" for kw in keywords[:2]]
        return {
            "task": "debate_analysis",
            "steps": ["extract_thesis", "enumerate_pros_cons", "weigh_evidence"],
            "arguments": {"pros": pos, "cons": neg},
            "search_queries": search_terms,
        }


class CreativeStrategistExpert(BaseExpert):
    """Idea generation, TRIZ‐style problem solving."""

    name = "creative_strategist"
    KEYWORDS = ("아이디어", "전략", "creative", "triz", "혁신", "해결책")

    def generate_plan(self, user_text, keywords, profile):
        ideas = [f"🎯 Idea-{i+1}: combine {kw} with XR" for i, kw in enumerate(keywords[:3])]
        return {
            "task": "creative_strategy",
            "framework": "TRIZ",
            "ideas": ideas,
            "search_queries": [f"{kw} disruptive tech" for kw in keywords[:1]],
        }


class GenericExpert(BaseExpert):
    """Fallback if no specialised expert fits."""

    name = "generic"
    KEYWORDS = ()

    def generate_plan(self, user_text, keywords, profile):
        return {
            "task": "generic_outline",
            "summary": f"Echo of input: {user_text[:60]} …",
            "search_queries": [],
        }


# ────────────────────────────────────────────────────────────────────────────
#  Registry & Router
# ────────────────────────────────────────────────────────────────────────────
EXPERT_REGISTRY: Tuple[BaseExpert.__class__, ...] = (
    DebateExpert,
    CreativeStrategistExpert,
    GenericExpert,
)


class ExpertRouter:
    """
    Choose expert with highest suitability score.
    Very light-weight; can be swapped for ML classifier later.
    """

    def __init__(self, experts: Tuple[BaseExpert.__class__, ...] = EXPERT_REGISTRY):
        self._experts = experts

    # ------------------------------------------------------------------ #
    #  Public
    # ------------------------------------------------------------------ #
    def select_expert(
        self, keywords: List[str], user_profile: Dict[str, Any]
    ) -> BaseExpert:
        scored = [
            (exp_cls, exp_cls.handles(keywords, user_profile)) for exp_cls in self._experts
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_score = scored[0][1]
        # tie-break: random among top candidates
        top_candidates = [c for c, s in scored if s == top_score]
        chosen_cls = random.choice(top_candidates)
        return chosen_cls()  # instantiate fresh each call

    # diagnostic helper
    def ranking(self, keywords: List[str], profile: Dict[str, Any]):
        return sorted(
            [(c.name, c.handles(keywords, profile)) for c in self._experts],
            key=lambda x: x[1],
            reverse=True,
        )
