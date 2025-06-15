"""
harin.core.goal_estimator
~~~~~~~~~~~~~~~~~~~~~~~~~

의미 임베딩·LLM 기반 루프 우선순위 스코어러.

철학
-----
* **키워드 매칭 없음** – 입력 문장을 embedding/LLM 분석 후 루프 목적 적합도를 0~1 실수로 반환.
* **백엔드 주입형** – 실제 계산(embedding or LLM)은 Strategy 객체를 외부에서 주입해 교체 가능.
* **루프 네임 확장 가능** – Registry 의 루프명을 그대로 받아 점수 딕셔너리 반환.

인터페이스
-----------
`GoalEstimator.score(user_input: str, loop_names: List[str]) -> Dict[str, float]`
"""

from __future__ import annotations

from typing import List, Dict, Protocol, runtime_checkable

# ──────────────────────────────────────────────────────────────────────────
#  Strategy protocol – external embedding/LLM scoring backend
# ──────────────────────────────────────────────────────────────────────────


@runtime_checkable
class SemanticScorer(Protocol):
    """Return similarity score [0,1] between *text* and *label*."""

    def similarity(self, text: str, label: str) -> float: ...


# ------------------------------------------------------------------------
#  Fallback mock scorer (length heuristic → no keyword trigger)
# ------------------------------------------------------------------------


class MockScorer:
    def similarity(self, text: str, label: str) -> float:
        # Simple length-based proxy (for demo only)
        import math

        return 1.0 / (1.0 + math.fabs(len(text) - len(label) * 10) / 100)


# ------------------------------------------------------------------------
#  GoalEstimator – converts semantic similarity into loop score mapping
# ------------------------------------------------------------------------


class GoalEstimator:
    def __init__(self, scorer: SemanticScorer | None = None) -> None:
        self.scorer: SemanticScorer = scorer or MockScorer()

    def score(self, user_input: str, loop_names: List[str]) -> Dict[str, float]:
        """Return dict(loop_name -> score [0,1]) based on semantic relevance."""
        scores: Dict[str, float] = {}
        for loop in loop_names:
            sim = self.scorer.similarity(user_input, loop)
            scores[loop] = max(0.0, min(sim, 1.0))
        # Normalize to sum=1 for probabilistic interpretation
        total = sum(scores.values()) or 1.0
        for k in scores:
            scores[k] /= total
        return scores
"""
