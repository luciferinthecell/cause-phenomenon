"""
harin.core.judgment
~~~~~~~~~~~~~~~~~~~

문맥·의미 기반 사고 평가 객체.

• Judgment          : 단일 루프 결과 및 자체 평가 메타 보존
• JudgmentAggregator: 다중 루프 결과를 합산·비교

특징
-----
* 점수는 단순 가중치가 아닌, ▲설득력 ▲일관성 ▲신뢰도 ▲정서 적합성 네 벡터로 표현.
* 평가 로직은 키워드가 아닌 LLM self-critique 프롬프트 또는 모델 기반 벡터 유사도 사용.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any

# ──────────────────────────────────────────────────────────────────────────
#  Judgment Vector
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class ScoreVector:
    persuasiveness: float  # 0~1
    consistency: float     # 논리·근거 충돌 여부
    credibility: float     # 외부 검증·출처 적합성
    affect_match: float    # 사용자 감정 톤과의 정합성

    def overall(self, w: Dict[str, float] | None = None) -> float:
        w = w or {
            "persuasiveness": 0.35,
            "consistency": 0.25,
            "credibility": 0.25,
            "affect_match": 0.15,
        }
        return (
            self.persuasiveness * w["persuasiveness"]
            + self.consistency * w["consistency"]
            + self.credibility * w["credibility"]
            + self.affect_match * w["affect_match"]
        )


# ──────────────────────────────────────────────────────────────────────────
#  Judgment Data
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class Judgment:
    loop_id: str
    output_text: str
    score: ScoreVector
    rationale: str = ""  # LLM self‑critique 서술 or external explanation
    meta: Dict[str, Any] = field(default_factory=dict)

    # 편의 함수
    def as_dict(self) -> Dict[str, Any]:
        d = {
            "loop_id": self.loop_id,
            "output": self.output_text,
            "rationale": self.rationale,
            "meta": self.meta,
            "scores": {
                "persuasiveness": self.score.persuasiveness,
                "consistency": self.score.consistency,
                "credibility": self.score.credibility,
                "affect_match": self.score.affect_match,
                "overall": self.score.overall(),
            },
        }
        return d


# ──────────────────────────────────────────────────────────────────────────
#  Aggregator
# ──────────────────────────────────────────────────────────────────────────


class JudgmentAggregator:
    """여러 루프의 Judgment 객체를 받아 최상위 결과 선택."""

    def __init__(self) -> None:
        self.judgments: List[Judgment] = []

    def add(self, j: Judgment) -> None:
        self.judgments.append(j)

    def best(self) -> Judgment | None:
        if not self.judgments:
            return None
        return max(self.judgments, key=lambda j: j.score.overall())

    def summary(self) -> List[Dict[str, Any]]:
        return [j.as_dict() for j in sorted(self.judgments, key=lambda j: j.score.overall(), reverse=True)]
"""
