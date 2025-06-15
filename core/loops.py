"""
harin.core.loops
~~~~~~~~~~~~~~~~

동적 의미 기반 사고 루프 매퍼.

철학
-----
* **키워드 if/else 금지** – 루프 선택은 ‹GoalEstimator›가 산출한 임베딩/목적 스코어를 기반으로 한다.
* **플러그인 구조** – 각 루프는 `Loop` 프로토콜 구현체로 등록; 언제든 교체/추가 가능.
* **평가 피드백 루프** – 실행 후 `Judgment`를 반환하고, `JudgmentAggregator`에서 최종 선택.

주요 구성
---------
1. Loop (Protocol)       : `run(memory, context, user_input) -> Judgment`
2. LoopRegistry          : 루프명 → Loop 객체 매핑 (동적 추가/제거)
3. ThoughtProcessor      : GoalEstimator 점수를 기반으로 루프 호출 & 평가 반환

Note: GoalEstimator 는 외부 모듈에서 `score(loop_name) -> float` 형태로 주입.
"""

from __future__ import annotations

from typing import Protocol, Dict, Any, Callable, List, Tuple

from harin.memory.adapter import MemoryEngine
from .context import UserContext
from .judgment import Judgment, ScoreVector, JudgmentAggregator


# ──────────────────────────────────────────────────────────────────────────
#  Loop interface (no keyword triggers)
# ──────────────────────────────────────────────────────────────────────────


class Loop(Protocol):
    name: str
    def run(self, *, memory: MemoryEngine, context: UserContext, user_input: str) -> Judgment: ...


# ──────────────────────────────────────────────────────────────────────────
#  Example loop implementations (Simplified)
# ──────────────────────────────────────────────────────────────────────────


class RetrievalLoop:
    name = "retrieval"

    def run(self, *, memory: MemoryEngine, context: UserContext, user_input: str) -> Judgment:
        sims = memory.similarity(user_input, top_k=3)
        answer_parts = [n.text for n in sims]
        output = "\n".join(answer_parts) if answer_parts else "(정보 부족)"
        sv = ScoreVector(
            persuasiveness=0.6,
            consistency=0.8,
            credibility=0.7,
            affect_match=0.6,
        )
        return Judgment(loop_id=self.name, output_text=output, score=sv, rationale="retrieval based on similarity")


class CreativeLoop:
    name = "creative"

    def run(self, *, memory: MemoryEngine, context: UserContext, user_input: str) -> Judgment:
        output = f"{user_input}…에 대한 새로운 시나리오를 상상해볼게."  # placeholder
        sv = ScoreVector(0.7, 0.6, 0.4, 0.8)
        return Judgment(loop_id=self.name, output_text=output, score=sv, rationale="imaginative expansion")


class DebateLoop:
    name = "debate"

    def run(self, *, memory: MemoryEngine, context: UserContext, user_input: str) -> Judgment:
        output = f"{user_input}에 대한 찬반 논점을 구성해보자."  # placeholder
        sv = ScoreVector(0.65, 0.75, 0.6, 0.6)
        return Judgment(loop_id=self.name, output_text=output, score=sv, rationale="structured pro/con analysis")


# ──────────────────────────────────────────────────────────────────────────
#  Registry & Processor
# ──────────────────────────────────────────────────────────────────────────


class LoopRegistry:
    def __init__(self) -> None:
        self._loops: Dict[str, Loop] = {}

    def register(self, loop: Loop) -> None:
        self._loops[loop.name] = loop

    def get(self, name: str) -> Loop | None:
        return self._loops.get(name)

    def all(self) -> List[Loop]:
        return list(self._loops.values())


class ThoughtProcessor:
    """GoalEstimator 로부터 루프별 점수를 받아 실행·평가 후 최적 결과 반환."""

    def __init__(
        self,
        memory: MemoryEngine,
        context: UserContext,
        goal_estimator: Callable[[str], Dict[str, float]],  # loop_name → score (0~1)
    ) -> None:
        self.memory = memory
        self.context = context
        self.goals = goal_estimator
        self.registry = LoopRegistry()
        # 기본 루프 등록
        for loop_cls in (RetrievalLoop, CreativeLoop, DebateLoop):
            self.registry.register(loop_cls())

    # ---------------------------------------------------------------
    def process(self, user_input: str) -> Tuple[Judgment, List[Judgment]]:
        # 1) context 업데이트 (의미 기반 분석)
        self.context.update_from_input(user_input)

        # 2) 루프 점수 매핑
        scores = self.goals(user_input)  # 예: {"retrieval": 0.8, "creative": 0.2, …}

        # 3) 가중치 상위 N개 루프 실행 (여기선 3개 제한)
        ranked_loops = sorted(scores.items(), key=lambda t: t[1], reverse=True)[:3]

        agg = JudgmentAggregator()
        for name, _ in ranked_loops:
            loop = self.registry.get(name)
            if loop:
                j = loop.run(memory=self.memory, context=self.context, user_input=user_input)
                agg.add(j)

        best = agg.best()
        return best, agg.judgments
"""
