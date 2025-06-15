"""
harin.memory.schema
~~~~~~~~~~~~~~~~~~~

MemoryEngine 및 PalantirGraph에서 사용할 공통 태그 / 열거형 구조 정의.

• ThoughtType     : 사고 노드 유형 (질문, 주장, 사실 등)
• Stance          : 입장 / 태도 표현 (찬성, 반대, 중립 등)
• LinkType        : 노드 간 의미 연결 (근거, 반례, 조건 등)
• MemoryMetaKey   : 저장 시 사용할 표준 메타 키워드
"""

from enum import Enum
from typing import Literal


class ThoughtType(str, Enum):
    INPUT = "input"              # 사용자의 원 질문, 발언
    FACT = "fact"                # 확인된 사실
    CLAIM = "claim"              # 주관적 주장
    QUESTION = "question"        # 열린 질문
    STRATEGY = "strategy"        # 계획, 대응안
    REFLECTION = "reflection"    # 메타인지, 내적 응시
    EVALUATION = "evaluation"    # 평가 결과
    HYPOTHESIS = "hypothesis"    # 추정 / 가능성 제시


class Stance(str, Enum):
    AGREE = "agree"
    DISAGREE = "disagree"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class LinkType(str, Enum):
    SUPPORTS = "supports"        # → 근거 관계
    CONTRADICTS = "contradicts"  # → 반례, 논박
    ELABORATES = "elaborates"    # → 추가 설명
    REPHRASES = "rephrases"      # → 표현만 다름
    RESULTS_IN = "results_in"    # → 결과 관계
    IS_CAUSED_BY = "is_caused_by" # ← 원인 관계
    TAGGED_AS = "tagged_as"      # → 분류/속성 부여


# 사용자가 입력한 문장 또는 에이전트가 생성한 응답에 붙는 키워드
MemoryMetaKey = Literal[
    "user_input",
    "loop_id",
    "goal",
    "expert_selected",
    "plausibility",
    "timestamp",
    "tone",
    "emotion",
    "topic"
]
