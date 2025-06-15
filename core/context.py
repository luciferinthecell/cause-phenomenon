"""
harin.core.context
~~~~~~~~~~~~~~~~~~

UserContext v2 – 감정 + 목적 + 인지 + 행동흐름 + 모드 기록

특징
─────
• 기존 mood, cognitive_level 유지
• last_mode = chat/research/control 기록
• active_plugins = 최근 호출 플러그인명 리스트
• context_trace = 질문 흐름 간단 로그
• last_action = 의미 기반 태그 (예: confirm, summarize)

메소드
────────
• update_from_input() : 임베딩/LLM 기반 상태 추론
• add_trace(label)    : 중복 방지 후 상황 레이블 추가
• as_meta()           : 메모리 기록용 메타 딕셔너리 반환
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class EmbeddingAnalyzer(Protocol):
    def classify_sentiment(self, text: str) -> str: ...
    def estimate_cognition(self, text: str) -> str: ...
    def extract_topics(self, text: str, *, top_k: int = 3) -> List[str]: ...


@runtime_checkable
class LLMReflector(Protocol):
    def reflect(self, text: str) -> Dict[str, Any]: ...


class MockAnalyzer:
    def classify_sentiment(self, text: str) -> str:
        return "neutral"

    def estimate_cognition(self, text: str) -> str:
        return "beginner" if len(text) < 60 else "expert"

    def extract_topics(self, text: str, *, top_k: int = 3) -> List[str]:
        return text.split()[:top_k]


class MockReflector:
    def reflect(self, text: str) -> Dict[str, Any]:
        return {}


@dataclass
class UserContext:
    mood: str = "unknown"
    cognitive_level: str = "unknown"
    goal_keywords: List[str] = field(default_factory=list)
    last_topic: str | None = None
    last_loop_id: str | None = None

    # v2 확장
    last_mode: str = "chat"
    active_plugins: List[str] = field(default_factory=list)
    context_trace: List[str] = field(default_factory=list)
    last_action: Optional[str] = None

    analyzer: EmbeddingAnalyzer = field(default_factory=MockAnalyzer)
    reflector: LLMReflector = field(default_factory=MockReflector)

    def update_from_input(self, user_input: str) -> None:
        self.mood = self.analyzer.classify_sentiment(user_input)
        self.cognitive_level = self.analyzer.estimate_cognition(user_input)
        topics = self.analyzer.extract_topics(user_input, top_k=5)
        if topics:
            self.last_topic = topics[0]
            self.goal_keywords = topics
        meta_extra = self.reflector.reflect(user_input)
        for k, v in meta_extra.items():
            setattr(self, k, v)

    def add_trace(self, label: str) -> None:
        if not self.context_trace or self.context_trace[-1] != label:
            self.context_trace.append(label)

    def as_meta(self) -> Dict[str, Any]:
        return {
            "mood": self.mood,
            "cognitive_level": self.cognitive_level,
            "goal_keywords": self.goal_keywords,
            "last_topic": self.last_topic,
            "last_loop_id": self.last_loop_id,
            "last_mode": self.last_mode,
            "context_trace": self.context_trace,
            "active_plugins": self.active_plugins,
            "last_action": self.last_action,
        }
"""
