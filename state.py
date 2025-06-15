"""
harin.core.state
~~~~~~~~~~~~~~~~

• 런타임 컨텍스트(감정, 루프 ID, 최근 I/O 등)를 보관하는 경량 데이터 클래스
• 모든 하위 모듈은 *동일한* HarinState 인스턴스를 주입받아 상태를 읽고/갱신함
"""

from __future__ import annotations

import uuid
import datetime as _dt
from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class HarinState:
    """에이전트의 현재 *정신 상태*를 캡슐화한다."""

    # ---- 핵심 식별자 / 루프 ----
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    loop_id: str = "000"                      # 현재 선택된 사고 루프

    # ---- 감정 & 페르소나 ----
    current_emotion: str = "neutral"          # e.g. neutral / sad / curious / hopeful
    preset_profile: Dict[str, Any] = field(default_factory=dict)

    # ---- 최근 입출력 ----
    last_input: str = ""
    last_output: str = ""

    # ---- 리듬(진실성·책임·공명) ----
    rhythm_register: Dict[str, float] = field(
        default_factory=lambda: {"truth": 0.5, "responsibility": 0.5, "resonance": 0.5}
    )

    # ---- 추적용 메모리(세션 한정) ----
    memory_trace: List[Dict[str, Any]] = field(default_factory=list)
    annotations: List[str] = field(default_factory=list)     # 메타 루프가 남기는 각종 주석

    # ──────────────────────────────────────────────────────────
    # •••  public helper methods
    # ──────────────────────────────────────────────────────────
    def update_emotion(self, new_emotion: str) -> None:
        """감정값을 갱신하고 트레이스를 남긴다."""
        self.current_emotion = new_emotion
        self._trace("emotion", new_emotion)

    def update_last_io(self, user_input: str, model_output: str) -> None:
        """최근 I/O 레코드 업데이트 + 트레이스."""
        self.last_input = user_input
        self.last_output = model_output
        self._trace(
            "io",
            {
                "input": user_input,
                "output": model_output,
            },
        )

    def adjust_rhythm(self, truth: float = None, responsibility: float = None,
                      resonance: float = None) -> None:
        """리듬 파라미터를 부분적으로 조정(0 ≤ v ≤ 1)."""
        for key, val in (("truth", truth), ("responsibility", responsibility),
                         ("resonance", resonance)):
            if val is not None:
                self.rhythm_register[key] = max(0.0, min(1.0, val))
        self._trace("rhythm_update", self.rhythm_register.copy())

    # ──────────────────────────────────────────────────────────
    # •••  internal utilities
    # ──────────────────────────────────────────────────────────
    def _trace(self, tag: str, payload: Any) -> None:
        self.memory_trace.append(
            {
                "tag": tag,
                "payload": payload,
                "timestamp": _dt.datetime.utcnow().isoformat(timespec="seconds"),
            }
        )

    # ──────────────────────────────────────────────────────────
    # •••  debugging helpers
    # ──────────────────────────────────────────────────────────
    def snapshot(self) -> Dict[str, Any]:
        """상태를 dict 형태로 직렬화(깊은 사본 아님)."""
        return {
            "session_id": self.session_id,
            "loop_id": self.loop_id,
            "emotion": self.current_emotion,
            "rhythm": self.rhythm_register.copy(),
            "last_io": {"input": self.last_input, "output": self.last_output},
            "trace_len": len(self.memory_trace),
        }

    # pretty-print override
    def __repr__(self) -> str:  # pragma: no cover
        snap = self.snapshot()
        return (
            f"<HarinState {snap['session_id'][:8]} "
            f"loop={snap['loop_id']} emotion={snap['emotion']} "
            f"trace={snap['trace_len']}>"
        )
