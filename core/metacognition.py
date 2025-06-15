"""
harin.core.metacognition
~~~~~~~~~~~~~~~~~~~~~~~~

Self‑reflection engine v2.1 – **context‑aware & backward‑compatible**

변경 사항
─────────
1. `SelfReflector.reflect(judgments, context:UserContext | None = None)`  
   • context가 주어지면 mood / mode / flow 포함한 확장 반추  
   • context가 None이면 v1과 동일한 기본 요약 로직 동작 (기존 코드 호환)
2. 리플렉션 텍스트 → `ReflectionWriter` 통해 메타와 함께 MemoryEngine 저장
"""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable, Dict, Any

from .judgment import Judgment
from harin.memory.adapter import MemoryEngine

# optional import to avoid circular
try:
    from .context import UserContext  # type: ignore
except ImportError:
    UserContext = Any  # fallback for type checkers


@runtime_checkable
class Reflector(Protocol):
    def reflect(self, txt: str) -> str: ...


class MockReflector:
    def reflect(self, txt: str) -> str:  # noqa: D401
        return f"[reflection]\n{txt}"


class SelfReflector:
    """Generate self‑reflection that merges Judgment info and (optionally) UserContext."""

    def __init__(self, reflector: Reflector | None = None):
        self.reflector: Reflector = reflector or MockReflector()

    def reflect(
        self,
        judgments: List[Judgment],
        *,
        context: UserContext | None = None,
    ) -> str:
        """Return reflection text.

        If *context* is provided → include mood/mode/flow; else legacy summary only.
        """
        if context is None:
            # legacy behaviour (keep original functionality)
            lines = [f"Loop {j.loop_id}: {j.score.overall():.2f}" for j in judgments]
            composite = " | ".join(lines)
            return self.reflector.reflect(composite)

        # context‑aware reflection
        ctx_summary = (
            f"Mood={context.mood}; Mode={context.last_mode}; Flow={'→'.join(context.context_trace[-4:])}"
        )
        j_lines = [f"{j.loop_id}:{j.score.overall():.2f}" for j in judgments]
        composite = ctx_summary + "\n" + " | ".join(j_lines)
        return self.reflector.reflect(composite)


class ReflectionWriter:
    def __init__(self, memory: MemoryEngine):
        self.memory = memory

    def write(self, reflection_text: str) -> str:
        node = self.memory.store(
            reflection_text,
            node_type="reflection",
            vectors={"E": 1.0},
            meta={"generated_by": "metacognition"},
        )
        return node.id
"""
