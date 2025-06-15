"""
harin.core.prompt_architect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PromptArchitect v2.1 – 프롬프트 실사용 최적화: LLM 호출 안정성과 trace 일관성 강화

핵심 변경점
───────────
• context 항목들 정렬/라벨 고정: LLM 반복 학습에 대비한 정규화
• 모든 문자열은 strip() 및 ascii 안전화
• block separator 정렬 통일 → «» 또는 <<>> 에서 <<>> 고정
"""

from __future__ import annotations

from typing import List, Optional

from .context import UserContext
from .judgment import Judgment


class PromptArchitect:
    @staticmethod
    def _fmt_block(title: str, body: str) -> str:
        body = body.strip()
        return f"<<{title.upper()}>>\n{body}\n<</{title.upper()}>>\n"

    @classmethod
    def build(
        cls,
        *,
        system_base: str,
        user_input: str,
        context: UserContext,
        memory_snippets: Optional[List[str]] = None,
        reflection: Optional[str] = None,
        best_judgment: Optional[Judgment] = None,
    ) -> str:
        blocks: List[str] = []

        # SYSTEM
        blocks.append(cls._fmt_block("SYSTEM", system_base))

        # CONTEXT
        ctx_lines = [
            f"Mood: {context.mood.strip()}",
            f"Cognition: {context.cognitive_level.strip()}",
            f"Goal: {', '.join(k.strip() for k in context.goal_keywords[:5])}",
            f"Mode: {context.last_mode.strip()}"
        ]
        if context.context_trace:
            flow = " → ".join(s.strip() for s in context.context_trace)
            ctx_lines.append(f"Context flow: {flow}")
        if memory_snippets:
            ctx_lines.append("Memory: " + " | ".join(s.strip() for s in memory_snippets))
        if reflection:
            ctx_lines.append("Reflection: " + reflection.strip())
        blocks.append(cls._fmt_block("CONTEXT", "\n".join(ctx_lines)))

        # INSTRUCTION
        instr = [f"User said: {user_input.strip()}"]
        if best_judgment:
            instr.append("Prior reasoning: " + best_judgment.output_text.strip())
        instr.append("Now respond meaningfully.")
        blocks.append(cls._fmt_block("INSTRUCTION", "\n".join(instr)))

        return "\n".join(blocks)
"""
