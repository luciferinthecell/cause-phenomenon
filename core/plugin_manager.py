"""
harin.core.prompt_architect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PromptArchitect v2 – 의미 기반 컨텍스트 흐름까지 포함하는 구조화 프롬프트 생성기

기능
─────
1. SYSTEM / CONTEXT / INSTRUCTION 블록 구분
2. UserContext의 감정·목표 + 모드 + 컨텍스트 트레이스 포함
3. Judgment 결과를 INSTRUCTION에 반영
4. reflection 포함 시 요약 블록 삽입

최종 프롬프트는 LLMClient.complete(prompt)로 호출됨.
"""

from __future__ import annotations

from typing import List, Optional

from .context import UserContext
from .judgment import Judgment


class PromptArchitect:
    @staticmethod
    def _fmt_block(title: str, body: str) -> str:
        return f"<<{title}>>\n{body}\n<</{title}>>\n"

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
        blocks.append(cls._fmt_block("SYSTEM", system_base.strip()))

        # CONTEXT
        ctx_lines = [
            f"Mood: {context.mood}",
            f"Cognition: {context.cognitive_level}",
            f"Goal: {', '.join(context.goal_keywords[:5])}",
            f"Mode: {context.last_mode}"
        ]
        if context.context_trace:
            ctx_lines.append("Context flow: " + " → ".join(context.context_trace))
        if memory_snippets:
            ctx_lines.append("Memory: " + " | ".join(memory_snippets))
        if reflection:
            ctx_lines.append("Reflection: " + reflection)
        blocks.append(cls._fmt_block("CONTEXT", "\n".join(ctx_lines)))

        # INSTRUCTION
        instr = [f"User said: {user_input.strip()}"]
        if best_judgment:
            instr.append("Prior reasoning: " + best_judgment.output_text)
        instr.append("Now respond meaningfully.")
        blocks.append(cls._fmt_block("INSTRUCTION", "\n".join(instr)))

        return "\n".join(blocks)
"""
