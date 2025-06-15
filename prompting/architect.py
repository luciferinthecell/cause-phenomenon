"""
PromptArchitect
════════════════
Builds the *master-prompt* sent to the external LLM **and**
runs an interview-style clarification loop when information
is missing.

Workflow
────────
1)  analyse_missing()         → list[str]        ┐  (auto-called)
2)  generate_questions()      → dict[id,q]       │  ask the user
3)  ingest_answers(dict[id,answer])              │  feed replies
   — repeat until pending_slots == ∅ —           ┘
4)  build_master_prompt()     → str              (LLM ready)

Dependencies
────────────
• IdentityManager  – stable persona paragraphs
• MemoryEngine     – recall() for context snippets
• ThoughtFlow      – produced by ThoughtProcessor

The class keeps *session* state internally, so HarinAgent can
instantiate once per dialogue turn.
"""

from __future__ import annotations

import textwrap
from typing import Any, Dict, List, Sequence

from harin.identity.manager import IdentityManager  # type: ignore
from harin.memory.adapter import MemoryEngine       # type: ignore
from harin.reasoning.thought_processor import ThoughtFlow  # type: ignore


# ────────────────────────────────────────────────────────────────────────────
#  Clarification schema
# ────────────────────────────────────────────────────────────────────────────
_STANDARD_SLOTS: Dict[str, str] = {
    "goal": "당신의 최종 목표는 무엇인가요? (ex: 투자자용 1-page 요약)",
    "audience": "이 작업 결과물을 읽게 될 주요 대상은 누구인가요?",
    "tone": "원하는 문체/톤이 있나요? (예: 격식, 대화체)",
    "deadline": "완료 시점(또는 긴급도)을 알려주세요.",
    "constraints": "반드시 지켜야 할 제약 조건이 있나요?",
}

# each plan may include   plan["required_slots"]   list[str]
_FALLBACK_SLOTS = ["goal", "audience"]


# ────────────────────────────────────────────────────────────────────────────
#  PromptArchitect
# ────────────────────────────────────────────────────────────────────────────
class PromptArchitect:
    def __init__(
        self,
        *,
        identity_mgr: IdentityManager,
        memory_engine: MemoryEngine,
    ) -> None:
        self.persona = identity_mgr.get_persona_block()  # rich string
        self.memory = memory_engine

        # interview state
        self.pending_slots: List[str] = []
        self.answers: Dict[str, str] = {}

    # ------------------------------------------------------------------ #
    #  Step-1  Missing checker
    # ------------------------------------------------------------------ #
    def analyse_missing(self, flow: ThoughtFlow, user_profile: Dict[str, Any]) -> List[str]:
        """Fill `self.pending_slots` and return it."""
        required = flow.plan.get("required_slots", _FALLBACK_SLOTS)
        self.pending_slots = [s for s in required if s not in user_profile and s not in self.answers]
        return self.pending_slots

    # ------------------------------------------------------------------ #
    #  Step-2  Question generator
    # ------------------------------------------------------------------ #
    def generate_questions(self) -> Dict[str, str]:
        """Return {slot: natural_language_question}"""
        return {s: _STANDARD_SLOTS.get(s, f"‘{s}’에 대한 구체적 정보를 알려주세요.") for s in self.pending_slots}

    # ------------------------------------------------------------------ #
    #  Step-3  Store user replies
    # ------------------------------------------------------------------ #
    def ingest_answers(self, answer_map: Dict[str, str]) -> None:
        self.answers.update(answer_map)
        # remove satisfied slots
        self.pending_slots = [s for s in self.pending_slots if s not in answer_map]

    # ------------------------------------------------------------------ #
    #  Step-4  Final prompt builder
    # ------------------------------------------------------------------ #
    def build_master_prompt(
        self,
        flow: ThoughtFlow,
        user_profile: Dict[str, Any],
        *,
        max_mem_snippets: int = 2,
    ) -> str:
        """
        Return a single **multi-part** prompt with:
            ① Persona system message
            ② Objective + constraints
            ③ Thought plan + meta reflection
            ④ Memory context
            ⑤ Chain-of-thought request (multi-step reasoning)
        """

        # 0) ensure no missing info
        if self.pending_slots:
            raise RuntimeError("Clarification required before prompt build.")

        # 1) merge profile + interview answers
        context_info = {**user_profile, **self.answers}

        # 2) memory snippets
        memories = self.memory.recent(max_mem_snippets)
        mem_block = "\n".join(f"• {m.text[:120]}…" for m in memories) if memories else "None."

        # 3) assemble
        objective_lines = [f"{k.capitalize()}: {v}" for k, v in context_info.items()]
        objective_block = "\n".join(objective_lines)

        plan_block = textwrap.indent(
            textwrap.fill(str(flow.plan), width=88),
            prefix=" " * 4,
        )
        meta_block = f"Trust-score {flow.metacognition['trust_score']} │ {flow.metacognition['reflection']}"

        # 4) reasoning instructions – force multi-step chain-of-thought
        reasoning_instr = (
            "### Reasoning Instructions\n"
            "1. Think step-by-step, referencing the plan.\n"
            "2. If external evidence is required, list a short bullet per source.\n"
            "3. Finally, produce **ONLY** the answer in requested style."
        )

        prompt = textwrap.dedent(
            f"""
            ### System Persona
            {self.persona}

            ### Task Objective
            {objective_block}

            ### Harin Thought-Plan
            {plan_block}

            ### Self-Reflection
            {meta_block}

            ### Memory Context
            {mem_block}

            {reasoning_instr}
            """
        ).strip()

        return prompt
