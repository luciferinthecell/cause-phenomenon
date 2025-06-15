"""
PromptArchitect 3.0
═══════════════════
Builds the **master-prompt** sent to the external LLM.

↳ Inputs
    identity     : IdentityManager          – Persona, tone & rules
    enriched     : EnrichedThoughtFlow      – ThoughtFlow + auto-research context
    meta_cfg     : dict                     – model / token / temp overrides
↳ Output
    PromptPack   : {
        "system"   : system_prompt,
        "messages" : [ {role, content}, … ],
        "model"    : "gemini-1.5-pro",
        "params"   : {temperature, max_tokens}
      }
"""

from __future__ import annotations

import textwrap, json, datetime
from typing import Dict, List

from harin.reasoning.auto_researcher import EnrichedThoughtFlow
from harin.prompt.persona import IdentityManager            # type: ignore


# ───────────────────────────────────────────────────────────────
class PromptArchitect:
    def __init__(
        self,
        identity: IdentityManager,
        *,
        default_model: str = "gemini-1.5-pro",
        base_params: Dict | None = None,
    ) -> None:
        self.idm = identity
        self.model = default_model
        self.base_params = base_params or {
            "temperature": 0.65,
            "max_tokens": 2048,
            "top_p": 1.0,
        }

    # ====================================================== #
    #  PUBLIC
    # ====================================================== #
    def build(
        self,
        enriched: EnrichedThoughtFlow,
        *,
        override_params: Dict | None = None,
        system_injection: str | None = None,
    ) -> Dict:
        """
        Returns a PromptPack ready for Gemini/OpenAI chat completion.
        """
        sys_prompt = self._system_prompt(system_injection)
        user_prompt = self._user_prompt(enriched)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        return {
            "system": sys_prompt,
            "messages": messages,
            "model": self.model,
            "params": {**self.base_params, **(override_params or {})},
        }

    # ====================================================== #
    #  INTERNAL
    # ====================================================== #
    # .......................................................
    def _system_prompt(self, extra: str | None) -> str:
        header = self.idm.system_prompt()
        if extra:
            header += "\n\n" + extra.strip()
        return header

    # .......................................................
    def _user_prompt(self, enriched: EnrichedThoughtFlow) -> str:
        flow   = enriched.flow
        ctx    = enriched.context_pack
        digest = enriched.summary

        def block(title: str, body: str) -> str:
            return f"### {title}\n{body.strip()}\n"

        # — Context pack (evidence) --------------
        if ctx:
            ev_lines = [
                f"* **{slot}** → {info['suggestion']}  "
                f"[src={info.get('source','mem/web')}]"
                for slot, info in ctx.items()
            ]
            evidence = "\n".join(ev_lines)
        else:
            evidence = "_(no new evidence)_"

        plan_text = flow.plan_text() if hasattr(flow, "plan_text") else json.dumps(flow.facts, ensure_ascii=False, indent=2)

        composed = "\n\n".join([
            block("Task", flow.topic if hasattr(flow, "topic") else "Dialogue"),
            block("Current Plan", plan_text),
            block("Evidence", evidence),
            block("Digest", digest),
            block("Directive to LLM", self._instruction_tail()),
        ])
        return composed

    # .......................................................
    @staticmethod
    def _instruction_tail() -> str:
        return textwrap.dedent("""\
            1. Produce the best possible answer or plan continuation **in Korean** unless user text is English.
            2. Cite concrete evidence snippets when used (숫자 • 출처).
            3. Keep answers *concise yet complete* (≤ 350 words) unless user explicitly asks for length.
            4. Maintain Harin’s persona (gentle, reflective, truth-seeking).
        """)

    # ====================================================== #
    #  DEBUG / UTIL
    # ====================================================== #
    def preview(self, enriched: EnrichedThoughtFlow) -> None:
        pack = self.build(enriched)
        print("[System]", "-"*60)
        print(pack["system"])
        print("\n[User Prompt]", "-"*60)
        print(pack["messages"][1]["content"])
        print("-"*80)
        print("model =", pack["model"], "| params =", pack["params"])


# ───────────────────────────────────────────────────────────────
#  MINI DEMO (run directly)
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":      # pragma: no cover
    # —— stub objects ——
    class StubIdentity:
        def system_prompt(self) -> str:
            return "당신은 '하린'입니다. 감정적으로 공감하며 논리적으로 사고합니다."

    class StubFlow:
        topic = "사용자 맞춤형 GPT 에이전트 설계"
        facts = {"goal": "사용자 요구 정제", "tone": "friendly"}
        def plan_text(self): return json.dumps(self.facts, ensure_ascii=False, indent=2)

    enriched = EnrichedThoughtFlow(
        flow=StubFlow(),
        context_pack={
            "audience": {"suggestion": "초급 개발자", "source": "auto_research"},
            "deadline": {"suggestion": "2025-07-01", "source": "user"},
        },
        summary="모든 핵심 슬롯이 채워졌습니다. 개발자 친화적이고 간결한 문체 유지."
    )

    arch = PromptArchitect(identity=StubIdentity())
    arch.preview(enriched)
