"""
ThoughtProcessor 2.1
════════════════════
Orchestrates Harin’s **tree-of-thought** pipeline.

Pipeline
────────
 1.  ⇢  _preflight()           – sanitise + light intent pass
 2.  ⇢  ExpertRouter.route()   – choose expert(s)
 3.  ⇢  each expert.run()      – produce PartialThought
 4.  ⇢  Metacognition.score()  – trust / coherence
 5.  ⇢  AutoResearcher.enrich()– fill missing slots
 6.  ⇢  return ThoughtFlow     – final structured object
"""

from __future__ import annotations
import datetime, uuid
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

# ── local deps
from harin.memory.engine import MemoryEngine
from harin.reasoning.auto_researcher import AutoResearcher, EnrichedThoughtFlow
from harin.reasoning.expert_router import ExpertRouter, PartialThought   # you already created / will create
from harin.reasoning.metacognition import Metacognition                 # same
from harin.prompt.persona import IdentityManager                        # same


# ───────────────────────────────────────────────────────────────────────
@dataclass
class ThoughtFlow:
    id: str
    topic: str
    facts: Dict[str, str] = field(default_factory=dict)
    partials: List[PartialThought] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())

    # convenient helpers
    def plan_text(self) -> str:
        import json, textwrap
        body = json.dumps(self.facts, ensure_ascii=False, indent=2)
        parts = "\n".join(p.render() for p in self.partials)
        return textwrap.dedent(f"""
            ▶ Facts
            {body}

            ▶ Thought Steps
            {parts}
        """).strip()


# ───────────────────────────────────────────────────────────────────────
class ThoughtProcessor:
    """
    glue for MemoryEngine • ExpertRouter • Metacognition • AutoResearcher
    --------------------------------------------------------------------
    You can inject custom ExpertRouter / Metacognition for testing.
    """

    def __init__(
        self,
        memory: MemoryEngine,
        researcher: AutoResearcher,
        router: ExpertRouter,
        meta: Metacognition,
    ) -> None:
        self.mem  = memory
        self.rs   = researcher
        self.router = router
        self.meta = meta

    # ================================================================ #
    #  PUBLIC
    # ================================================================ #
    def run(self, user_text: str, *, ctx: Dict | None = None) -> EnrichedThoughtFlow:
        """
        Full processing from raw user input to enriched ThoughtFlow
        ready for PromptArchitect.
        """
        ctx = ctx or {}
        flow = self._preflight(user_text, ctx)

        # 2. expert routing  + execution
        experts = self.router.route(user_text)
        for exp in experts:
            part = exp.run(user_text, context=ctx)
            flow.partials.append(part)

        # 3. meta-evaluation  (confidence, issues …)
        trust = self.meta.evaluate(flow.partials)
        flow.facts["trust_score"] = f"{trust.score:.2f}"
        flow.facts["self_reflection"] = trust.note

        # 4. auto-research for missing slots
        enriched = self.rs.enrich(flow)
        return enriched

    # ================================================================ #
    #  INTERNAL
    # ================================================================ #
    def _preflight(self, user_text: str, ctx: Dict) -> ThoughtFlow:
        # naive slot-fill from ctx
        facts = {
            "goal": ctx.get("goal") or "",
            "audience": ctx.get("audience") or "",
            "tone": ctx.get("tone") or "",
            "deadline": ctx.get("deadline") or "",
        }
        tid = f"TF-{uuid.uuid4().hex[:8]}"
        return ThoughtFlow(id=tid, topic=user_text.strip()[:60], facts=facts)


# ───────────────────────────────────────────────────────────────────────
#  Quick demo stub
# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":  # pragma: no cover
    from stub import (
        embed_fn, vectordb,
        DummySearchClient, DummyExpertRouter, DummyExpert, DummyMetacog,
        DummyPersona, DummyLLM
    )

    mem = MemoryEngine(embed_fn, vectordb)
    researcher = AutoResearcher(
        mem,
        DummySearchClient(),
        DummyPersona(),
        DummyLLM()
    )
    tp = ThoughtProcessor(
        mem,
        researcher,
        DummyExpertRouter([DummyExpert()]),
        DummyMetacog()
    )

    result = tp.run("사이드 프로젝트용 GPT 요약봇 설계를 돕고 싶어.")
    print("[Digest]", result.summary)


# === Harin Patch Injection ===

# === Sandbox Fallback ===
from HarinSandboxRunner import HarinSandboxRunner

if decision["confidence"] < 0.5:
    sandbox = HarinSandboxRunner()
    sb_result = sandbox.process(input_text)
    print("Fallback Result:", sb_result)
