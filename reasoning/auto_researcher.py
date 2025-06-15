"""
AutoResearcher
══════════════
High-level *gap-filler* orchestrator that:
  1️⃣ Detects missing knowledge slots in ThoughtFlow
  2️⃣ Calls SearchPlanner for evidence
  3️⃣ Summarises results with an LLM                              (Gemini/OpenAI)
  4️⃣ Stores distilled knowledge back into MemoryEngine
  5️⃣ Returns an **EnrichedThoughtFlow** ready for PromptArchitect

The class is fully asynchronous-ready (awaitable search / LLM if desired)
but works synchronously by default for easier drop-in use.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

from harin.memory.engine import MemoryEngine
from harin.reasoning.search_planner import SearchPlanner, SearchEpisode
from harin.tools.websearch import WebSearchClient
from harin.prompt.persona import IdentityManager          # type: ignore
from harin.llm_clients.gemini_client import GeminiClient   # type: ignore
from harin.reasoning.thought_processor import ThoughtFlow  # type: ignore


# ────────────────────────────────────────────────────────────────────────────
@dataclass
class EnrichedThoughtFlow:
    flow: ThoughtFlow
    context_pack: Dict[str, Dict]          # slot → {"suggestion", "evidence"}
    summary: str                           # LLM-generated digest
    updated_at: float = time.time()


# ────────────────────────────────────────────────────────────────────────────
class AutoResearcher:
    """
    Parameters
    ----------
    memory          : MemoryEngine
    search_client   : WebSearchClient
    persona         : IdentityManager (tone for summariser)
    llm             : GeminiClient (or any object with `.complete(prompt)`)
    planner_cfg     : dict – forwarded to SearchPlanner (e.g., max_iter)
    """

    def __init__(
        self,
        memory: MemoryEngine,
        search_client: WebSearchClient,
        persona: IdentityManager,
        llm: GeminiClient,
        planner_cfg: Dict | None = None,
    ) -> None:
        self.mem = memory
        self.persona = persona
        self.llm = llm
        self.planner = SearchPlanner(memory, search_client, **(planner_cfg or {}))

    # ==================================================================== #
    #  PUBLIC
    # ==================================================================== #
    def enrich(self, flow: ThoughtFlow) -> EnrichedThoughtFlow:
        missing = self._detect_missing(flow)
        if not missing:
            return EnrichedThoughtFlow(flow, {}, summary="No gaps detected.")

        ctx = self.planner.fulfil(missing, flow)
        digest = self._summarise(flow, ctx)
        self._write_back(ctx)

        # patch ThoughtFlow inplace
        for slot, obj in ctx.items():
            flow.facts[slot] = obj["suggestion"]

        return EnrichedThoughtFlow(flow, ctx, digest)

    # ==================================================================== #
    #  INTERNAL
    # ==================================================================== #
    @staticmethod
    def _detect_missing(flow: ThoughtFlow) -> List[str]:
        required = {"goal", "audience", "tone", "deadline"}
        return [s for s in required if s not in flow.facts or not flow.facts[s]]

    # ......................................................................
    def _summarise(self, flow: ThoughtFlow, ctx: Dict[str, Dict]) -> str:
        bullets = "\n".join(
            f"• **{slot}** → {info['suggestion']}" for slot, info in ctx.items()
        )
        sys_prompt = self.persona.system_prompt()
        user_msg = (
            f"Update the plan below by filling missing slots.\n\n"
            f"### Current plan\n{flow.plan_text()}\n\n"
            f"### Fetched evidence\n{bullets}\n\n"
            f"Produce a concise summary in **max 90 words**."
        )
        resp = self.llm.complete([{"role": "system", "content": sys_prompt},
                                   {"role": "user", "content": user_msg}],
                                 temperature=0.3)
        return resp.strip()

    # ......................................................................
    def _write_back(self, ctx: Dict[str, Dict]) -> None:
        for info in ctx.values():
            text = info["suggestion"]
            meta = {"origin": "auto_research", "tags": ["summarised"]}
            self.mem.add(text, meta=meta, trust=0.8)

    # ==================================================================== #
    #  BATCH API  (optional)
    # ==================================================================== #
    def enrich_many(self, flows: Sequence[ThoughtFlow]) -> List[EnrichedThoughtFlow]:
        return [self.enrich(f) for f in flows]


# ────────────────────────────────────────────────────────────────────────────
# Example glue (pseudo-code)
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":  # pragma: no cover
    from stub import embed_fn, vectordb, web_backend      # placeholders

    mem = MemoryEngine(embed_fn, vectordb)
    web_client = WebSearchClient(backend=web_backend)
    persona = IdentityManager("Harin", version="v3.2")
    llm = GeminiClient(api_key="YOUR_API_KEY")

    researcher = AutoResearcher(mem, web_client, persona, llm)

    thought_flow = ThoughtFlow(topic="LLM fine-tuning")  # Simplified stub
    enriched = researcher.enrich(thought_flow)
    print("Summary:", enriched.summary)
