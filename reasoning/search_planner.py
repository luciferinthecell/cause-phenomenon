"""
SearchPlanner
═════════════
Memory-aware, self-evaluating planner that crafts **multi-stage
web-search strategies** for Harin.

Motivation
──────────
• PromptArchitect/AutoResearcher ➜ ‘slot’ 단위 검색(1-hop)          → 충분하지 않다
• 우리는 Harin의 *팔란티어 기억* + 현재 ThoughtFlow 를 활용해
  ▸ 쿼리 후보를 다각도로 생성
  ▸ 실행하며 품질·커버리지·신뢰도를 스스로 판정
  ▸ 부족하면 새 키워드로 **loop** (max_iter)
  ▸ 최종적으로 “컨텍스트 팩” 반환 → PromptArchitect 로 전달

Key Concepts
────────────
SearchGoal          –  어떤 정보가 필요한지 선언적 정의
QueryCandidate      –  single search string  (+ provenance score)
SearchEpisode       –  {query, results, coverage_score, trust_score}

Pipeline
────────
     build_goals()      ← ThoughtFlow, missing_slots, memory
        ↓
     propose_queries()  – n probes per goal (semantic & keyword blend)
        ↓
     run_queries()      – via injected WebSearchClient
        ↓
     evaluate_episode() – recency + domain_cred + semantic_relevance
        ↓
     loop (if coverage < θ)  ↺ mutate_queries()
        ↓
     pack_context()     – pick best hits per goal (top-k)
"""

from __future__ import annotations

import random
import re
import statistics as _stats
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

# externals (already provided elsewhere)
from harin.memory.adapter import MemoryEngine               # type: ignore
from harin.tools.websearch import WebSearchClient           # type: ignore
from harin.reasoning.thought_processor import ThoughtFlow   # type: ignore


# ────────────────────────────────────────────────────────────────────────────
#  Dataclasses
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class SearchGoal:
    slot: str
    description: str
    importance: float = 1.0


@dataclass
class QueryCandidate:
    text: str
    source: str      # "memory", "flow", "fallback"
    score: float = 1.0


@dataclass
class SearchEpisode:
    goal: SearchGoal
    query: QueryCandidate
    hits: List[Dict]
    coverage: float
    trust: float


# ────────────────────────────────────────────────────────────────────────────
class SearchPlanner:
    """
    Parameters
    ----------
    memory          : MemoryEngine – semantic recall()
    search_client   : WebSearchClient – executes searches
    max_iter        : loop iterations per goal
    """

    _RE_PUNCT = re.compile(r"[^\w\s]")

    def __init__(
        self,
        memory: MemoryEngine,
        search_client: WebSearchClient,
        *,
        max_iter: int = 3,
    ) -> None:
        self.mem = memory
        self.search = search_client
        self.max_iter = max_iter

    # ==================================================================== #
    #  PUBLIC ENTRY
    # ==================================================================== #
    def fulfil(
        self,
        missing_slots: Sequence[str],
        flow: ThoughtFlow,
        user_profile: Dict[str, str] | None = None,
    ) -> Dict[str, Dict]:
        """
        Returns
        -------
        slot → {
            "suggestion" : str,
            "evidence"   : List[Dict]   # {title, href, body, score}
        }
        """
        goals = self._build_goals(missing_slots, flow)
        context_pack: Dict[str, Dict] = {}

        for g in goals:
            episodes = self._run_goal(g, flow)
            best_ep = max(episodes, key=lambda e: e.coverage * e.trust)

            # store evidence (top-3 hits)
            context_pack[g.slot] = {
                "suggestion": self._derive_value(g.slot, best_ep),
                "evidence"  : [
                    {
                        "title": h["title"],
                        "href": h["href"],
                        "snippet": h["body"],
                        "score": (best_ep.coverage * best_ep.trust),
                    }
                    for h in best_ep.hits[:3]
                ],
            }

        return context_pack

    # ==================================================================== #
    #  GOAL GENERATION
    # ==================================================================== #
    def _build_goals(self, slots: Sequence[str], flow: ThoughtFlow) -> List[SearchGoal]:
        """weight = importance; can be improved with flow.meta."""
        return [
            SearchGoal(
                slot=s,
                description=f"Find resonant information for slot “{s}” given topic {flow.topic}",
                importance=1.2 if s == "goal" else 1.0,
            )
            for s in slots
        ]

    # ==================================================================== #
    #  LOOP PER GOAL
    # ==================================================================== #
    def _run_goal(self, goal: SearchGoal, flow: ThoughtFlow) -> List[SearchEpisode]:
        episodes: List[SearchEpisode] = []
        queries = self._propose_queries(goal, flow)

        for it in range(self.max_iter):
            if not queries:
                break

            q = queries.pop(0)
            hits = self.search.search(q.text, k=8)
            cov, trust = self._evaluate_hits(goal, hits)

            episodes.append(SearchEpisode(goal, q, hits, cov, trust))

            # success criterion
            if cov * trust >= 0.65:
                break

            # else produce mutations for next loop
            queries.extend(self._mutate_query(q, hits, factor=it + 1))

        return episodes

    # ==================================================================== #
    #  QUERY PROPOSAL
    # ==================================================================== #
    def _propose_queries(self, goal: SearchGoal, flow: ThoughtFlow) -> List[QueryCandidate]:
        memory_kw = self._keywords_from_memory(flow.topic)
        flow_kw = self._keywords_from_flow(flow)

        candidates: List[QueryCandidate] = []

        for base in [goal.slot] + flow_kw[:2]:
            q = f"{base} {flow.topic}"
            candidates.append(QueryCandidate(text=q, source="flow", score=1.0))

        # memory-derived combos
        for kw in memory_kw[:2]:
            q = f"{goal.slot} {kw}"
            candidates.append(QueryCandidate(text=q, source="memory", score=0.9))

        random.shuffle(candidates)
        return candidates

    # .....................................................................
    def _keywords_from_memory(self, topic: str) -> List[str]:
        rec = self.mem.recall(topic, top_k=5)
        kw: List[str] = []
        for m in rec:
            words = self._tokenise(m.text)[:5]
            kw.extend(words)
        return self._dedup(kw)

    def _keywords_from_flow(self, flow: ThoughtFlow) -> List[str]:
        words = self._tokenise(flow.plan.get("summary", ""))[:8]
        return self._dedup(words)

    # ==================================================================== #
    #  HIT EVALUATION
    # ==================================================================== #
    def _evaluate_hits(self, goal: SearchGoal, hits: List[Dict]) -> Tuple[float, float]:
        if not hits:
            return 0.0, 0.0

        # coverage – simplistic: fraction containing slot keyword
        cov = sum(1 for h in hits if goal.slot.lower() in h["body"].lower()) / len(hits)

        # trust – domain + recency (reuse WebSearchClient logic)
        dom = sum(1 for h in hits if ".gov" in h["href"] or ".org" in h["href"]) / len(hits)
        rec = sum(1 for h in hits if "published" in h) / len(hits)
        trust = _stats.mean([dom, rec])

        return cov, trust

    # ==================================================================== #
    #  QUERY MUTATION
    # ==================================================================== #
    def _mutate_query(self, q: QueryCandidate, hits: List[Dict], *, factor: int) -> List[QueryCandidate]:
        """Add synonyms or longest common substring from first hit."""
        if not hits:
            return []

        body = hits[0]["body"]
        tokens = self._tokenise(body)
        # pick rare-ish tokens
        pick = [t for t in tokens if 5 <= len(t) <= 12][:2]
        mutated = [
            QueryCandidate(text=f"{w} {q.text}", source="mutated", score=q.score * 0.8)
            for w in pick
        ]
        # exponential backoff on #mutations
        return mutated[: max(1, 3 - factor)]

    # ==================================================================== #
    #  VALUE DERIVATION
    # ==================================================================== #
    def _derive_value(self, slot: str, ep: SearchEpisode) -> str:
        if slot == "goal":
            return ep.hits[0]["title"][:80]
        if slot == "audience":
            for tok in ("developer", "investor", "student", "CEO", "manager"):
                if tok in ep.hits[0]["snippet"].lower():
                    return tok
        if slot == "tone":
            for tone in ("formal", "informal", "friendly", "professional", "persuasive"):
                if tone in ep.hits[0]["snippet"].lower():
                    return tone
        if slot == "deadline":
            if any(w in ep.hits[0]["snippet"] for w in ("hour", "24h")):
                return "24h"
            if "week" in ep.hits[0]["snippet"]:
                return "1 week"
        # fallback
        return ep.hits[0]["title"][:60] + "…"

    # ==================================================================== #
    #  UTILS
    # ==================================================================== #
    def _tokenise(self, text: str) -> List[str]:
        cleaned = self._RE_PUNCT.sub(" ", text.lower())
        return [w for w in cleaned.split() if len(w) > 3]

    @staticmethod
    def _dedup(seq: Sequence[str]) -> List[str]:
        seen: set[str] = set()
        out: List[str] = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out
