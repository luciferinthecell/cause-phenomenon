"""
harin.evaluators.verification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Purpose*   Verify **freshness, trust, and user-fit** of information that
            comes from an external search / API call.

Design notes
------------

• No fantasy “Velith/Lysara/Ashariel” scoring.  
• Pure heuristics + lightweight NLP; can be swapped with ML later.  
• Uses three signals:

  1. **Source trust**   (domain allow-list + basic HTTPS / author check)
  2. **Time freshness** (days since «date»)            → recency_score
  3. **User–fit**       – relevance (goal keywords)  
                        – knowledge gap vs. `user_profile["known_terms"]`  
                        – novelty  (was the fact already in Palantir memory?)

Composite score ∈ [0, 1].  Down-stream code may keep only entries ≥ 0.6.
"""

from __future__ import annotations

import re
import math
import datetime as _dt
from dataclasses import dataclass, field
from typing import List, Dict, Any, Iterable, Tuple

from dateutil import parser as _date_parse

# local import – memory graph
from ..memory.palantir import PalantirGraph


# --------------------------------------------------------------------------- #
#  Utility helpers
# --------------------------------------------------------------------------- #


_TRUSTED_DOMAINS = {
    # news / science
    "reuters.com",
    "apnews.com",
    "bbc.co.uk",
    "nature.com",
    "science.org",
    # tech-blogs
    "openai.com",
    "google.com",
    "arxiv.org",
}


def _domain_of(url: str) -> str:
    m = re.match(r"https?://([^/]+)/?", url)
    return m.group(1) if m else ""


def _normalize(text: str) -> List[str]:
    return re.sub(r"[^A-Za-z0-9가-힣 ]+", " ", text.lower()).split()


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# --------------------------------------------------------------------------- #
#  User profile dataclass
# --------------------------------------------------------------------------- #


@dataclass
class UserProfile:
    """Very coarse user model supplied by caller."""

    knowledge_terms: List[str] = field(default_factory=list)   # words user already knows
    goal_terms: List[str] = field(default_factory=list)        # what user *wants*
    cognitive_level: str = "normal"                           # or: beginner / expert

    def novelty_against_user(self, content_tokens: List[str]) -> float:
        """Portion of tokens *not* already known by user (0 = none new, 1 = all new)."""
        if not content_tokens:
            return 0.0
        unknown = [t for t in content_tokens if t not in self.knowledge_terms]
        return len(unknown) / len(content_tokens)


# --------------------------------------------------------------------------- #
#  Main engine
# --------------------------------------------------------------------------- #


class InfoVerificationEngine:
    """Stateless evaluator – just feed JSON blob + context, get score dict."""

    # ---- Recency ---------------------------------------------------------

    @staticmethod
    def _recency_score(date_str: str, today: _dt.date | None = None) -> float:
        """
        Map *days old* → score in [0,1].

        0 days → 1.0   • 90 days → 0.0 (linear decay).  
        “date_str” may be YYYY-MM-DD… anything dateutil can parse.
        """
        today = today or _dt.date.today()
        try:
            article_date = _date_parse.parse(date_str).date()
        except Exception:
            return 0.0  # unknown date → cannot trust recency
        days_old = (today - article_date).days
        return max(0.0, 1 - days_old / 90)

    # ---- Trust -----------------------------------------------------------

    @staticmethod
    def _domain_trust(url: str) -> float:
        """
        1.0   trusted list  
        0.5   https but unlisted  
        0.2   http or suspicious
        """
        dom = _domain_of(url)
        if dom in _TRUSTED_DOMAINS:
            return 1.0
        if url.startswith("https://"):
            return 0.5
        return 0.2

    # ---- User-fit & Novelty ---------------------------------------------

    @staticmethod
    def _goal_relevance(tokens: List[str], goal_terms: List[str]) -> float:
        if not goal_terms:
            return 0.5
        return _jaccard(tokens, goal_terms)

    @staticmethod
    def _novelty_vs_memory(tokens: List[str], memory: PalantirGraph) -> float:
        """
        0 → already well-covered in memory  
        1 → entirely new concept
        """
        if not memory.nodes():
            return 1.0
        mem_tokens: List[str] = []
        for n in memory.find_recent(limit=200):
            mem_tokens.extend(_normalize(n.content))
        return 1 - _jaccard(tokens, mem_tokens)

    # ---- Composite evaluation -------------------------------------------

    @classmethod
    def evaluate_single(
        cls,
        info: Dict[str, Any],
        user: UserProfile,
        memory: PalantirGraph | None = None,
        today: _dt.date | None = None,
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        info
            {
              "title": str,
              "content": str,     # raw snippet or article body
              "date": "2025-05-10",
              "url": "https://reuters.com/…"
            }
        user
            *UserProfile* instance
        memory
            PalantirGraph for novelty metric (optional)
        """
        memory = memory or PalantirGraph()
        tokens = _normalize(info.get("content", "") + " " + info.get("title", ""))
        recency   = cls._recency_score(info.get("date", ""), today)
        trust     = cls._domain_trust(info.get("url", ""))
        relevance = cls._goal_relevance(tokens, user.goal_terms)
        novelty_u = user.novelty_against_user(tokens)
        novelty_m = cls._novelty_vs_memory(tokens, memory)

        # weight vector (tweak as needed)
        w = {"trust": 0.25, "recency": 0.25, "relevance": 0.25,
             "novelty_user": 0.15, "novelty_memory": 0.10}

        composite = (
            w["trust"]          * trust +
            w["recency"]        * recency +
            w["relevance"]      * relevance +
            w["novelty_user"]   * novelty_u +
            w["novelty_memory"] * novelty_m
        )

        return {
            "title": info.get("title", "")[:120],
            "url": info.get("url"),
            "recency": round(recency, 2),
            "trust": trust,
            "relevance": round(relevance, 2),
            "novelty_user": round(novelty_u, 2),
            "novelty_memory": round(novelty_m, 2),
            "score": round(composite, 3),
        }

    @classmethod
    def evaluate_batch(
        cls,
        batch: List[Dict[str, Any]],
        user: UserProfile,
        memory: PalantirGraph | None = None,
        drop_below: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """Return best→worst but keep only items ≥ drop_below."""
        scored = [
            cls.evaluate_single(item, user, memory)
            for item in batch
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return [s for s in scored if s["score"] >= drop_below]
