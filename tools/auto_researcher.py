"""
AutoResearcher
══════════════
When the conversation lacks critical parameters (*slots*), this helper:

• Generates a focused web-query for each missing slot.
• Executes the query through an injectable `search_func`.
• Picks the most recent / credible snippet.
• Returns ⇒ { slot : { "suggestion": str, "sources": […] } }

Notes
-----
• No external dependency hard-wired: pass in any callable that performs
  the actual search (DuckDuckGo, SerpAPI, web.run wrapper …).
• Recency cut-off is configurable; defaults to 120 days.
• “Credibility” is a naive keyword match against preferred domains,
  but can be replaced with your own scorer.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

# quick date regex (YYYY-MM-DD or 2025/04/12 etc.)
_DATE_RX = re.compile(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})")


# ────────────────────────────────────────────────────────────────────────────
@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    published: _dt.date | None = None
    raw: Dict | None = None   # keep vendor-specific blob for traceability


# preferred trustworthy domains (edit to taste)
_TRUSTED = (
    "wikipedia.org",
    "reuters.com",
    "nature.com",
    "arxiv.org",
    "nytimes.com",
    "github.com",
)

# ---------------------------------------------------------------------------


class AutoResearcher:
    """Plug-and-play research companion."""

    def __init__(
        self,
        search_func: Callable[[str, int], Sequence[Dict]],
        *,
        recency_days: int = 120,
    ) -> None:
        """
        Parameters
        ----------
        search_func   a function like  lambda q, k: List[dict]
                      Each dict should expose at least 'title','href','body'.
        recency_days  ignore articles older than this.
        """
        self._search = search_func
        self._max_age = recency_days

    # ------------------------------------------------------------------ #
    #  Public
    # ------------------------------------------------------------------ #
    def research_missing(
        self,
        slots: Sequence[str],
        *,
        context_keywords: Sequence[str] | None = None,
        k: int = 6,
    ) -> Dict[str, Dict]:
        """
        slots → suggestions dict
        {
          slot: {
            "suggestion" : str,
            "sources"    : [ SearchResult, … ]
          }
        }
        """
        context_keywords = list(context_keywords or [])
        suggestions: Dict[str, Dict] = {}

        for slot in slots:
            query = self._craft_query(slot, context_keywords)
            raw_hits = self._search(query, k)
            results = [self._parse_hit(h) for h in raw_hits]
            results = [r for r in results if r]

            # rank
            ranked = sorted(
                results,
                key=lambda r: (
                    -self._domain_score(r.url),
                    -self._recency_score(r.published),
                ),
            )

            best = ranked[0] if ranked else None
            if best:
                suggestions[slot] = {
                    "suggestion": self._derive_suggestion(slot, best),
                    "sources": ranked[:3],  # top-3 trace
                }

        return suggestions

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    def _craft_query(self, slot: str, ctx: Sequence[str]) -> str:
        """Very naïve query generator."""
        base_kw = {
            "goal": "project objective definition",
            "audience": "target audience definition",
            "tone": "writing tone examples",
            "deadline": "typical turnaround time",
            "constraints": "common constraints requirements",
        }.get(slot, slot)
        return " ".join(ctx + [base_kw])

    # ..................................................................

    def _parse_hit(self, hit: Dict) -> SearchResult | None:
        title = hit.get("title") or hit.get("href") or ""
        url = hit.get("href") or ""
        snippet = hit.get("body") or hit.get("snippet") or ""
        date = self._extract_date(snippet) or self._extract_date(title)

        try:
            published = _dt.date(*date) if date else None
        except Exception:
            published = None

        return SearchResult(title=title, url=url, snippet=snippet, published=published, raw=hit)

    def _extract_date(self, text: str) -> tuple[int, int, int] | None:
        m = _DATE_RX.search(text)
        if m:
            y, mo, d = map(int, m.groups())
            return y, mo, d
        return None

    # ..................................................................
    @staticmethod
    def _domain_score(url: str) -> int:
        return 1 if any(dom in url for dom in _TRUSTED) else 0

    def _recency_score(self, pub: _dt.date | None) -> float:
        if not pub:
            return 0.0
        age = (_dt.date.today() - pub).days
        return 1.0 if age <= self._max_age else 0.0

    # ..................................................................
    def _derive_suggestion(self, slot: str, hit: SearchResult) -> str:
        """Slot-specific heuristics to turn a snippet → concrete value."""
        low = hit.snippet.lower()

        if slot == "tone":
            for tone in ["formal", "informal", "friendly", "professional", "persuasive"]:
                if tone in low:
                    return tone
        if slot == "deadline":
            if "hour" in low:
                return "24h"
            if "week" in low:
                return "1 week"
        # fallback: trimmed title
        return hit.title[:60] + "…"
