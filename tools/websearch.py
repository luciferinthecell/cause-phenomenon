"""
WebSearchClient
═══════════════
Minimal, *pluggable* search wrapper.

Why a wrapper?
--------------
• ❌  *NO hard dependency* on a specific engine (SerpAPI, Brave, DuckDuckGo…).
• ✅  Agent code can swap backend by passing a callable into `SearchGateway`.
• 🕒  Adds published-date parsing & normalises hits for downstream modules
      (AutoResearcher, SearchPlanner, PromptArchitect …).

Public API
──────────
• client = WebSearchClient(backend=callable)
• hits   = client.search("query", k=8)
        → List[dict]  # uniform keys: title, href, body, published (date)
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import Callable, Dict, List, Sequence

# ---------------------------------------------------------------------------

_NORMAL_KEYS = ("title", "href", "body")  # mandatory
_DATE_RX = re.compile(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})")


class WebSearchError(RuntimeError):
    """Raised when backend fails."""


class WebSearchClient:
    """
    Parameters
    ----------
    backend :
        Callable[[str, int], Sequence[Dict]]
        Must return iterable of dicts with *at least* keys in `_NORMAL_KEYS`.
        (If you're wiring to `web.run`, see helper below.)
    """

    def __init__(self, backend: Callable[[str, int], Sequence[Dict]]) -> None:
        self._backend = backend

    # ------------------------------------------------------------------ #
    def search(self, query: str, k: int = 6) -> List[Dict]:
        try:
            raw_hits = list(self._backend(query, k))
        except Exception as exc:  # noqa: BLE001
            raise WebSearchError(f"Backend failure: {exc}") from exc

        normalised: List[Dict] = []
        for h in raw_hits:
            if not all(key in h for key in _NORMAL_KEYS):
                continue

            hit = {
                "title": h["title"].strip(),
                "href": h["href"],
                "body": h["body"][:260].strip(),
            }
            # try date extraction
            date = self._extract_date(h) or self._extract_date(h["body"]) or self._extract_date(h["title"])
            if date:
                hit["published"] = "-".join(f"{n:02d}" for n in date)  # YYYY-MM-DD

            # keep vendor-specific blob for traceability
            hit["raw"] = h
            normalised.append(hit)

        return normalised[:k]

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    def _extract_date(self, blob) -> tuple[int, int, int] | None:
        if isinstance(blob, dict):
            # vendor often exposes 'date' or 'published'
            for key in ("date", "published", "pub_date"):
                if key in blob and blob[key]:
                    return self._try_parse(blob[key])
        elif isinstance(blob, str):
            m = _DATE_RX.search(blob)
            if m:
                return tuple(map(int, m.groups()))
        return None

    def _try_parse(self, text: str) -> tuple[int, int, int] | None:
        try:
            d = _dt.date.fromisoformat(text[:10])
            return d.year, d.month, d.day
        except Exception:  # noqa: BLE001
            return None


# ---------------------------------------------------------------------------
# Helper: wrap `web.run(search_query=[…])` inside a backend callable
# ---------------------------------------------------------------------------
def web_run_backend(query: str, k: int) -> Sequence[Dict]:
    """
    Example backend that calls the `web.run` tool (from ChatGPT runtime).
    Replace with a real implementation when outside this environment.
    """

    from typing import Any  # lazy import to avoid tool at module-load time

    # The following is pseudo-code; in a real ChatGPT-tool context you'd:
    #   result = web.run(search_query=[{"q": query, "response_length": "short"}])
    # Here we just raise to remind integrators.
    raise NotImplementedError("Connect to web.run or external API here")


# ---------------------------------------------------------------------------
# Quick demo (when a real backend is plugged)
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import pprint, sys

    # Replace with real backend (e.g., SerpAPI wrapper)
    client = WebSearchClient(backend=lambda q, k: [])
    try:
        hits = client.search("OpenAI GPT-4o speed", k=5)
        pprint.pprint(hits)
    except WebSearchError as e:
        print("Search failed:", e, file=sys.stderr)
