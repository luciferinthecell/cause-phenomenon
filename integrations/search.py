"""
harin.integrations.search
~~~~~~~~~~~~~~~~~~~~~~~~~

*One-stop helper* that a higher-level loop(ThoughtProcessor 등)에서 호출해
“웹에서 최신 정보를 긁어와 → 신뢰·신선도·사용자-적합성 평가 →
필요하면 간략 요약까지 반환” 까지 자동으로 처리한다.

* 외부 검색 엔진 호출은 **함수 인젝션**(dependency-injection) 방식 — 
  실제 실행 환경(Google-Custom-Search · SerpAPI · web.run 등)을
  최상단 애플리케이션에서 주입하면 테스트가 쉽고 API 키가 코드에
  하드코딩되지 않는다.
"""

from __future__ import annotations

import datetime as _dt
from typing import Callable, List, Dict, Any

from ..evaluators.verification import (
    UserProfile,
    InfoVerificationEngine,
)
from ..memory.palantir import PalantirGraph

# Type alias:  (query, k) -> List[Dict[str, Any]]
# Each dict  {title, content/snippet, url, date}
SearchFn = Callable[[str, int], List[Dict[str, str]]]


class SearchService:
    """
    Parameters
    ----------
    search_fn
        Function that actually performs web search.  
        Signature: `search_fn(query: str, top_k: int) -> List[dict]`.
    """

    def __init__(self, search_fn: SearchFn, *, top_k: int = 15) -> None:
        self._search = search_fn
        self.top_k = top_k

    # ------------------------------------------------------------------ #
    #  public high-level API
    # ------------------------------------------------------------------ #

    def run(
        self,
        query: str,
        user: UserProfile,
        memory: PalantirGraph | None = None,
        *,
        limit: int = 5,
        min_score: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """
        Full pipeline:
        1) call search_fn ➜ raw hits
        2) InfoVerificationEngine 평가
        3) score ≥ min_score만 keep & 앞에서 *limit*개 리턴
        """
        raw_hits = self._search(query, self.top_k)
        if not raw_hits:
            return []

        ranked = InfoVerificationEngine.evaluate_batch(
            raw_hits, user=user, memory=memory, drop_below=min_score
        )
        return ranked[:limit]

    # ------------------------------------------------------------------ #
    #  convenience wrapper – query + format→markdown
    # ------------------------------------------------------------------ #

    def quick_md(
        self,
        query: str,
        user: UserProfile,
        memory: PalantirGraph | None = None,
        *,
        limit: int = 5,
    ) -> str:
        """
        Human-readable Markdown (for ThoughtProcessor log / debug UI).
        """
        rows = self.run(query, user, memory, limit=limit)
        if not rows:
            return f"> **검색 결과 없음** – _{query}_"

        lines = [f"### 🔍 Top {len(rows)} results for **{query}**\n"]
        for i, r in enumerate(rows, 1):
            lines.append(
                f"{i}. [{r['title']}]({r['url']}) "
                f"(score {r['score']:.2f}, trust {r['trust']}, "
                f"recency {r['recency']:.2f})"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  ★ 기본 “search_fn” 예시 – web.run wrapper ★
# --------------------------------------------------------------------------- #
# 실제 프로덕션에서는 별도 모듈에서 정의한 뒤 SearchService(search_fn)으로 주입.
# 아래 코드는 “어떻게 연결하는지”를 보여주는 *참고용* 예시이다.
# --------------------------------------------------------------------------- #

def web_run_search_adapter(query: str, k: int) -> List[Dict[str, str]]:
    """
    Dummy adapter; **실제 웹 검색 호출 시 교체**.

    구조만 맞춰서 반환하면 SearchService는 그대로 사용할 수 있다.
    """
    # ── replace below with real `web.run({"search_query": …})` logic ──
    today_str = _dt.date.today().isoformat()
    return [
        {
            "title": f"Mock article {i} about {query}",
            "content": f"{query} 관련 모의 내용 {i}",
            "date": today_str,
            "url": f"https://example.com/{query}/{i}",
        }
        for i in range(1, k + 1)
    ]


# --------------------------------------------------------------------------- #
#  test-drive CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    user = UserProfile(goal_terms=["gpt-4o", "multimodal"])
    svc = SearchService(web_run_search_adapter)
    print(svc.quick_md("GPT-4o", user))
