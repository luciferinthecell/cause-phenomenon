from ..tools.web_search import WebSearch

class WebResearchAgent:
    def __init__(self, browser):
        self.searcher = WebSearch(browser)

    def run(self, objective: str, refine: bool = True):
        raw = self.searcher.search(objective, recency_days=365)
        hits = []
        for src in raw:
            if isinstance(src, dict) and "title" in src and "snippet" in src:
                hits.append({"title": src["title"], "snippet": src["snippet"]})
        if refine and len(hits) > 5:
            hits = hits[:5]
        return hits
