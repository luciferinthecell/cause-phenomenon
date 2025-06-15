"""WebSearch tool – naive wrapper over web.run for Harin."""
class WebSearch:
    def __init__(self, browser):
        self.browser = browser

    def search(self, query: str, recency_days: int | None = None, domains=None):
        params = {"search_query": [{"q": query, "recency": recency_days, "domains": domains}]}
        return self.browser.run(params)
