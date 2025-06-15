
from research_tools.web_search import WebSearch
from research_tools.source_evaluator import SourceEvaluator

class Researcher:
    def __init__(self, max_results: int = 10):
        self.search = WebSearch()
        self.eval = SourceEvaluator()
        self.max_results = max_results

    def run(self, query: str):
        hits = self.search.search(query, self.max_results)
        rated = [self.eval.evaluate(h, [query]) | {"content": h} for h in hits]
        rated.sort(key=lambda x: -x["score"])
        return rated
