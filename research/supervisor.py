
class ResearchSupervisor:
    """Assign queries to multiple researchers and aggregate results"""
    def __init__(self, researcher_cls, num_workers: int = 1):
        self.researchers = [researcher_cls() for _ in range(num_workers)]

    def gather(self, queries):
        all_hits = []
        for r, q in zip(self.researchers, queries):
            all_hits.extend(r.run(q))
        return all_hits
