
class TaskSynth:
    """Generate refined queries from an initial vague request"""
    def synthesize(self, request: str):
        parts = [p.strip() for p in request.split(',')]
        return [f"검색: {p}" for p in parts if p]
