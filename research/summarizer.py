
class Summarizer:
    def summarize(self, hits):
        joined = "\n".join(h["content"] for h in hits)
        return joined[:500] + ("..." if len(joined) > 500 else "")
