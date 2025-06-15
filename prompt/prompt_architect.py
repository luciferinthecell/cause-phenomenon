
class PromptArchitect:
    def build(self, user_query, research_summary, persona):
        return f"[SYSTEM] You are {persona}.\n[USER] {user_query}\n[RESEARCH] {research_summary}"
