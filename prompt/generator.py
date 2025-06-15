
class PromptSynth:
    def assemble(self, plan, metacog):
        return f"{plan}\n\n# Self‑reflection\n{metacog}"
