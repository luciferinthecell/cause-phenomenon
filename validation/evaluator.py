from ..tools.llm_verifier import LLMVerifier
MIN_PASS_SCORE=0.78

class Evaluator:
    def __init__(self):
        self.verifier=LLMVerifier()
    def evaluate(self,question:str,draft:str):
        prompt=(f"[Verifier]\nQuestion:\n{question}\n\nDraft Answer:\n{draft}\n\n"
                "Rate 0-1 on factual accuracy, reasoning clarity, completeness. List issues.")
        res=self.verifier.check(prompt)
        return {"score":res["score"],"issues":res["issues"],
                "verdict":"pass" if res["score"]>=MIN_PASS_SCORE else "fail"}
