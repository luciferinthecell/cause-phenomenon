from ..prompting.architect import PromptArchitect
from .evaluator import Evaluator

class SelfCorrector:
    def __init__(self,llm_func):
        self.eval=Evaluator()
        self.llm=llm_func
        self.arch=PromptArchitect()
    def process(self,question:str,draft:str):
        review=self.eval.evaluate(question,draft)
        if review["verdict"]=="pass":
            return draft,review
        fix_prompt=self._repair_prompt(question,draft,review["issues"])
        revised=self.llm(fix_prompt,temperature=0.4)
        review2=self.eval.evaluate(question,revised)
        return revised,review2
    def _repair_prompt(self,q,a,issues):
        bullet="\n- ".join(issues) if issues else "None"
        return (self.arch.identity_block()+f"\nRewrite answer fixing issues:\n- {bullet}\n"
                f"Original Question:\n{q}\n\nOriginal Draft:\n{a}\n")
