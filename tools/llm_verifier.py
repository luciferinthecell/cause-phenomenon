import openai
class LLMVerifier:
    def __init__(self,model="gpt-3.5-turbo-0125"):
        self.model=model
    def check(self,prompt:str):
        resp=openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role":"user","content":prompt}],
            temperature=0,max_tokens=256)
        txt=resp.choices[0].message.content.strip()
        lines=txt.splitlines()
        try:
            score_line=next(l for l in lines if "score" in l.lower())
            score=float(score_line.split(":")[1].strip())
        except Exception:
            score=0.5
        issues=[l for l in lines if l.startswith("-")]
        return {"score":score,"issues":issues,"raw":txt}
