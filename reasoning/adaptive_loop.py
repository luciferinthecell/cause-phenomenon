from datetime import datetime
from .user_state import UserModel
from .drift_monitor import DriftMonitor

class AdaptiveReasoningLoop:
    def __init__(self):
        self.user=UserModel()
        self.drift=DriftMonitor()
        self.last_strategy="baseline"
    def register_turn(self, question:str, answer:str, rhythm_score:float):
        self.user.update_from_input(question)
        self.drift.push(rhythm_score)
    def decide_strategy(self):
        u=self.user.snapshot()
        d=self.drift.series[-1] if self.drift.series else None
        if d is None:
            return {"strategy":"baseline","user":u,"drift":"n/a"}
        strategy=self.last_strategy
        drift_flag=self.drift.push(0).get("drift",False)
        if drift_flag:
            strategy="slow_cot"
        elif u["preferred_depth"]=="deep":
            strategy="full_chain"
        elif u["knowledge_level"]=="novice":
            strategy="teach_mode"
        else:
            strategy="baseline"
        self.last_strategy=strategy
        return {"strategy":strategy,"user":u,"drift":self.drift.push(0)}
