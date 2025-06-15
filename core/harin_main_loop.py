# === harin/core/harin_main_loop.py ===
# Master orchestrator loop for HarinAgent v6.1

from harin.core.state import HarinState
from harin.memory.palantir import PalantirGraph
from harin.memory.persistence import MemoryPersistenceManager
from harin.reasoning.recursive_reasoner import HarinRecursiveReasoner
from harin.evaluators.judges import JudgeEngine, UserProfile
from harin.evaluators.verification import InfoVerificationEngine
from harin.evaluators.verification import UserProfile
from harin.reflection.self_verifier import SelfVerifier
from harin.reflection.output_corrector import OutputCorrector
from harin.interface.session_archiver import SessionArchiver

# optional
# from harin.integrations.search import SearchService, web_run_search_adapter

def run_harin_session(user_input: str) -> dict:
    # ─────────────── 1. 상태 & 메모리 초기화 ───────────────
    state = HarinState()
    memory = PalantirGraph.load()
    persistence = MemoryPersistenceManager(memory)
    session = SessionArchiver()

    # ─────────────── 2. 사용자 모델 구성 ───────────────
    user = UserProfile(
        goal_terms=["multimodal", "reasoning", "trust"],  # customize this
        knowledge_terms=["GPT", "agent", "LLM"],
        cognitive_level="normal"
    )

    # ─────────────── 3. 사고 루프 실행 ───────────────
    verifier = SelfVerifier(llm_client=None)  # inject actual LLM
    reasoner = HarinRecursiveReasoner(memory, identity_manager=None, verifier=verifier)
    steps = reasoner.run(user_input)
    core_output = steps[-1].data["response"]

    # ─────────────── 4. 판단 평가 ───────────────
    judge = JudgeEngine()
    scored = judge.evaluate_batch([{"content": core_output}], user=user, memory=memory)
    best = scored[0]

    # ─────────────── 5. 자기검증 + 보정 ───────────────
    sv_result = verifier.verify(core_output, context_text=user_input)
    corrector = OutputCorrector(llm_client=None)  # inject actual LLM
    final_output = corrector.correct(core_output, sv_result, context_text=user_input)

    # ─────────────── 6. 세션 저장 ───────────────
    session.record(user_input, final_output, [s.__dict__ for s in steps])
    persistence.save()

    return {
        "output": final_output,
        "score": best["composite"],
        "verifier": sv_result,
        "trace": [s.summary for s in steps],
    }