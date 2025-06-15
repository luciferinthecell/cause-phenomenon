"""
harin.core.agent
~~~~~~~~~~~~~~~~

**HarinAgent v3 – Reasoning + Research 통합**  
Odin Guardrails · Dynamic Auth/ACL · Telemetry · Plugin · Research Pipeline 지원.

모드 흐름
──────────
1.  **System**   : Guard → Auth/ACL → Plugins(before)
2.  **Routing**  : GoalEstimator로 *(chat / research / plugin / control)* 판단
3.  **Execution**:
    • chat      → Reasoning 파이프라인 (Palantir Memory)
    • research  → Researcher → Aggregator → Summarizer → Synthesis Engine
    • control   → telemetry flush / token generate / plugin exec 등
4.  **Output**   : Plugins(after) → Telemetry 기록 → Memory 저장

연구 흐름은 *키워드 if/else 없이* `GoalEstimator.score()`에서 "research" 루프 점수가 가장 높을 때 자동 진입.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional, List, Tuple

# ─── Core Reasoning Stack ────────────────────────────────────────────
from harin.memory.adapter import MemoryEngine
from harin.core.context import UserContext
from harin.core.goal_estimator import GoalEstimator
from harin.core.loops import ThoughtProcessor
from harin.core.metacognition import SelfReflector, ReflectionWriter
from harin.core.prompt_architect import PromptArchitect

from harin.integrations.llm_client import LLMClient
from harin.security.policy_guard import PolicyGuard, Verdict
from harin.security.auth_manager import AuthManager
from harin.utils.telemetry import TelemetryTracker, TelemetryDest, NullTelemetry
from harin.security.access_control import DefaultACL

# ─── Research Stack (legacy unified) ────────────────────────────────
from research.researcher import Researcher  # type: ignore
from research.aggregator import Aggregator  # type: ignore
from research.summarizer import Summarizer  # type: ignore
from research.task_synth import TaskSynth  # type: ignore
from research.synthesis import SynthesisEngine  # type: ignore

# ─── Plugin & Logger (stubs for DI) ─────────────────────────────────
class NullLogger:
    def info(self, *a, **kw): ...
    def warning(self, *a, **kw): ...
    def error(self, *a, **kw): ...


class NullPluginManager:
    def before_process(self, *a, **k): ...
    def after_process(self, *a, **k): ...
    def execute(self, name: str, meta: dict):
        return f"(plugin {name} not found)"


# ────────────────────────────────────────────────────────────────────
class HarinAgent:
    """Reasoning + Research orchestrator with guard/auth/telemetry."""

    def __init__(
        self,
        *,
        memory_path: Path = Path("harin_memory.jsonl"),
        graph_path: Path = Path("palantir_graph.json"),
        logger=None,
        guard: PolicyGuard | None = None,
        auth: AuthManager | None = None,
        plugins=None,
        telemetry: TelemetryTracker | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        # core subsystems
        self.memory = MemoryEngine(bucket_path=memory_path, graph_path=graph_path)
        self.context = UserContext()
        self.goal_estimator = GoalEstimator()
        self.processor = ThoughtProcessor(
            memory=self.memory,
            context=self.context,
            goal_estimator=lambda txt: self.goal_estimator.score(
                txt, [l.name for l in self.processor.registry.all()]
            ),
        )
        self.self_reflector = SelfReflector()
        self.reflect_writer = ReflectionWriter(self.memory)
        self.llm = llm_client or LLMClient.from_env()

        # research stack
        self.researcher = Researcher()
        self.aggregator = Aggregator()
        self.summarizer = Summarizer()
        self.task_synth = TaskSynth()
        self.synth_engine = SynthesisEngine()

        # DI subsystems
        self.logger = logger or NullLogger()
        self.guard = guard or PolicyGuard.from_env()
        self.auth = auth or AuthManager.from_env()
        self.plugins = plugins or NullPluginManager()
        self.telemetry = telemetry or NullTelemetry()

        self._start_time: Optional[datetime.datetime] = None
        self._running = False

    # ─── lifecycle ────────────────────────────────────────────────
    def start(self):
        if self._running:
            return
        self._start_time = datetime.datetime.utcnow()
        self.logger.info("Agent started at %s", self._start_time)
        self._running = True

    def shutdown(self):
        if not self._running:
            return
        self.memory.save()
        self.logger.info("Memory saved. Shutdown.")
        self._running = False

    # ─── helper: decide route (chat / research / control) ──────────
    def _route(self, text: str) -> str:
        scores = self.goal_estimator.score(text, ["chat", "research", "control"])
        return max(scores, key=scores.get)

    # ─── main entry ───────────────────────────────────────────────
    def chat(
        self,
        user_input: str,
        *,
        user_id: str = "anon",
        token: str | None = None,
    ) -> str:
        if not self._running:
            self.start()

        # 1) Guardrails
        verdict, reason = self.guard.check(user_input)
        if verdict == Verdict.BLOCK:
            self.telemetry.log(user_id=user_id, event="block", reason=reason)
            return "(blocked)"

        # 2) Auth & ACL
        mode = self._route(user_input)
        ok_auth, role, reason_auth = self.auth.verify_user(
            user_id=user_id, token=token, action=mode
        )
        if not ok_auth:
            return f"(auth fail: {reason_auth})"
        self.context.user_tags.append(role)

        # plugins before
        self.plugins.before_process(user_input=user_input, user_id=user_id, mode=mode)

        # 3) MODE EXEC
        if mode == "research":
            reply = self._run_research(user_input)
        elif mode == "control":
            reply = self._run_control(user_input, user_id, token)
        else:  # chat
            reply = self._run_chat(user_input)

        # plugins after
        self.plugins.after_process(reply=reply, user_id=user_id, mode=mode)
        self.telemetry.log(user_id=user_id, event=mode, length=len(reply))

        return reply

    # ─── chat pipeline (reasoning) ────────────────────────────────
    def _run_chat(self, text: str) -> str:
        best_judgment, all_judgments = self.processor.process(text)
        reflection_txt = self.self_reflector.reflect(all_judgments)
        self.reflect_writer.write(reflection_txt)
        prompt = PromptArchitect.build(
            system_base="You are Harin, an AI reasoning agent.",
            user_input=text,
            context=self.context,
            memory_snippets=[n.text for n in self.memory.recent(3)],
            reflection=reflection_txt,
            best_judgment=best_judgment,
        )
        reply = self.llm.complete(prompt, max_tokens=512, temperature=0.7)
        self.memory.store(reply, node_type="assistant", meta=self.context.as_meta())
        return reply

    # ─── research pipeline ────────────────────────────────────────
    def _run_research(self, text: str) -> str:
        queries = self.task_synth.synthesize(text)
        hits = []
        for q in queries:
            hits.extend(self.researcher.run(q))
        agg = self.aggregator.aggregate(hits)
        summary = self.summarizer.summarize(agg)
        report = self.synth_engine.synthesize(text, summary)
        return report

    # ─── control commands (telemetry flush / plugin etc.) ─────────
    def _run_control(self, text: str, user_id: str, token: str | None) -> str:
        if text == "telemetry flush":
            path = Path("harin_telemetry.log").resolve()
            return f"Flushed telemetry to {path}"
        if text.startswith("run plugin:"):
            name = text.split(":", 1)[1].strip()
            return self.plugins.execute(name, {"user": user_id})
        if text.startswith("login "):
            new_user = text.split(" ", 1)[1].strip()
            new_tok = self.auth.generate_token(new_user, role="admin")
            return f"TOKEN {new_tok}"
        return "(unknown control)"


# ─── quick demo ───────────────────────────────────────────────────
if __name__ == "__main__":
    agent = HarinAgent(telemetry=TelemetryTracker(dest=TelemetryDest.STDOUT))
    while True:
        try:
            raw = input("You> ")
        except EOFError:
            break
        if raw in {"exit", "quit"}:
            break
        print("Harin>", agent.chat(raw, user_id="cli"))
    agent.shutdown()
"""
