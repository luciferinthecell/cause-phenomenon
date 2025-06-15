"""
harin.security.policy_guard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Guardrails ON ↔ OFF 스위치가 가능한 의미·맥락 기반 보안 필터.

* **GuardMode.OFF**   : 무조건 통과(연구·내부망)
* **GuardMode.AUDIT** : 모두 통과 + 위험 점수 로그(검수)
* **GuardMode.FULL**  : 점수에 따라 ALLOW / AUDIT / BLOCK (운영)

환경변수
────────
* `HARIN_POLICY_MODE`    : off / audit / full  (default full)
* `HARIN_POLICY_BLOCK`   : 차단 임계값 (default 0.80)
* `HARIN_POLICY_AUDIT`   : 감사 임계값 (default 0.60)

```python
from harin.security.policy_guard import PolicyGuard
verdict, reason = PolicyGuard.from_env().check(text)
```
"""

from __future__ import annotations

import json
import os
from enum import Enum
from typing import Protocol, Tuple

from harin.integrations.llm_client import LLMClient


class Verdict(str, Enum):
    ALLOW = "allow"
    AUDIT = "audit"
    BLOCK = "block"


class GuardMode(str, Enum):
    OFF = "off"     # always allow
    AUDIT = "audit" # allow + log
    FULL = "full"   # full guardrails


# ──────────────────────────────────────────────────────────────────────────
#  Strategy protocol – external policy model (embedding / LLM)
# ──────────────────────────────────────────────────────────────────────────


class PolicyModel(Protocol):
    def risk_score(self, text: str) -> float: ...  # 0 safe → 1 violation


# ------------------------------------------------------------------------
#  LLM‑based policy model (simple JSON score parser)
# ------------------------------------------------------------------------


def _policy_prompt(text: str) -> str:
    return (
        "You are a compliance assistant. Rate the following user text for policy risk "
        "on a scale 0 (safe) to 1 (severe violation). Return JSON: {\"score\": float, "
        "\"reason\": string}.\n\nUSER_TEXT:\n" + text
    )


class LLMPolicyModel:
    def __init__(self, client: LLMClient | None = None):
        self.client = client or LLMClient.from_env()

    def risk_score(self, text: str) -> float:  # noqa: D401
        resp = self.client.complete(_policy_prompt(text), max_tokens=100, temperature=0.0)
        try:
            data = json.loads(resp)
            return float(data.get("score", 0.0))
        except Exception:
            return 0.5  # unknown → audit


# ------------------------------------------------------------------------
#  Guard wrapper
# ------------------------------------------------------------------------


class PolicyGuard:
    def __init__(
        self,
        *,
        model: PolicyModel | None = None,
        mode: GuardMode | None = None,
    ) -> None:
        self.model = model or LLMPolicyModel()
        self.mode = mode or GuardMode(os.getenv("HARIN_POLICY_MODE", "full"))
        self.block_threshold = float(os.getenv("HARIN_POLICY_BLOCK", "0.8"))
        self.audit_threshold = float(os.getenv("HARIN_POLICY_AUDIT", "0.6"))

    # ---------------------------------------------------------------
    def check(self, text: str) -> Tuple[Verdict, str]:
        # OFF 모드 – 항상 허용
        if self.mode == GuardMode.OFF:
            return Verdict.ALLOW, "guard off"

        score = self.model.risk_score(text)

        # AUDIT 모드 – 무조건 통과, 점수 기록
        if self.mode == GuardMode.AUDIT:
            return Verdict.AUDIT, f"audit‑only score={score:.2f}"

        # FULL 모드 – 기존 임계값 판단
        if score >= self.block_threshold:
            return Verdict.BLOCK, f"score={score:.2f} ≥ block"
        if score >= self.audit_threshold:
            return Verdict.AUDIT, f"score={score:.2f} ≥ audit"
        return Verdict.ALLOW, f"score={score:.2f}"

    # factory helper
    @staticmethod
    def from_env() -> "PolicyGuard":
        return PolicyGuard()
"""
