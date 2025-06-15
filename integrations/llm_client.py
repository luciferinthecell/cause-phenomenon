"""
harin.integrations.llm_client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

의미 기반 Harin 에이전트용 통합 LLM 클라이언트.

원칙
------
* **모델 이름이 아닌 의미** (task → params) 를 입력받아 프롬프트/옵션을 자동 매핑.
* **백엔드 선택**은 환경변수·설정파일로 주입, 동적 전환 가능.
* **키워드 트리거 사용 금지** – 정책·시스템 지침은 SYSTEM 프롬프트 블록으로 표현.

Public API
-----------
```python
client = LLMClient.from_env()
reply = client.complete(prompt, max_tokens=512, temperature=0.7)
```
"""

from __future__ import annotations

import os
from typing import Protocol, Dict, Any

# optional imports guarded
try:
    import openai  # type: ignore
except ImportError:  # pragma: no cover
    openai = None

try:
    from google.generativeai import GenerativeModel  # type: ignore
except ImportError:  # pragma: no cover
    GenerativeModel = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────
#  Protocol (meaning-first)
# ──────────────────────────────────────────────────────────────────────────


class LLMBackend(Protocol):
    name: str

    def complete(self, prompt: str, **kw) -> str: ...


# ------------------------------------------------------------------------
#  Concrete backends
# ------------------------------------------------------------------------


class OpenAIBackend:
    name = "openai"

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        if openai is None:
            raise RuntimeError("openai package not installed")
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY missing")
        openai.api_key = api_key
        self.model = model

    def complete(self, prompt: str, **kw) -> str:  # noqa: D401
        resp = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kw,
        )
        return resp.choices[0].message.content.strip()


class GeminiBackend:
    name = "gemini"

    def __init__(self, model: str = "gemini-pro", api_key: str | None = None):
        if GenerativeModel is None:
            raise RuntimeError("google-generativeai package not installed")
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY missing")
        os.environ["GOOGLE_API_KEY"] = api_key
        self.model = GenerativeModel(model)

    def complete(self, prompt: str, **kw) -> str:
        resp = self.model.generate_content(prompt, generation_config=kw or {})
        return resp.text.strip()


class EchoBackend:
    name = "echo"

    def complete(self, prompt: str, **kw) -> str:  # noqa: D401
        return "(echo) " + prompt.splitlines()[-1]


# ------------------------------------------------------------------------
#  High-level client with graceful fallback
# ------------------------------------------------------------------------


class LLMClient:
    def __init__(self, backend: LLMBackend):
        self.backend = backend

    # main call
    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.7) -> str:
        try:
            return self.backend.complete(prompt, max_tokens=max_tokens, temperature=temperature)
        except Exception as e:  # pragma: no cover  – fallback
            return f"(LLM error: {e.__class__.__name__}) {EchoBackend().complete(prompt)}"

    # ------------------------------------------------------------------
    # factory helpers
    # ------------------------------------------------------------------
    @staticmethod
    def from_env() -> "LLMClient":
        target = os.getenv("HARIN_LLM_BACKEND", "openai").lower()
        if target == "openai":
            try:
                return LLMClient(OpenAIBackend())
            except Exception:  # fallback
                pass
        if target == "gemini":
            try:
                return LLMClient(GeminiBackend())
            except Exception:
                pass
        # default echo
        return LLMClient(EchoBackend())
"""
