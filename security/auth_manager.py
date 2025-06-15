"""
harin.security.auth_manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dynamic Token Auth + Basic ACL + Rate‑Limit.

• `generate_token(user_id, role, ttl)`  → uuid4 토큰 발급 (TTL 초)  
• `revoke_token(token)`                 → 즉시 폐기  
• `validate_token(token)`               → (ok, user_id, role, reason) 반환  
• `verify_user(user_id, token, action)` → 토큰 또는 정적 env‑token 확인 + ACL + 레이트리밋

환경변수
────────
* `HARIN_AUTH_TOKENS`   : "t1:admin,t2:user" (static)  
* `HARIN_AUTH_MODE`     : `static` / `dynamic` (default static)  
* `HARIN_RATE_WINDOW`   : 초, default 60  
* `HARIN_RATE_MAX`      : 메시지, default 30
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

from harin.security.access_control import AccessControl, DefaultACL


@dataclass
class _TokenInfo:
    user_id: str
    role: str
    exp: float  # epoch seconds


@dataclass
class _UserWindow:
    last_ts: float = field(default_factory=time.time)
    count: int = 0


class AuthManager:
    """Token store + rate‑limit + ACL checker."""

    def __init__(
        self,
        *,
        rate_window: int = 60,
        max_msgs: int = 30,
        acl: AccessControl | None = None,
    ) -> None:
        self.rate_window = rate_window
        self.max_msgs = max_msgs
        self.mode = os.getenv("HARIN_AUTH_MODE", "static").lower()

        # static tokens from env
        self.static_tokens: Dict[str, str] = {}
        for pair in os.getenv("HARIN_AUTH_TOKENS", "").split(","):
            if ":" in pair:
                tok, role = pair.split(":", 1)
                self.static_tokens[tok.strip()] = role.strip()

        # dynamic token store
        self.tokens: Dict[str, _TokenInfo] = {}

        # rate‑limit window per user
        self._windows: Dict[str, _UserWindow] = {}

        # ACL
        self.acl = acl or DefaultACL()

    # ────────────────────────────────────────────────────────────
    # Token API (dynamic mode)
    # ────────────────────────────────────────────────────────────
    def generate_token(self, user_id: str, *, role: str = "user", ttl: int | None = 3600) -> str:
        token = uuid.uuid4().hex
        exp = time.time() + ttl if ttl else float("inf")
        self.tokens[token] = _TokenInfo(user_id=user_id, role=role, exp=exp)
        return token

    def revoke_token(self, token: str) -> None:
        self.tokens.pop(token, None)

    def validate_token(self, token: str) -> Tuple[bool, str, str]:
        # returns ok, user_id, role
        if token in self.static_tokens:
            return True, "static", self.static_tokens[token]
        info = self.tokens.get(token)
        if not info:
            return False, "", "anon"
        if info.exp < time.time():
            self.tokens.pop(token, None)
            return False, info.user_id, info.role
        return True, info.user_id, info.role

    # ────────────────────────────────────────────────────────────
    # verify_user – token + ACL + rate‑limit
    # ────────────────────────────────────────────────────────────
    def verify_user(
        self,
        *,
        user_id: str,
        token: Optional[str] = None,
        action: str | None = None,
    ) -> Tuple[bool, str, str]:  # ok, role, reason
        # 1) token / role
        role = "anon"
        ok_token = False
        if self.mode == "dynamic" and token:
            ok_token, user_from_tok, role = self.validate_token(token)
            if ok_token and user_from_tok != "static":
                user_id = user_from_tok
        else:
            role = self.static_tokens.get(token or "", "anon")
            ok_token = True if role != "anon" else False

        # 2) ACL check
        if action and not self.acl.check(role, action):
            return False, role, "no permission"

        # 3) rate‑limit
        win = self._windows.setdefault(user_id, _UserWindow())
        now = time.time()
        if now - win.last_ts > self.rate_window:
            win.count = 0
            win.last_ts = now
        win.count += 1
        if win.count > self.max_msgs:
            return False, role, "rate limit"

        return True, role, "ok"

    # factory
    @staticmethod
    def from_env() -> "AuthManager":
        w = int(os.getenv("HARIN_RATE_WINDOW", "60"))
        m = int(os.getenv("HARIN_RATE_MAX", "30"))
        return AuthManager(rate_window=w, max_msgs=m)
"""
