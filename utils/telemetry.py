"""
harin.utils.telemetry
~~~~~~~~~~~~~~~~~~~~~

Lightweight TelemetryTracker – 운영 시 이벤트·지표를 파일·stdout·callback 등으로 전송.

철학
------
* **의미 기반 이벤트** – 키워드·카테고리 대신 `event` 문자열과 `payload(dict)`를 자유 전송.
* **경량화** – 연구 모드에서는 `NullTelemetry`(no‑op) 사용.
* **Destinations** – `file`, `stdout`, 또는 `custom_callable` 주입 가능.

Public API
-----------
```python
tracker = TelemetryTracker(dest="file", path="harin_telemetry.log")
tracker.log(user_id="u1", event="reply", length=123)
```
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Any


class TelemetryDest:
    """Enum‑like strings to select output destination."""

    FILE = "file"
    STDOUT = "stdout"
    CALLBACK = "callback"


class TelemetryTracker:
    def __init__(
        self,
        *,
        dest: str = TelemetryDest.STDOUT,
        path: Path | None = None,
        callback: Callable[[Dict[str, Any]], None] | None = None,
    ) -> None:
        self.dest = dest
        self.path = path or Path("harin_telemetry.log")
        self.callback = callback

        if self.dest == TelemetryDest.FILE:
            # create file if missing
            self.path.touch(exist_ok=True)

    # ------------------------------------------------------------------
    def log(self, **payload: Any) -> None:  # noqa: D401
        record = {
            "ts": time.time(),
            **payload,
        }

        if self.dest == TelemetryDest.STDOUT:
            print("[telemetry]", json.dumps(record, ensure_ascii=False))
        elif self.dest == TelemetryDest.FILE:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        elif self.dest == TelemetryDest.CALLBACK and self.callback:
            try:
                self.callback(record)
            except Exception:  # pragma: no cover
                pass
        # else: silently ignore


# no‑op instance for research mode
class NullTelemetry(TelemetryTracker):
    def __init__(self):
        super().__init__(dest=TelemetryDest.CALLBACK, callback=lambda _: None)

    def log(self, **payload: Any) -> None:  # noqa: D401
        return  # ignore
"""
