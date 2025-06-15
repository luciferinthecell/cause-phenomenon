"""
harin.plugins.manager
──────────────────────
Harin 통합형 Plugin Manager – 의미 기반 자동 로딩·분류·실행 구조

특징
─────
1. 플러그인은 run(meta:dict) 함수, tags 리스트, manifest 딕셔너리 포함 필요
2. 실시간 리로드 가능 (pm.reload())
3. tags/capabilities 기반 분류 및 질의
4. 오류·예외·존재하지 않는 플러그인에 대해 감정 기반 로깅 연결 가능 (optional)

Public API
────────────
pm = HarinPluginManager(Path("harin_plugins"))
pm.execute("hello", {"user":"yohan"})
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, Any


class HarinPluginManager:
    def __init__(self, directory: Path):
        self.dir = directory
        self.dir.mkdir(exist_ok=True)
        self._mods: Dict[str, ModuleType] = {}
        self.load_all()

    def load_all(self):
        self._mods.clear()
        for py in self.dir.glob("*.py"):
            mod = self._load_module(py)
            if mod and hasattr(mod, "run"):
                self._mods[py.stem] = mod

    def reload(self):
        self.load_all()

    def execute(self, name: str, meta: Dict[str, Any]) -> str:
        mod = self._mods.get(name)
        if not mod:
            return f"(plugin {name} not found)"
        try:
            return str(mod.run(meta))  # type: ignore
        except Exception as e:
            return f"(plugin {name} failed: {e})"

    def info(self) -> Dict[str, Any]:
        return {
            name: {
                "tags": getattr(mod, "tags", []),
                "manifest": getattr(mod, "manifest", {}),
            }
            for name, mod in self._mods.items()
        }

    def search(self, tag: str) -> list[str]:
        return [
            name for name, mod in self._mods.items()
            if tag in getattr(mod, "tags", [])
        ]

    @staticmethod
    def _load_module(path: Path) -> ModuleType | None:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if not spec or not spec.loader:
            return None
        mod = importlib.util.module_from_spec(spec)
        try:
            sys.modules[path.stem] = mod
            spec.loader.exec_module(mod)  # type: ignore
            return mod
        except Exception:
            return None
