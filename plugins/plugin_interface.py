
from abc import ABC, abstractmethod
from typing import Any, Dict

class HarinPlugin(ABC):
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def execute(self, data: Dict[str, Any]) -> Any: ...
