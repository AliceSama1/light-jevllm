"""Plugin protocols and compatibility registry."""
from __future__ import annotations
from typing import Any, Protocol, runtime_checkable
from ..decision_ir import ReadoutPlan, ReadoutResult

@runtime_checkable
class Family(Protocol):
    def compile(self, request: dict[str, Any]) -> Any: ...
    def readout(self, items: Any, output: Any) -> Any: ...
    def close(self) -> None: ...

@runtime_checkable
class Backend(Protocol):
    async def execute(self, items: Any, **kwargs: Any) -> Any: ...
    def close(self) -> None: ...

@runtime_checkable
class Readout(Protocol):
    """Pure adapter from backend output to typed readout results."""
    def execute(self, output: Any, plan: ReadoutPlan) -> list[ReadoutResult]: ...

class Registry:
    def __init__(self): self._factories: dict[str, Any] = {}
    @property
    def factories(self): return self._factories
    def register(self, name: str, factory: Any) -> None:
        if name in self._factories: raise ValueError("duplicate plugin: " + name)
        if not callable(factory): raise ValueError("plugin factory must be callable")
        self._factories[name] = factory
    def create(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if name not in self._factories: raise ValueError("unknown plugin: " + name)
        return self._factories[name](*args, **kwargs)
    def names(self): return tuple(self._factories)

FAMILIES = Registry()
BACKENDS = Registry()
READOUTS = Registry()
