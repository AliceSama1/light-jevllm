"""Transport-neutral data only; deadlines use a monotonic clock."""
from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelIdentity:
    name: str
    version: str | None = None
    family: str | None = None
    revision: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("model name must be non-blank")
        for value in (self.version, self.family, self.revision):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError("identity fields must be non-blank when supplied")


@dataclass(frozen=True)
class Deadline:
    expires_at: float

    def __post_init__(self) -> None:
        if type(self.expires_at) not in (int, float) or not math.isfinite(self.expires_at):
            raise ValueError("expires_at must be finite monotonic seconds")

    @classmethod
    def after_ms(cls, timeout_ms: float, *, now: float | None = None) -> Deadline:
        if type(timeout_ms) not in (int, float) or not math.isfinite(timeout_ms) or timeout_ms < 0:
            raise ValueError("timeout_ms must be finite and non-negative")
        return cls((time.monotonic() if now is None else now) + timeout_ms / 1000)

    def remaining(self, *, now: float | None = None) -> float:
        return max(0.0, self.expires_at - (time.monotonic() if now is None else now))

    def expired(self, *, now: float | None = None) -> bool:
        return self.remaining(now=now) == 0


@dataclass(frozen=True)
class CostEstimate:
    compute_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    memory_bytes: int = 0

    def __post_init__(self) -> None:
        if type(self.compute_ms) not in (int, float) or not math.isfinite(self.compute_ms) or self.compute_ms < 0:
            raise ValueError("compute_ms must be finite and non-negative")
        for value in (self.input_tokens, self.output_tokens, self.memory_bytes):
            if type(value) is not int or value < 0:
                raise ValueError("cost counts must be non-negative integers")


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.healthy) is not bool:
            raise ValueError("healthy must be boolean")
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


@dataclass(frozen=True)
class ExecutionOutput:
    """Opaque backend output; interpretation belongs to the readout layer."""
    value: Any
    model: ModelIdentity | None = None
    cost: CostEstimate | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
