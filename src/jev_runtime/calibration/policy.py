"""Pure-Python probability calibration policy."""
from __future__ import annotations
import math
from dataclasses import dataclass
from collections.abc import Sequence
from ..heads.readouts import normalized_entropy, stable_softmax

@dataclass(frozen=True)
class CalibrationPolicy:
    temperature: float = 1.0
    version: str = "temperature-v1"
    metadata: dict[str, str] | None = None
    def __post_init__(self) -> None:
        if not isinstance(self.temperature, (int, float)) or not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("version must be non-blank")
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def apply(self, logits: Sequence[float]) -> tuple[float, ...]:
        values = tuple(float(x) for x in logits)
        if not values or any(not math.isfinite(x) for x in values):
            raise ValueError("logits must be finite and non-empty")
        return stable_softmax(tuple(x / self.temperature for x in values))

    def describe(self) -> dict[str, object]:
        return {"version": self.version, "temperature": self.temperature, "metadata": dict(self.metadata or {})}

    def confidence(self, probabilities: Sequence[float]) -> float:
        return normalized_entropy(probabilities)

__all__ = ["CalibrationPolicy"]
