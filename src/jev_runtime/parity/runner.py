"""Reference/optimized parity comparison and gating."""
from __future__ import annotations
import math
from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence
from typing import Any

@dataclass(frozen=True)
class ParityReport:
    samples: int
    choice_agreement: float
    argmax_agreement: float
    max_probability_drift: float
    mean_probability_drift: float
    passed: bool
    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

class ParityRunner:
    def __init__(self, *, max_probability_drift: float = 1e-5, min_choice_agreement: float = 1.0, min_argmax_agreement: float = 1.0):
        if not (math.isfinite(max_probability_drift) and max_probability_drift >= 0 and 0 <= min_choice_agreement <= 1 and 0 <= min_argmax_agreement <= 1):
            raise ValueError("invalid parity thresholds")
        self.max_probability_drift = max_probability_drift
        self.min_choice_agreement = min_choice_agreement
        self.min_argmax_agreement = min_argmax_agreement

    def run(self, cases: Sequence[Any], reference: Callable[[Any], Any], optimized: Callable[[Any], Any]) -> ParityReport:
        """Execute both implementations once per case, then apply the gate."""
        if not cases:
            raise ValueError("parity cases must be non-empty")
        return self.compare(tuple(reference(case) for case in cases), tuple(optimized(case) for case in cases))

    def compare(self, reference: Sequence[Any], optimized: Sequence[Any]) -> ParityReport:
        if len(reference) != len(optimized) or not reference:
            raise ValueError("reference and optimized outputs must have equal non-zero length")
        choices = argmaxes = 0
        drifts: list[float] = []
        for left, right in zip(reference, optimized):
            lp, rp = self._probabilities(left), self._probabilities(right)
            if len(lp) != len(rp) or not lp:
                raise ValueError("probability vectors must have equal non-zero length")
            drift = max(abs(a - b) for a, b in zip(lp, rp))
            drifts.append(drift)
            la, ra = max(range(len(lp)), key=lp.__getitem__), max(range(len(rp)), key=rp.__getitem__)
            argmaxes += la == ra
            choices += self._choice(left, lp) == self._choice(right, rp)
        ca, aa = choices / len(reference), argmaxes / len(reference)
        maximum, mean = max(drifts), sum(drifts) / len(drifts)
        return ParityReport(len(reference), ca, aa, maximum, mean, ca >= self.min_choice_agreement and aa >= self.min_argmax_agreement and maximum <= self.max_probability_drift)

    @staticmethod
    def _probabilities(value: Any) -> tuple[float, ...]:
        values = value.get("probabilities") if isinstance(value, Mapping) else value
        values = tuple(values.values()) if isinstance(values, Mapping) else tuple(values)
        if not values or any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) for x in values):
            raise ValueError("probabilities must be finite and non-empty")
        return tuple(float(x) for x in values)

    @staticmethod
    def _choice(value: Any, probabilities: Sequence[float]) -> Any:
        if isinstance(value, Mapping) and ("choice" in value or "argmax" in value):
            return value.get("choice", value.get("argmax"))
        return max(range(len(probabilities)), key=probabilities.__getitem__)

def run_parity(reference: Sequence[Any], optimized: Sequence[Any], **kwargs: Any) -> ParityReport:
    return ParityRunner(**kwargs).compare(reference, optimized)

__all__ = ["ParityReport", "ParityRunner", "run_parity"]
