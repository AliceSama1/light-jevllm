"""Explicit readout adapters for Decision execution outputs.

Readouts deliberately accept opaque backend outputs and a ``ReadoutPlan``;
they do not know HTTP request schemas or model-family internals.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any
from ..plugins import Readout

from ..decision_ir import (
    ReadoutPlan,
    ReadoutResult,
)


def _finite(values: Sequence[float], what: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(not math.isfinite(value) for value in result):
        raise ValueError(f"{what} must contain finite values")
    return result


def stable_softmax(logits: Sequence[float]) -> tuple[float, ...]:
    """Compute a finite, normalized softmax without silently repairing output."""
    values = _finite(logits, "logits")
    pivot = max(values)
    exponentials = tuple(math.exp(value - pivot) for value in values)
    total = sum(exponentials)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("logits do not produce a finite probability distribution")
    return tuple(value / total for value in exponentials)


def normalized_entropy(probabilities: Sequence[float]) -> float:
    """Return ``1 - H(p)/log(K)`` for a normalized distribution."""
    values = tuple(float(value) for value in probabilities)
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("probabilities must be finite and non-negative")
    total = sum(values)
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError("probabilities must sum to one")
    if len(values) == 1:
        return 1.0
    entropy = -sum(value * math.log(value) for value in values if value > 0)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(len(values))))


def _rows(output: Any, plan: ReadoutPlan) -> list[Sequence[float]]:
    """Extract one logits row per slot from common backend output shapes."""
    if isinstance(output, Mapping):
        rows: list[Sequence[float]] = []
        for slot in plan.slots:
            if slot.question_id not in output:
                raise ValueError(f"missing output for question {slot.question_id!r}")
            row = output[slot.question_id]
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                raise ValueError("readout output rows must be sequences")
            rows.append(row)
        return rows
    if not isinstance(output, Sequence) or isinstance(output, (str, bytes)):
        raise ValueError("readout output must be a mapping or sequence")
    rows = list(output)
    if len(rows) != len(plan.slots):
        raise ValueError("readout output row count does not match readout slots")
    if any(not isinstance(row, Sequence) or isinstance(row, (str, bytes)) for row in rows):
        raise ValueError("readout output rows must be sequences")
    return rows


class _LogitReadout:
    """Shared candidate-logit implementation for token and head readouts."""

    def execute(self, output: Any, plan: ReadoutPlan) -> list[ReadoutResult]:
        rows = _rows(output, plan)
        labels = plan.labels
        if not labels:
            raise ValueError("readout plan must declare labels")
        results: list[ReadoutResult] = []
        for slot, row in zip(plan.slots, rows, strict=True):
            logits = _finite(row, "logits")
            if len(logits) != len(labels):
                raise ValueError("candidate logit count does not match labels")
            results.append(
                ReadoutResult(
                    question_id=slot.question_id,
                    candidate_ids=labels,
                    logits=logits,
                    probabilities=stable_softmax(logits),
                )
            )
        return results


class NextTokenReadout(_LogitReadout):
    """Read candidate logits from next-token positions."""


class CandidateHeadReadout(_LogitReadout):
    """Read candidate logits emitted by a Decision head."""


class EncoderHeadReadout(_LogitReadout):
    """Read candidate logits emitted by an encoder head."""


class SequenceClassificationReadout(_LogitReadout):
    """Read logits from a sequence-classification output head."""


class DiffusionCanvasReadout(_LogitReadout):
    """Legacy logit adapter only; not a diffusion executor and not registered."""


READOUTS: dict[str, type[_LogitReadout]] = {
    "next_token_logits": NextTokenReadout,
    "candidate_head": CandidateHeadReadout,
    "encoder_head": EncoderHeadReadout,

    "sequence_classification": SequenceClassificationReadout,
}


def readout_for(plan: ReadoutPlan) -> Readout:
    """Construct the explicit readout declared by a plan."""
    try:
        return READOUTS[plan.mode]()
    except KeyError as exc:
        raise ValueError(f"unsupported readout mode: {plan.mode}") from exc


__all__ = [
    "CandidateHeadReadout",
    "DiffusionCanvasReadout",
    "EncoderHeadReadout",
    "NextTokenReadout",
    "Readout",
    "SequenceClassificationReadout",
    "normalized_entropy",
    "readout_for",
    "stable_softmax",
]
