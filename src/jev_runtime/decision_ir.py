"""Transport-neutral Decision IR shared by families, readouts and backends.

The runtime's existing ``RenderedItem`` and ``SurfacePlan`` remain compatible
with this additive IR. These objects intentionally contain no executable model
code and no HTTP-specific parsing.
"""

from __future__ import annotations

import math
from .core.ir import ModelIdentity, Deadline, CostEstimate, HealthStatus, ExecutionOutput
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

ReadoutMode = Literal[
    "next_token_logits",
    "candidate_head",
    "encoder_head",
    "diffusion_canvas",
    "sequence_classification",
]
QuestionKind = Literal["noul", "choice", "score"]


def _frozen_metadata(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True)
class CalibrationSpec:
    """Probability/confidence semantics for one readout plan."""

    confidence_type: str | None = None
    calibration_type: str | None = None
    version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata))


@dataclass(frozen=True)
class Candidate:
    """Stable candidate identity and optional human-readable description."""

    id: str
    label: str | None = None
    description: Any = None


@dataclass(frozen=True)
class DecisionRequest:
    """Validated, transport-neutral decision request."""

    model: str
    state: Any
    questions: tuple["CompiledQuestion", ...]
    images: tuple[Any, ...] = ()
    deadline_ms: int | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-blank string")
        if self.deadline_ms is not None and (
            isinstance(self.deadline_ms, bool)
            or not isinstance(self.deadline_ms, int)
            or self.deadline_ms < 0
        ):
            raise ValueError("deadline_ms must be a non-negative integer")
        object.__setattr__(self, "questions", tuple(self.questions))
        object.__setattr__(self, "images", tuple(self.images))
        object.__setattr__(self, "options", _frozen_metadata(self.options))


@dataclass(frozen=True)
class CompiledQuestion:
    """A question after family validation, before model-specific execution."""

    id: str
    kind: QuestionKind
    candidates: tuple[Candidate, ...] = ()
    instructions: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("question id must not be blank")
        if self.kind not in ("noul", "choice", "score"):
            raise ValueError(f"unsupported question kind: {self.kind}")
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata))


@dataclass(frozen=True)
class ReadoutSlot:
    """One stable output location consumed by a readout adapter."""

    question_id: str
    index: int
    label: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("readout slot index must be non-negative")
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata))


@dataclass(frozen=True)
class ReadoutPlan:
    """Explicit output/readout semantics for one execution batch."""

    mode: ReadoutMode
    slots: tuple[ReadoutSlot, ...] = ()
    labels: tuple[str, ...] = ()
    calibration: CalibrationSpec | None = None
    constraints: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "slots", tuple(self.slots))
        object.__setattr__(self, "labels", tuple(self.labels))
        object.__setattr__(self, "constraints", _frozen_metadata(self.constraints))
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("readout labels must be unique")


@dataclass(frozen=True)
class BatchKey:
    """Compatibility key; work with different values must not share a forward."""

    model_id: str
    backend: str
    mode: ReadoutMode
    lane: str = "text"
    shape: tuple[int, ...] = ()
    capability_signature: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "capability_signature", tuple(self.capability_signature))


@dataclass(frozen=True)
class ExecutionBatch:
    """Opaque tensors plus the explicit plan consumed by an execution backend."""

    model_id: str
    backend: str
    tensors: Any
    readout_plan: ReadoutPlan
    batch_key: BatchKey
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.batch_key.model_id != self.model_id or self.batch_key.backend != self.backend:
            raise ValueError("batch key does not identify this execution batch")
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata))


@dataclass(frozen=True)
class ReadoutResult:
    """Raw and normalized result for one question."""

    question_id: str
    candidate_ids: tuple[str, ...]
    logits: tuple[float, ...] = ()
    probabilities: tuple[float, ...] | None = None
    raw_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_ids", tuple(self.candidate_ids))
        object.__setattr__(self, "logits", tuple(float(v) for v in self.logits))
        if self.probabilities is not None:
            object.__setattr__(self, "probabilities", tuple(float(v) for v in self.probabilities))
        object.__setattr__(self, "raw_metadata", _frozen_metadata(self.raw_metadata))


@dataclass(frozen=True)
class DecisionAnswer:
    """API-neutral answer and its confidence semantics."""

    question_id: str
    kind: QuestionKind
    value: Any
    probabilities: Mapping[str, float] = field(default_factory=dict)
    confidence: float | None = None
    confidence_type: str | None = None
    calibration_version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        probabilities = {str(k): float(v) for k, v in self.probabilities.items()}
        if any(not math.isfinite(value) or value < 0 for value in probabilities.values()):
            raise ValueError("probabilities must be finite and non-negative")
        if probabilities and not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-6):
            raise ValueError("probabilities must sum to one")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between zero and one")
        object.__setattr__(self, "probabilities", MappingProxyType(probabilities))
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata))


@dataclass(frozen=True)
class DecisionCapabilities:
    """Semantic capabilities exposed by a Decision family."""

    supports_noul: bool = True
    supports_choice: bool = True
    supports_score: bool = True
    supports_images: bool = False
    supports_shared_context: bool = False
    supports_request_batching: bool = True
    supports_read_batching: bool = True
    supports_streaming: bool = False
    max_context_tokens: int | None = None
    max_choices: int | None = None
    max_score_levels: int | None = None
    confidence_type: str = "model_defined"
    calibration_type: str | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extras", _frozen_metadata(self.extras))

    def supports_kind(self, kind: str) -> bool:
        return {
            "noul": self.supports_noul,
            "choice": self.supports_choice,
            "score": self.supports_score,
        }.get(kind, False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "supports_noul": self.supports_noul,
            "supports_choice": self.supports_choice,
            "supports_score": self.supports_score,
            "supports_images": self.supports_images,
            "supports_shared_context": self.supports_shared_context,
            "supports_request_batching": self.supports_request_batching,
            "supports_read_batching": self.supports_read_batching,
            "supports_streaming": self.supports_streaming,
            "max_context_tokens": self.max_context_tokens,
            "max_choices": self.max_choices,
            "max_score_levels": self.max_score_levels,
            "confidence_type": self.confidence_type,
            "calibration_type": self.calibration_type,
            **dict(self.extras),
        }
