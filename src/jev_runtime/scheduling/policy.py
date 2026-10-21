"""Pure scheduling/admission policy data structures."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class AdmissionDecision:
    admitted: bool
    reason: str = "accepted"

@dataclass(frozen=True)
class AdmissionPolicy:
    """Bound active work and waiting work; this policy never runs requests."""
    max_concurrent: int = 4
    queue_capacity: int = 8

    def __post_init__(self) -> None:
        if type(self.max_concurrent) is not int or self.max_concurrent < 1:
            raise ValueError("max_concurrent must be a positive integer")
        if type(self.queue_capacity) is not int or self.queue_capacity < 0:
            raise ValueError("queue_capacity must be a non-negative integer")

    def decide(self, active: int, queued: int = 0) -> AdmissionDecision:
        if type(active) is not int or type(queued) is not int or active < 0 or queued < 0:
            raise ValueError("active and queued counts must be non-negative integers")
        if active < self.max_concurrent:
            return AdmissionDecision(True)
        if queued < self.queue_capacity:
            return AdmissionDecision(True, "queued")
        return AdmissionDecision(False, "capacity_exhausted")

@dataclass(frozen=True)
class BatchPolicy:
    """Declarative batching choice. It does not group or execute model work."""
    mode: Literal["reference", "coalesced"] = "reference"
    max_batch_size: int = 1
    max_wait_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in ("reference", "coalesced"):
            raise ValueError("mode must be reference or coalesced")
        if type(self.max_batch_size) is not int or self.max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        if type(self.max_wait_ms) not in (int, float) or not math.isfinite(self.max_wait_ms) or self.max_wait_ms < 0:
            raise ValueError("max_wait_ms must be finite and non-negative")
        if self.mode == "reference" and (self.max_batch_size != 1 or self.max_wait_ms != 0):
            raise ValueError("reference policy has batch size 1 and no wait")

    @classmethod
    def reference(cls) -> "BatchPolicy":
        return cls("reference", 1, 0.0)

    @classmethod
    def coalesced(cls, max_batch_size: int = 16, max_wait_ms: float = 2.0) -> "BatchPolicy":
        return cls("coalesced", max_batch_size, max_wait_ms)

ReferenceBatchPolicy = BatchPolicy.reference()
