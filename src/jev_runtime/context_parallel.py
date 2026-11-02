"""Context-parallel control-plane metadata and validation.

CP=1 is the safe default. This module never shards tensors or performs
collectives; those operations belong to the delegated vLLM worker group.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from .parallel import ParallelTopology


class CPProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class ContextParallelPlan:
    mode: Literal["none", "cp", "mla_cp", "sp"] = "none"
    world_size: int = 1
    rank: int = 0
    topology_id: str | None = None
    shard_axis: str = "sequence"
    token_range: tuple[int, int] | None = None
    absolute_position_start: int = 0
    absolute_position_end: int = 0
    kv_layout: str | None = None
    phase: Literal["prefill", "decode"] = "prefill"
    owner: Literal["external_vllm", "native_backend", "reference"] = "reference"

    def __post_init__(self) -> None:
        if self.mode == "none" and self.world_size != 1:
            raise CPProtocolError("mode=none requires world_size=1")
        if type(self.world_size) is not int or self.world_size < 1:
            raise CPProtocolError("world_size must be positive")
        if type(self.rank) is not int or not 0 <= self.rank < self.world_size:
            raise CPProtocolError("rank is outside context-parallel world")
        if self.absolute_position_start < 0 or self.absolute_position_end < self.absolute_position_start:
            raise CPProtocolError("invalid absolute position range")
        if self.token_range is not None:
            start, end = self.token_range
            if type(start) is not int or type(end) is not int or not 0 <= start <= end:
                raise CPProtocolError("invalid token range")

    @classmethod
    def safe(cls) -> "ContextParallelPlan":
        return cls()

    @property
    def enabled(self) -> bool:
        return self.mode != "none" and self.world_size > 1

    def capability_signature(self) -> tuple[str, ...]:
        return (self.mode, str(self.world_size), self.shard_axis, self.kv_layout or "", self.phase, self.owner)

    def validate_topology(self, topology: ParallelTopology) -> None:
        if self.world_size != topology.context_parallel:
            raise CPProtocolError("CP plan world size does not match topology")


@dataclass(frozen=True)
class KVHandle:
    session_id: str
    prefix_hash: str
    block_ids: tuple[int, ...] = ()
    block_size: int = 0
    dtype: str = ""
    layout: str = ""
    epoch: int = 0
    checksum: str = ""
    placement: str = "external"

    def __post_init__(self) -> None:
        if not self.session_id or not self.prefix_hash:
            raise CPProtocolError("KV handle needs session_id and prefix_hash")
        if type(self.epoch) is not int or self.epoch < 0:
            raise CPProtocolError("KV epoch must be non-negative")
        if type(self.block_size) is not int or self.block_size < 0:
            raise CPProtocolError("KV block_size must be non-negative")
        if any(type(block) is not int or block < 0 for block in self.block_ids):
            raise CPProtocolError("KV block IDs must be non-negative integers")

    def fingerprint(self) -> str:
        raw = "|".join((self.session_id, self.prefix_hash, str(self.epoch), self.dtype, self.layout, ",".join(map(str, self.block_ids))))
        return hashlib.sha256(raw.encode()).hexdigest()

    def validate(self, *, session_id: str, epoch: int, checksum: str | None = None) -> None:
        if self.session_id != session_id or self.epoch != epoch:
            raise CPProtocolError("stale or mismatched KV handle")
        if checksum is not None and self.checksum and checksum != self.checksum:
            raise CPProtocolError("KV handle checksum mismatch")


@dataclass(frozen=True)
class PhasePlan:
    prefill_tokens: int = 0
    decode_steps: int = 0
    microbatch: int = 1
    token_budget: int = 0

    def __post_init__(self) -> None:
        for name in ("prefill_tokens", "decode_steps", "token_budget"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise CPProtocolError(f"{name} must be non-negative integer")
        if type(self.microbatch) is not int or self.microbatch < 1:
            raise CPProtocolError("microbatch must be positive")


@dataclass(frozen=True)
class DelegationCapabilities:
    api: str
    model_revision: str | None = None
    parallel_modes: tuple[str, ...] = ()
    world_size: int = 1
    phase_split: bool = False
    kv_transfer: bool = False
    media_inputs: bool = False
    protocol_version: str = "1"

    def supports(self, mode: str, world_size: int) -> bool:
        return mode in self.parallel_modes and world_size <= self.world_size

    def as_dict(self) -> dict[str, object]:
        return {
            "api": self.api,
            "model_revision": self.model_revision,
            "parallel_modes": list(self.parallel_modes),
            "world_size": self.world_size,
            "phase_split": self.phase_split,
            "kv_transfer": self.kv_transfer,
            "media_inputs": self.media_inputs,
            "protocol_version": self.protocol_version,
        }


def choose_cp_plan(requested: ContextParallelPlan, capabilities: DelegationCapabilities | None) -> tuple[ContextParallelPlan, str | None]:
    """Select a safe plan; unsupported requests fall back to CP=1."""
    if not requested.enabled:
        return ContextParallelPlan.safe(), None
    if capabilities is None or not capabilities.supports(requested.mode, requested.world_size):
        return ContextParallelPlan.safe(), "delegation_capability_missing"
    if requested.phase == "decode" and not capabilities.phase_split:
        return ContextParallelPlan.safe(), "decode_phase_unsupported"
    return requested, None
