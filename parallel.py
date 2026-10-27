"""Backend-neutral parallel topology and group lifecycle descriptors.

This module intentionally performs no distributed initialization. It provides
stable rank/group derivation for delegated backends such as vLLM nightly and a
CPU-testable lifecycle contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


class ParallelConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class ParallelTopology:
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    data_parallel: int = 1
    expert_parallel: int = 1
    context_parallel: int = 1
    decode_context_parallel: int = 1
    external_data_parallel: int = 1

    def __post_init__(self) -> None:
        names = (
            "tensor_parallel", "pipeline_parallel", "data_parallel",
            "expert_parallel", "context_parallel", "decode_context_parallel",
            "external_data_parallel",
        )
        for name in names:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ParallelConfigurationError(f"{name} must be a positive integer")
        if self.expert_parallel not in (1, self.tensor_parallel * self.data_parallel):
            raise ParallelConfigurationError(
                "expert_parallel must be 1 or tensor_parallel * data_parallel"
            )

    @property
    def model_parallel_world_size(self) -> int:
        return self.tensor_parallel * self.pipeline_parallel * self.data_parallel

    @property
    def world_size(self) -> int:
        return self.model_parallel_world_size * self.external_data_parallel

    @property
    def cp_world_size(self) -> int:
        return self.context_parallel

    @property
    def is_single_process(self) -> bool:
        return self.world_size == 1

    def validate_world(self, world_size: int) -> None:
        if type(world_size) is not int or world_size != self.world_size:
            raise ParallelConfigurationError(
                f"world_size={world_size!r} does not match topology world_size={self.world_size}"
            )

    def rank_axes(self, rank: int) -> dict[str, int]:
        self.validate_rank(rank)
        value = rank
        axes: dict[str, int] = {}
        for name, size in (
            ("tensor", self.tensor_parallel),
            ("pipeline", self.pipeline_parallel),
            ("data", self.data_parallel),
            ("external_data", self.external_data_parallel),
        ):
            axes[name] = value % size
            value //= size
        return axes

    def validate_rank(self, rank: int) -> None:
        if type(rank) is not int or not 0 <= rank < self.world_size:
            raise ParallelConfigurationError(f"rank={rank!r} is outside topology")

    def group(self, name: str, rank: int) -> "GroupDescriptor":
        axes = self.rank_axes(rank)
        if name == "tp":
            ranks = tuple(self._rank(t, axes["pipeline"], axes["data"], axes["external_data"])
                          for t in range(self.tensor_parallel))
        elif name == "pp":
            ranks = tuple(self._rank(axes["tensor"], p, axes["data"], axes["external_data"])
                          for p in range(self.pipeline_parallel))
        elif name == "dp":
            ranks = tuple(self._rank(axes["tensor"], axes["pipeline"], d, axes["external_data"])
                          for d in range(self.data_parallel))
        elif name == "external_dp":
            ranks = tuple(self._rank(axes["tensor"], axes["pipeline"], axes["data"], e)
                          for e in range(self.external_data_parallel))
        elif name in ("cp", "pcp", "dcp"):
            # CP/PCP/DCP are delegated execution groups. Their rank lists are
            # intentionally not invented here because vLLM's layouts depend on
            # model runner and attention implementation details.
            size = self.context_parallel if name != "dcp" else self.decode_context_parallel
            ranks = tuple(range(size))
        else:
            raise ParallelConfigurationError(f"unknown parallel group: {name}")
        return GroupDescriptor(name, ranks, ranks.index(rank) if rank in ranks else 0, len(ranks), "external")

    def _rank(self, tensor: int, pipeline: int, data: int, external_data: int) -> int:
        return (((external_data * self.data_parallel + data) * self.pipeline_parallel + pipeline)
                * self.tensor_parallel + tensor)


@dataclass(frozen=True)
class GroupDescriptor:
    name: str
    ranks: tuple[int, ...]
    local_rank: int
    world_size: int
    backend: str = "external"

    def __post_init__(self) -> None:
        if not self.name or not self.ranks or len(set(self.ranks)) != len(self.ranks):
            raise ParallelConfigurationError("group must have a unique non-empty rank list")
        if not 0 <= self.local_rank < len(self.ranks) or self.world_size != len(self.ranks):
            raise ParallelConfigurationError("invalid group local rank or world size")


class ParallelLifecycle:
    """Idempotent control-plane lifecycle; no process groups are created."""

    def __init__(self, topology: ParallelTopology):
        self.topology = topology
        self.state = "new"
        self._world_size: int | None = None

    def init(self, world_size: int = 1) -> None:
        self.topology.validate_world(world_size)
        if self.state == "destroyed":
            raise RuntimeError("parallel lifecycle cannot be reinitialized")
        if self.state == "initialized" and self._world_size != world_size:
            raise ParallelConfigurationError("parallel lifecycle already initialized differently")
        self._world_size = world_size
        self.state = "initialized"

    def ensure_compatible(self, topology: ParallelTopology) -> None:
        if topology != self.topology:
            raise ParallelConfigurationError("parallel topology does not match lifecycle")
        if self.state not in ("initialized", "ready"):
            raise RuntimeError("parallel lifecycle is not initialized")
        self.state = "ready"

    def destroy(self) -> None:
        self.state = "destroyed"

    @property
    def initialized(self) -> bool:
        return self.state in ("initialized", "ready")
