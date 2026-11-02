"""Optional pure-Python optimization policies.

These policies describe eligibility and ownership; they do not allocate GPU
memory, capture CUDA graphs, or execute model work. A real engine must provide
its own implementation before enabling them in production.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from typing import Hashable, Callable, Any


@dataclass(frozen=True)
class PrefixBlock:
    block_id: int
    token_hash: str
    refcount: int = 0


class PrefixBlockPool:
    """Small reference block pool for tests and policy validation only."""

    def __init__(self, capacity: int):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._blocks: OrderedDict[Hashable, PrefixBlock] = OrderedDict()
        self._next_id = 0

    def acquire(self, token_hash: str) -> PrefixBlock:
        block = self._blocks.get(token_hash)
        if block is not None:
            self._blocks.move_to_end(token_hash)
            updated = PrefixBlock(block.block_id, block.token_hash, block.refcount + 1)
            self._blocks[token_hash] = updated
            return updated
        while len(self._blocks) >= self.capacity:
            victim_key, victim = next(iter(self._blocks.items()))
            if victim.refcount:
                raise RuntimeError("all prefix blocks are referenced")
            del self._blocks[victim_key]
        block = PrefixBlock(self._next_id, token_hash, 1)
        self._next_id += 1
        self._blocks[token_hash] = block
        return block

    def release(self, token_hash: str) -> None:
        block = self._blocks.get(token_hash)
        if block is None:
            raise KeyError(token_hash)
        if block.refcount < 1:
            raise RuntimeError("prefix block reference underflow")
        self._blocks[token_hash] = PrefixBlock(block.block_id, block.token_hash, block.refcount - 1)

    def stats(self) -> dict[str, int]:
        return {"entries": len(self._blocks), "referenced": sum(b.refcount > 0 for b in self._blocks.values())}


@dataclass(frozen=True)
class GraphEligibility:
    enabled: bool
    reason: str


def cuda_graph_eligibility(*, dynamic_shapes: bool, batch_size: int, max_batch_size: int) -> GraphEligibility:
    if dynamic_shapes:
        return GraphEligibility(False, "dynamic_shapes")
    if type(batch_size) is not int or type(max_batch_size) is not int or batch_size < 1 or max_batch_size < batch_size:
        return GraphEligibility(False, "invalid_batch_shape")
    return GraphEligibility(True, "stable_shape")


@dataclass(frozen=True)
class ParallelTopology:
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    data_parallel: int = 1
    expert_parallel: int = 1

    def __post_init__(self) -> None:
        for name in ("tensor_parallel", "pipeline_parallel", "data_parallel", "expert_parallel"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def world_size(self) -> int:
        return self.tensor_parallel * self.pipeline_parallel * self.data_parallel


@dataclass(frozen=True)
class MemoryCurve:
    """Measured peak-memory points used by an admission policy."""

    points: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if any(type(tokens) is not int or type(bytes_) is not int or tokens < 0 or bytes_ < 0 for tokens, bytes_ in self.points):
            raise ValueError("memory curve points must be non-negative integer pairs")
        if tuple(sorted(self.points)) != self.points:
            raise ValueError("memory curve points must be sorted")

    def estimate(self, tokens: int) -> int:
        if not self.points:
            return 0
        if tokens <= self.points[0][0]:
            return self.points[0][1]
        for (left_tokens, left_mem), (right_tokens, right_mem) in zip(self.points, self.points[1:]):
            if tokens <= right_tokens:
                ratio = (tokens - left_tokens) / max(1, right_tokens - left_tokens)
                return round(left_mem + ratio * (right_mem - left_mem))
        return self.points[-1][1]
