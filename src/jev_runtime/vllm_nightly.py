"""Versioned control-plane connector for vLLM nightly workers.

This connector sends delegation metadata only. It never sends KV tensors,
creates process groups, or performs TP/CP collectives. A real vLLM nightly
worker may implement the referenced control-plane endpoint, while CPU tests
use the same JSON contract with a mock HTTP server.
"""
from __future__ import annotations

import asyncio
import json
import math
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Mapping

from .context_parallel import (
    CPProtocolError,
    ContextParallelPlan,
    DelegationCapabilities,
    KVHandle,
    PhasePlan,
    choose_cp_plan,
)
from .parallel import ParallelTopology

_MAX_BODY = 8 * 1024 * 1024


class VLLMNightlyError(RuntimeError):
    def __init__(self, message: str, *, kind: str = "upstream", status: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.status = status


@dataclass(frozen=True)
class VLLMNightlyOptions:
    base_url: str
    model: str | None = None
    timeout_seconds: float = 30.0
    capability_endpoint: str = "/light-jevllm/v1/capabilities"
    execute_endpoint: str = "/light-jevllm/v1/execute"
    protocol_version: str = "light-jevllm.parallel.v1"
    max_concurrency: int = 8

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str) or not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must use http or https")
        if type(self.timeout_seconds) not in (int, float) or isinstance(self.timeout_seconds, bool) or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        for name in ("capability_endpoint", "execute_endpoint"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.startswith("/"):
                raise ValueError(f"{name} must start with '/'")
        if type(self.max_concurrency) is not int or self.max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")


@dataclass(frozen=True)
class DelegationRequest:
    request_id: str
    sequence_id: str
    model: str
    revision: str | None
    topology: ParallelTopology
    context_parallel: ContextParallelPlan
    phase: PhasePlan
    payload: Mapping[str, Any]
    deadline_ms: int | None = None
    idempotency_key: str | None = None
    kv_handle: KVHandle | None = None

    def envelope(self, *, protocol_version: str) -> dict[str, Any]:
        if not self.request_id or not self.sequence_id:
            raise ValueError("request_id and sequence_id are required")
        cp = self.context_parallel
        return {
            "protocol_version": protocol_version,
            "request_id": self.request_id,
            "sequence_id": self.sequence_id,
            "model": self.model,
            "revision": self.revision,
            "topology": {
                "tensor_parallel": self.topology.tensor_parallel,
                "pipeline_parallel": self.topology.pipeline_parallel,
                "data_parallel": self.topology.data_parallel,
                "expert_parallel": self.topology.expert_parallel,
                "context_parallel": self.topology.context_parallel,
                "decode_context_parallel": self.topology.decode_context_parallel,
                "external_data_parallel": self.topology.external_data_parallel,
            },
            "context_parallel": {
                "mode": cp.mode,
                "world_size": cp.world_size,
                "rank": cp.rank,
                "topology_id": cp.topology_id,
                "shard_axis": cp.shard_axis,
                "token_range": list(cp.token_range) if cp.token_range else None,
                "absolute_position_start": cp.absolute_position_start,
                "absolute_position_end": cp.absolute_position_end,
                "kv_layout": cp.kv_layout,
                "phase": cp.phase,
                "owner": cp.owner,
            },
            "phase": {
                "prefill_tokens": self.phase.prefill_tokens,
                "decode_steps": self.phase.decode_steps,
                "microbatch": self.phase.microbatch,
                "token_budget": self.phase.token_budget,
            },
            "deadline_ms": self.deadline_ms,
            "idempotency_key": self.idempotency_key or self.request_id,
            "kv_handle": _kv_json(self.kv_handle),
            # The payload is an opaque control-plane payload. KV tensors are
            # intentionally not accepted by this schema.
            "payload": dict(self.payload),
        }


def _kv_json(handle: KVHandle | None) -> dict[str, Any] | None:
    if handle is None:
        return None
    return {
        "session_id": handle.session_id,
        "prefix_hash": handle.prefix_hash,
        "block_ids": list(handle.block_ids),
        "block_size": handle.block_size,
        "dtype": handle.dtype,
        "layout": handle.layout,
        "epoch": handle.epoch,
        "checksum": handle.checksum,
        "placement": handle.placement,
        "fingerprint": handle.fingerprint(),
    }


def _capabilities(value: Mapping[str, Any], *, protocol_version: str) -> DelegationCapabilities:
    if value.get("protocol_version") != protocol_version:
        raise VLLMNightlyError("parallel protocol version mismatch", kind="capability")
    modes = value.get("parallel_modes", [])
    if not isinstance(modes, list) or any(not isinstance(mode, str) for mode in modes):
        raise VLLMNightlyError("parallel_modes must be a string list", kind="capability")
    world_size = value.get("world_size", 1)
    if type(world_size) is not int or world_size < 1:
        raise VLLMNightlyError("capability world_size is invalid", kind="capability")
    return DelegationCapabilities(
        api=str(value.get("api", "")),
        model_revision=value.get("model_revision"),
        parallel_modes=tuple(modes),
        world_size=world_size,
        phase_split=bool(value.get("phase_split", False)),
        kv_transfer=bool(value.get("kv_transfer", False)),
        media_inputs=bool(value.get("media_inputs", False)),
        protocol_version=protocol_version,
    )


class VLLMNightlyConnector:
    name = "vllm_nightly"

    def __init__(self, options: VLLMNightlyOptions):
        self.options = options
        self.closed = False
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=options.max_concurrency, thread_name_prefix="light-jevllm-nightly")
        self._completed: dict[str, dict[str, Any]] = {}

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            self.closed = True
            executor = self._executor
            self._executor = None
        executor.shutdown(wait=False, cancel_futures=True)

    def _request_sync(self, method: str, endpoint: str, payload: Mapping[str, Any] | None) -> tuple[int, Mapping[str, Any]]:
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = urllib.request.Request(
            self.options.base_url.rstrip("/") + endpoint,
            data=data,
            headers={"accept": "application/json", "content-type": "application/json"} if data else {"accept": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.options.timeout_seconds) as response:
                status, body = int(response.status), response.read(_MAX_BODY + 1)
        except urllib.error.HTTPError as exc:
            with exc:
                status, body = int(exc.code), exc.read(_MAX_BODY + 1)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            text = str(exc).lower()
            kind = "timeout" if "timed out" in text else "transport"
            raise VLLMNightlyError("nightly upstream request failed", kind=kind) from exc
        if len(body) > _MAX_BODY:
            raise VLLMNightlyError("nightly upstream response too large", kind="response")
        try:
            parsed = json.loads(body.decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VLLMNightlyError("nightly upstream returned invalid JSON", kind="response") from exc
        if not isinstance(parsed, dict):
            raise VLLMNightlyError("nightly upstream response must be an object", kind="response")
        return status, parsed

    async def _request(self, method: str, endpoint: str, payload: Mapping[str, Any] | None, timeout: float) -> tuple[int, Mapping[str, Any]]:
        with self._lock:
            if self.closed:
                raise VLLMNightlyError("nightly connector closed", kind="closed")
            executor = self._executor
        try:
            return await asyncio.wait_for(asyncio.get_running_loop().run_in_executor(executor, self._request_sync, method, endpoint, payload), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise VLLMNightlyError("nightly upstream timeout", kind="timeout") from exc

    async def capabilities(self, *, timeout: float | None = None) -> DelegationCapabilities:
        budget = min(timeout, self.options.timeout_seconds) if timeout is not None else self.options.timeout_seconds
        status, body = await self._request("GET", self.options.capability_endpoint, None, budget)
        if status >= 400:
            raise VLLMNightlyError(f"nightly capability probe returned HTTP {status}", kind="capability", status=status)
        return _capabilities(body, protocol_version=self.options.protocol_version)

    async def execute(self, request: DelegationRequest, *, capabilities: DelegationCapabilities | None = None, timeout: float | None = None) -> dict[str, Any]:
        envelope = request.envelope(protocol_version=self.options.protocol_version)
        selected, fallback = choose_cp_plan(request.context_parallel, capabilities)
        if fallback is not None:
            envelope["context_parallel"] = {**envelope["context_parallel"], "mode": selected.mode, "world_size": selected.world_size, "rank": selected.rank, "owner": selected.owner}
            envelope["fallback_reason"] = fallback
        key = str(envelope["idempotency_key"])
        with self._lock:
            cached = self._completed.get(key)
        if cached is not None:
            return dict(cached)
        budget = min(timeout, self.options.timeout_seconds) if timeout is not None else self.options.timeout_seconds
        status, body = await self._request("POST", self.options.execute_endpoint, envelope, budget)
        if status == 409:
            raise VLLMNightlyError("nightly request idempotency conflict", kind="idempotency", status=status)
        if status >= 400:
            kind = "rate_limited" if status == 429 else "client_error" if status < 500 else "server_error"
            raise VLLMNightlyError(f"nightly execute returned HTTP {status}", kind=kind, status=status)
        if body.get("protocol_version") != self.options.protocol_version:
            raise VLLMNightlyError("nightly execute protocol mismatch", kind="response")
        result = dict(body)
        result.setdefault("fallback_reason", fallback)
        with self._lock:
            self._completed[key] = dict(result)
        return result


__all__ = [
    "DelegationRequest",
    "VLLMNightlyConnector",
    "VLLMNightlyError",
    "VLLMNightlyOptions",
]
