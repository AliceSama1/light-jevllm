"""Typed configuration; opt-in components do not change service defaults."""
from __future__ import annotations
import math
import re
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, Mapping


def _integer(name: str, value: int, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class RuntimeConfig:
    model: str
    upstream_url: str
    upstream_model: str
    family: str = "token_logit"
    backend: str = "vllm_http"
    workers: int = 4
    queue_capacity: int = 8
    timeout_seconds: float = 30.0
    close_timeout_seconds: float = 5.0
    max_request_bytes: int = 1_048_576
    host: str = "127.0.0.1"
    port: int = 8080
    tokenizer_json: str | None = None
    allow_nonlocal_bind: bool = False
    policy_mode: str = "reference"
    metrics_namespace: str = "jev"

    template_cache: bool = False
    result_cache: bool = False
    deterministic_results: bool = False
    model_revision: str = ""
    tokenizer_identity: str = ""
    cache_entries: int = 128
    cache_bytes: int = 16777216
    cache_ttl_seconds: float = 60.0
    temperature: float = 1.0
    calibration_version: str | None = None
    batch_window_ms: float = 2.0
    batch_max_rows: int = 64
    batch_max_cost: int = 1048576

    def __post_init__(self) -> None:
        for name in ("template_cache", "result_cache", "deterministic_results"):
            if type(getattr(self, name)) is not bool: raise ValueError(name + " must be boolean")
        for name in ("cache_entries", "cache_bytes", "batch_max_rows", "batch_max_cost"):
            _integer(name, getattr(self, name), 1)
        for name in ("cache_ttl_seconds", "temperature", "batch_window_ms"):
            v = getattr(self, name)
            if type(v) not in (int, float) or not math.isfinite(v) or v <= 0: raise ValueError(name + " must be positive")
        if self.temperature != 1 and not self.calibration_version: raise ValueError("non-unit temperature requires calibration_version")
        if self.result_cache and not (self.deterministic_results and self.model_revision and self.tokenizer_identity): raise ValueError("result cache requires determinism, revision and tokenizer identity")
        if self.backend == "vllm_systemone" and (self.policy_mode == "coalesced" or self.result_cache or self.template_cache or self.temperature != 1): raise ValueError("SystemOne optimization unsupported")
        if self.policy_mode not in ("reference", "coalesced", "disabled"):
            raise ValueError("policy_mode must be reference, coalesced or disabled")
        if not isinstance(self.metrics_namespace, str) or not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", self.metrics_namespace):
            raise ValueError("invalid metrics namespace")
        for name in ("model", "upstream_url", "upstream_model", "family", "backend", "host"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-blank text")
        _integer("workers", self.workers, 1)
        _integer("queue_capacity", self.queue_capacity)
        _integer("max_request_bytes", self.max_request_bytes, 1)
        _integer("port", self.port, 1)
        if self.port > 65535 or self.max_request_bytes > 8 * 1024 * 1024:
            raise ValueError("port or max_request_bytes exceeds limit")
        if type(self.allow_nonlocal_bind) is not bool:
            raise ValueError("allow_nonlocal_bind must be boolean")
        if self.host not in ("localhost", "127.0.0.1", "::1") and not self.allow_nonlocal_bind:
            raise ValueError("non-loopback bind requires allow_nonlocal_bind=true")
        for value in (self.timeout_seconds, self.close_timeout_seconds):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("timeouts must be finite and positive")
        if self.tokenizer_json is not None and (not isinstance(self.tokenizer_json, str) or not self.tokenizer_json.strip()):
            raise ValueError("tokenizer_json must be non-blank text")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RuntimeConfig:
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
        return cls(**value)

    def as_dict(self) -> dict[str, Any]:
        """Legacy Runtime accepts this mapping without component integration."""
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None}


@dataclass(frozen=True)
class CacheConfig:
    max_entries: int = 1024
    max_size: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        _integer("max_entries", self.max_entries, 1)
        _integer("max_size", self.max_size, 1)


@dataclass(frozen=True)
class MetricsConfig:
    namespace: str = "jev"

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, str) or not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", self.namespace):
            raise ValueError("invalid metrics namespace")
