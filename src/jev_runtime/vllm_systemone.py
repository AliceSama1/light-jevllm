"""Strict client for vLLM 0.32's native ``/v1/systemone`` endpoint.

This backend deliberately does not reuse :mod:`vllm_http`: System One owns the
request/response contract and returns already mapped answers, rather than
completion choices/logprobs.  It is additive; ``VLLMHTTPBackend`` remains the
completion backend.
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

from .decision_ir import DecisionCapabilities

_MAX_BODY = 8 * 1024 * 1024


class VLLMSystemOneError(RuntimeError):
    """A transport, upstream, capability, or response-contract failure."""

    def __init__(self, message: str, *, kind: str = "upstream", status: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.status = status


@dataclass(frozen=True)
class VLLMSystemOneOptions:
    base_url: str
    model: str | None = None
    timeout_seconds: float = 30.0
    endpoint: str = "/v1/systemone"
    max_concurrency: int = 8

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str) or not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must use http or https")
        if (isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float))
                or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0):
            raise ValueError("timeout_seconds must be positive")
        if type(self.max_concurrency) is not int or self.max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")
        if not isinstance(self.endpoint, str) or not self.endpoint.startswith("/"):
            raise ValueError("endpoint must start with '/'")


class VLLMSystemOneBackend:
    name = "vllm_systemone"

    def __init__(self, options: VLLMSystemOneOptions):
        self.options = options
        self.closed = False
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=options.max_concurrency, thread_name_prefix="light-jevllm-systemone")

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            self.closed = True
            executor = self._executor
            self._executor = None  # type: ignore[assignment]
        executor.shutdown(wait=False, cancel_futures=True)

    def _request_sync(self, method: str, url: str, payload: Mapping[str, Any] | None) -> tuple[int, Mapping[str, Any]]:
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
        headers = {"accept": "application/json"}
        if data is not None:
            headers["content-type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.options.timeout_seconds) as response:
                status, body = int(response.status), response.read(_MAX_BODY + 1)
        except urllib.error.HTTPError as exc:
            with exc:
                status, body = int(exc.code), exc.read(_MAX_BODY + 1)
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            reason = getattr(exc, "reason", None)
            timed_out = isinstance(exc, TimeoutError) or isinstance(reason, (TimeoutError, asyncio.TimeoutError)) or "timed out" in str(exc).lower()
            raise VLLMSystemOneError("upstream request timed out" if timed_out else "upstream request failed", kind="timeout" if timed_out else "transport") from exc
        if len(body) > _MAX_BODY:
            raise VLLMSystemOneError("upstream response too large", kind="response")
        try:
            parsed = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VLLMSystemOneError("upstream returned invalid JSON", kind="response", status=status) from exc
        if not isinstance(parsed, dict):
            raise VLLMSystemOneError("upstream JSON response must be an object", kind="response", status=status)
        return status, parsed

    async def _request(self, method: str, url: str, payload: Mapping[str, Any] | None, timeout: float) -> tuple[int, Mapping[str, Any]]:
        with self._lock:
            if self.closed:
                raise VLLMSystemOneError("backend closed", kind="closed")
            executor = self._executor
        try:
            return await asyncio.wait_for(asyncio.get_running_loop().run_in_executor(executor, self._request_sync, method, url, payload), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise VLLMSystemOneError("upstream request timed out", kind="timeout") from exc

    async def probe_capabilities(self, *, timeout: float | None = None) -> DecisionCapabilities:
        """Probe the OpenAI model card and expose only advertised capabilities.

        vLLM 0.32 model cards may omit capability metadata; omission means the
        native System One surface is assumed to support the three contract
        question kinds, but never changes confidence semantics.
        """
        budget = min(timeout, self.options.timeout_seconds) if timeout is not None else self.options.timeout_seconds
        status, body = await self._request("GET", self.options.base_url.rstrip("/") + "/v1/models", None, budget)
        if status >= 400:
            raise VLLMSystemOneError(f"capability probe returned HTTP {status}", kind="capability", status=status)
        data = body.get("data")
        if not isinstance(data, list):
            raise VLLMSystemOneError("capability probe response is incompatible", kind="capability", status=status)
        models = [entry for entry in data if isinstance(entry, Mapping)]
        if self.options.model is not None and not any(entry.get("id") == self.options.model for entry in models):
            raise VLLMSystemOneError("capability probe did not advertise the requested model", kind="capability", status=status)
        card = next((entry for entry in models if entry.get("id") == self.options.model), models[0] if models else None)
        if card is None:
            raise VLLMSystemOneError("capability probe returned no models", kind="capability", status=status)
        raw = card.get("capabilities", {})
        if raw is not None and not isinstance(raw, Mapping):
            raise VLLMSystemOneError("capability probe returned incompatible capabilities", kind="capability", status=status)
        raw = raw or {}
        for key in ('supports_noul','supports_choice','supports_score','supports_images'):
            if key in raw and type(raw[key]) is not bool:
                raise VLLMSystemOneError('invalid capability flag', kind='capability')
        if not isinstance(raw.get('confidence_type','upstream'),str):
            raise VLLMSystemOneError('invalid confidence type', kind='capability')
        return DecisionCapabilities(
            supports_noul=bool(raw.get("supports_noul", True)),
            supports_choice=bool(raw.get("supports_choice", True)),
            supports_score=bool(raw.get("supports_score", True)),
            supports_images=bool(raw.get("supports_images", False)),
            supports_request_batching=False, supports_read_batching=False,
            confidence_type=str(raw.get("confidence_type", "upstream")),
            calibration_type=raw.get("calibration_type"),
            extras={"source": "vllm-systemone", "model": card.get("id")},
        )

    async def execute(self, request: Mapping[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        if not isinstance(request, Mapping):
            raise ValueError("System One request must be an object")
        payload = dict(request)
        if self.options.model is not None:
            payload.setdefault("model", self.options.model)
        budget = min(timeout, self.options.timeout_seconds) if timeout is not None else self.options.timeout_seconds
        status, body = await self._request("POST", self.options.base_url.rstrip("/") + self.options.endpoint, payload, budget)
        if status >= 400:
            kind = "client_error" if 400 <= status < 429 else "rate_limited" if status == 429 else "server_error"
            raise VLLMSystemOneError(f"upstream returned HTTP {status}", kind=kind, status=status)
        from .systemone_mapper import map_systemone_response
        expected = request.get("questions")
        expected_ids = tuple(expected) if isinstance(expected, Mapping) else None
        result = map_systemone_response(body, expected_question_ids=expected_ids)
        if payload.get('model') is not None and result['model'] != payload['model']:
            raise VLLMSystemOneError('response model mismatch', kind='response')
        if isinstance(expected, Mapping):
            for key, question in expected.items():
                if question.get('type') and result['answers'][key]['type'] != question['type']:
                    raise VLLMSystemOneError('response question type mismatch', kind='response')
        return result


__all__ = ["VLLMSystemOneBackend", "VLLMSystemOneError", "VLLMSystemOneOptions"]
