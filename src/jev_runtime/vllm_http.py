"""Small, strict HTTP client for vLLM token-logit reads."""
from __future__ import annotations

import asyncio
import json
import math
import threading
import urllib.error
import urllib.request
from collections.abc import Awaitable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol
from .token_logit import TokenLogitQuestion

_MAX_BODY = 8 * 1024 * 1024

class VLLMUpstreamError(RuntimeError):
    """The upstream could not produce a valid completion response."""

class JSONTransport(Protocol):
    def __call__(self, url: str, payload: Mapping[str, Any], timeout: float) -> Awaitable[tuple[int, Mapping[str, Any]]]: ...

@dataclass(frozen=True)
class VLLMHTTPOptions:
    base_url: str
    timeout_seconds: float = 30.0
    endpoint: str = "/v1/completions"
    model: str | None = None
    max_concurrency: int = 8
    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str) or not self.base_url.startswith(("http://", "https://")): raise ValueError("base_url must use http or https")
        if not isinstance(self.timeout_seconds, (int, float)) or isinstance(self.timeout_seconds, bool) or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0: raise ValueError("timeout_seconds must be positive")
        if not self.endpoint.startswith("/"): raise ValueError("endpoint must start with '/'")
        if type(self.max_concurrency) is not int or self.max_concurrency < 1: raise ValueError("max_concurrency must be positive")

class VLLMHTTPBackend:
    name = "vllm_http"
    def __init__(self, options: VLLMHTTPOptions, transport: JSONTransport | None = None):
        self.options = options
        self._custom_transport = transport
        self._transport = transport or self._urllib_transport
        self._executor = None if transport else ThreadPoolExecutor(max_workers=max(1, options.max_concurrency), thread_name_prefix="jev-upstream")
        self._lock = threading.Lock()
        self.closed = False

    async def _urllib_transport(self, url: str, payload: Mapping[str, Any], timeout: float) -> tuple[int, Mapping[str, Any]]:
        executor = self._executor
        if executor is None:
            raise VLLMUpstreamError("backend closed")
        data = json.dumps(payload, allow_nan=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"}, method="POST")
        def request() -> tuple[int, Mapping[str, Any]]:
            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    status, body = int(response.status), response.read(_MAX_BODY + 1)
            except urllib.error.HTTPError as exc:
                status, body = int(exc.code), exc.read(_MAX_BODY + 1)
            if len(body) > _MAX_BODY: raise VLLMUpstreamError("upstream response too large")
            try: parsed = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc: raise VLLMUpstreamError("upstream returned invalid JSON") from exc
            if not isinstance(parsed, dict): raise VLLMUpstreamError("upstream JSON response must be an object")
            return status, parsed
        return await asyncio.get_running_loop().run_in_executor(executor, request)

    def close(self):
        with self._lock:
            if self.closed: return
            self.closed = True
            executor = self._executor
            self._executor = None
        if executor is not None:
            # Running urllib calls are allowed to finish at their socket timeout.
            executor.shutdown(wait=False, cancel_futures=True)

    def _payload(self, items: Sequence[TokenLogitQuestion], *, model: str | None = None) -> dict[str, Any]:
        if not items: raise ValueError("at least one token-logit question is required")
        payload = {"prompt": items[0].prompt if len(items) == 1 else [item.prompt for item in items], "max_tokens": 1, "temperature": 0, "logprobs": max(len(item.token_ids) for item in items), "echo": False}
        selected_model = model or self.options.model
        if selected_model is not None: payload["model"] = selected_model
        return payload

    async def execute(self, items: Sequence[TokenLogitQuestion], *, model: str | None = None, timeout: float | None = None) -> list[dict[str, float]]:
        with self._lock:
            if self.closed: raise VLLMUpstreamError("backend closed")
        budget = min(timeout, self.options.timeout_seconds) if timeout is not None else self.options.timeout_seconds
        if budget <= 0: raise VLLMUpstreamError("upstream request timed out")
        try:
            status, body = await asyncio.wait_for(self._transport(self.options.base_url.rstrip("/") + self.options.endpoint, self._payload(items, model=model), budget), timeout=budget)
        except asyncio.CancelledError: raise
        except asyncio.TimeoutError as exc: raise VLLMUpstreamError("upstream request timed out") from exc
        except VLLMUpstreamError: raise
        except Exception as exc: raise VLLMUpstreamError("upstream request failed") from exc
        if status >= 400: raise VLLMUpstreamError(f"upstream returned HTTP {status}")
        return self._parse(body, items)

    async def execute_many(self, batches, *, model=None, max_concurrency=None):
        limit = max_concurrency or self.options.max_concurrency
        if limit < 1: raise ValueError("max_concurrency must be positive")
        semaphore = asyncio.Semaphore(limit)
        async def one(batch):
            async with semaphore: return await self.execute(batch, model=model)
        return list(await asyncio.gather(*(one(batch) for batch in batches)))

    @staticmethod
    def _parse(body: Mapping[str, Any], items: Sequence[TokenLogitQuestion]) -> list[dict[str, float]]:
        choices = body.get("choices")
        if not isinstance(choices, list): raise VLLMUpstreamError("upstream choices must be a list")
        indexed = {}
        for choice in choices:
            if not isinstance(choice, Mapping): raise VLLMUpstreamError("completion choices must be objects")
            index = choice.get("index")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= len(items): raise VLLMUpstreamError("completion choice indices are invalid")
            if index in indexed: raise VLLMUpstreamError("completion choice indices repeat")
            indexed[index] = choice
        if set(indexed) != set(range(len(items))): raise VLLMUpstreamError("completion choice indices are incomplete")
        parsed = []
        for index, item in enumerate(items):
            logprobs = indexed[index].get("logprobs")
            top = logprobs.get("top_logprobs") if isinstance(logprobs, Mapping) else None
            if not isinstance(top, list) or not top or not isinstance(top[0], Mapping): raise VLLMUpstreamError("upstream response is missing logprobs")
            entry = top[0]
            if any(text not in entry for text in item.token_texts): raise VLLMUpstreamError("upstream omitted candidate token text")
            values = [entry[text] for text in item.token_texts]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in values): raise VLLMUpstreamError("candidate logprobs must be finite numbers")
            parsed.append(dict(zip(item.candidate_ids, (float(v) for v in values), strict=True)))
        return parsed

__all__ = ["VLLMHTTPBackend", "VLLMHTTPOptions", "VLLMUpstreamError"]
