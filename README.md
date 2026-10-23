# light-jevllm

A lightweight, modular, and pluggable Jev/Decision inference runtime. The distribution and primary CLI are named `light-jevllm`; the Python import package remains `jev_runtime` for compatibility.

## Run from source

```bash
cd light-jevllm
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m jev_runtime.cli serve --config config.example.json
```

Example configuration is in `config.example.json`. This implementation is text/token-logit and delegated System One only: it does not claim to execute native Decision3, encoder, or Diffusion models, and real GPU behavior has not been validated in this environment.

`backend` may be `vllm_http` (completion/logit adapter) or `vllm_systemone` (native upstream `/v1/systemone` proxy). The latter preserves upstream answer and confidence semantics and exposes upstream capability metadata at `/v1/models`; HTTP/client/upstream contract failures map to 4xx/502/504 without fabricating answers. `policy_mode` may be `reference`, `coalesced` (token-logit only), or `disabled`; coalesced mode uses a bounded prompt scheduler and actually sends compatible prompts in one completions request. Native System One mode intentionally does not coalesce. Native mode needs no tokenizer. Completion mode retains `normalized_entropy` confidence and null calibration version. Request cost, queue/admission, batch and cache metrics are exposed through `/metrics`.

## Current status

Implemented and CPU-tested: token-logit completions, vLLM 0.32 System One delegation, reference/coalesced/disabled policy modes, bounded request batching for completions, deterministic result TTL/LRU caching, temperature/calibration metadata, parity and media-validation utilities, scheduler/cache/queue metrics, and the served `/openapi.yaml` contract.

Delegated or not yet implemented: native Decision3/encoder/diffusion execution, local GPU model loading, real KV/cache execution, CUDA Graph capture, TP/PP/DP/EP communication, authentication/TLS, and production replica orchestration. Media normalizer and prefix/graph/topology utilities are policy/reference components only; they do not claim native model execution.


Endpoints: `/v1/decisions`, `/v1/systemone`, `/v1/models`, `/openapi.yaml`, `/health`, `/health/live`, `/metrics`. Admission is bounded and returns 429 when full. Deadline and upstream timeout handling are real; a timed-out stdlib socket is not falsely reported as cancelled (the worker may remain occupied until its socket timeout). Shutdown rejects new requests, waits only up to `close_timeout_seconds` (default 5 seconds), then closes plugins. The HTTP protocol schema is `api/openapi.yaml`.

The default bind is loopback. Non-loopback binding is a trusted-operator configuration, requires `allow_nonlocal_bind: true`, and provides no authentication or TLS; place it behind an authenticated/TLS reverse proxy when exposed.
