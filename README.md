# Jev standalone decision runtime

A separate Python project under `jev-runtime/`, with no imports from `vllm_srun`. It implements a strict token-logit family and an explicit plugin registry/lifecycle, using a stdlib bounded HTTP server.

## Run from source

```bash
cd jev-runtime
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m jev_runtime.cli serve --config config.example.json
```

`tokenizers` is optional for production and is only needed to load an explicit local `tokenizer.json`; no remote code is loaded or executed. The upstream is configured explicitly by URL and model. With `backend=vllm_http`, the service calls the upstream's OpenAI-compatible `/v1/completions` endpoint with `logprobs` and requires every candidate token text in `top_logprobs[0]`.

`backend` may be `vllm_http` (completion/logit adapter) or `vllm_systemone` (native upstream `/v1/systemone` proxy). The latter preserves upstream answer and confidence semantics and exposes upstream capability metadata at `/v1/models`; HTTP/client/upstream contract failures map to 4xx/502/504 without fabricating answers. `policy_mode` may be `reference`, `coalesced` (token-logit only), or `disabled`; coalesced mode uses a bounded prompt scheduler and actually sends compatible prompts in one completions request. Native System One mode intentionally does not coalesce. Native mode needs no tokenizer. Completion mode retains `normalized_entropy` confidence and null calibration version. Request cost, queue/admission, batch and cache metrics are exposed through `/metrics`.

Endpoints: `/v1/decisions`, `/v1/systemone`, `/v1/models`, `/health`, `/health/live`, `/metrics`. Admission is bounded and returns 429 when full. Deadline and upstream timeout handling are real; a timed-out stdlib socket is not falsely reported as cancelled (the worker may remain occupied until its socket timeout). Shutdown rejects new requests, waits only up to `close_timeout_seconds` (default 5 seconds), then closes plugins. The HTTP protocol schema is `api/openapi.yaml`.

The default bind is loopback. Non-loopback binding is a trusted-operator configuration, requires `allow_nonlocal_bind: true`, and provides no authentication or TLS; place it behind an authenticated/TLS reverse proxy when exposed.

Example configuration is in `config.example.json`. This implementation is text/token-logit only: it does not claim to execute Diffusion models, and real GPU behavior has not been validated in this environment.
