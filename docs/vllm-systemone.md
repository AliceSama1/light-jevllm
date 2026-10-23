# light-jevllm vLLM 0.32 System One backend

`VLLMSystemOneBackend` is an independent HTTP backend for vLLM 0.32's native
`POST /v1/systemone` surface. It does not replace or alter
`VLLMHTTPBackend`, which remains the OpenAI-compatible `/v1/completions`
token-logit backend.

The service uses `GET /v1/models` capability probes for readiness and model metadata; execution does not require a separate probe. Missing optional capability metadata defaults only to the three
contract question kinds (`choice`, `noul`, and `score`); malformed capability
payloads are rejected. Request failures are classified as client (400/429),
server (500+), transport, timeout, or capability failures through
`VLLMSystemOneError.kind` and `.status`.

Responses are validated strictly by `systemone_mapper.py`: answer type,
probability normalization and bounds, required value fields, score legends,
model, answers, and usage are checked. Unknown fields and incompatible shapes
are rejected rather than silently ignored. Probability maps retain the
upstream's key order and answer order is validated when an expected question
order is supplied.

Confidence is copied from the upstream response exactly. This adapter does
not derive normalized entropy, add `confidence_type`, or fabricate a
calibration version. vLLM System One's native confidence semantics must remain
upstream-owned; if a deployment omits or changes fields required by the
contract, the response is rejected as incompatible.

## Service configuration

Set `backend` to `vllm_systemone`; omit `tokenizer_json`. Both local decision
endpoints forward one request to upstream `/v1/systemone`. The configured
upstream model replaces the local alias on dispatch; the alias is restored on
response. Local `options.deadline_ms` is consumed, not forwarded. `token_labels`
is completion-only and rejected on the native path. No answer caching is enabled.

`policy_mode` accepts `reference` (default) or `disabled`. Neither executes
scheduler batching. Admission remains bounded in both modes. Request byte and
question counts are cost proxies, not GPU time or token estimates. Queue wait
seconds/depth and running counts describe service work, not upstream batches.
Native confidence, probabilities, score legends, per-answer errors, usage and
meta are preserved. Timeouts map to 504, upstream 429 to 429, upstream client
errors to 400, and transport/server/invalid-response failures to 502. A failed
capability probe returns 503 rather than fabricated capability metadata.
