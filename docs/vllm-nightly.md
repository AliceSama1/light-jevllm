# vLLM nightly connector

`VLLMNightlyConnector` is the light-jevllm control-plane adapter for an
external vLLM nightly worker group. It is intentionally separate from the
OpenAI-compatible completions and System One backends.

## Endpoints expected from an external worker

The connector defaults to two explicit endpoints:

```text
GET  /light-jevllm/v1/capabilities
POST /light-jevllm/v1/execute
```

The capability response must include:

```json
{
  "protocol_version": "light-jevllm.parallel.v1",
  "api": "vllm-nightly",
  "model_revision": "...",
  "parallel_modes": ["cp"],
  "world_size": 2,
  "phase_split": true,
  "kv_transfer": false,
  "media_inputs": false
}
```

The execute request is a versioned JSON control envelope containing request and
sequence IDs, model revision, topology axes, CP plan, phase plan, deadline,
idempotency key and optional KV metadata. It never contains KV tensors or
collective payloads.

## Safety behavior

- CP=1 is the default.
- Missing or incompatible capabilities cause an explicit CP=1 fallback with a
  `fallback_reason` in the envelope.
- Protocol version mismatches, stale KV metadata, duplicate idempotency
  conflicts, HTTP errors and timeouts are rejected.
- Completed responses are memoized by idempotency key within the connector
  lifetime, so a retry does not submit the same request twice.
- The connector does not claim that TP/PP/DP/EP/CP is executing locally.

## Configuration

The runtime can probe a nightly worker with:

```json
{
  "backend": "vllm_nightly",
  "upstream_url": "http://127.0.0.1:8000",
  "upstream_model": "my-model",
  "parallel_protocol_version": "light-jevllm.parallel.v1",
  "tensor_parallel": 1,
  "pipeline_parallel": 1,
  "data_parallel": 1,
  "context_parallel": 1
}
```

At this stage `vllm_nightly` is a control-plane/capability backend. Since no
Decision family is registered for its execute response yet, decision requests
return HTTP 501 rather than pretending that arbitrary nightly output is a
Decision answer. A real model-specific adapter must be added before enabling
execution.

The underlying vLLM nightly worker remains responsible for ProcessGroups,
GroupCoordinator, KV allocation, attention collectives, Model Runner V2,
MLA/DCP/PCP and GPU kernels.
