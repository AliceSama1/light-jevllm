# Jev-Decision runtime architecture notes

## Scope

The formal implementation remains `vllm_srun`. `openjev-main` and `dzyy-vllm-main`
are read-only references; neither is imported or modified by the runtime.

## Boundaries

```text
HTTP/OpenAPI
    -> runtime validation and lifecycle
    -> ModelFamily (package, rendering, readout and answers)
    -> Scheduler/Batcher (admission, deadline, lanes, cost and batch shape)
    -> Engine (backbone execution)
    -> Accelerator (device and kernels)
```

A family owns model-specific semantics. An engine never parses a package or a
question schema, and a scheduler never branches on a model name. The existing
`SurfacePlan`/`WorkItem` contract is the compatibility layer for built-in
families while the explicit Decision IR is adopted incrementally.

## Reference findings

### Existing `vllm_srun`

- `api/openapi.yaml` is the contract for decisions, classify, embeddings,
  rerank, bundle, health and metrics.
- `runtime.py` verifies manifests and pinned revisions, places one model at a
  time, creates one scheduler per model, performs golden readiness checks and
  owns result-cache lifecycle.
- `plugins/base.py` separates `ModelFamily`, `Engine`, `EngineModel` and
  `Accelerator`. `LoadedModel` already exposes batch invariance, packed rows,
  lanes and shared-context hooks.
- `scheduler/batching.py` has reference, coalesced, merged and shared-context
  paths. Reference execution is retained for parity; optimized paths require
  family declarations and records.
- `families/decision2` verifies the package manifest, model identity, pinned
  base revision and candidate head; its answer semantics are family-owned.
- Packages are data only: the runtime verifies bytes and never executes code
  shipped inside a model package. `trust_remote_code` is not used.

### `openjev-main`

OpenJev contributes Decision semantics, not a universal execution engine. Its
DiffusionGemma path compiles an answer template, locates one-token label slots,
reads only the required canvas positions, and aggregates samples. It also
provides useful bounded-concurrency and overload semantics. These concepts map
to a dedicated `DiffusionCanvasReadout`/family and must not be applied to Vega
or ordinary next-token models.

### `dzyy-vllm-main`

The v5 reference contributes decoder-only execution ideas: fixed KV blocks,
reference-counted prefix blocks, longest-prefix lookup, prefill/decode mixing
and stable-shape CUDA Graph buckets. These belong inside a decoder-compatible
engine. They must not replace the generic scheduler for encoder, head or
diffusion families. Its dtype parsing is not a complete quantization loader.

## Current implementation decision

The first increment adds an explicit, serializable Decision IR and readout
protocol while preserving the existing `RenderedItem`, `RequestPlan`,
`SurfacePlan` and answer implementations. The IR carries model/backend,
question slots, batch identity, calibration metadata and raw diagnostics. The
readout layer only converts execution outputs to results; it does not parse HTTP
requests.

Confidence is model-semantic. The shared helpers provide normalized entropy as
an opt-in metric, but existing families retain their established confidence
rules (for example Decision 1/Vela top-two margin versus Decision 2 entropy).
No low-confidence answer is silently changed.

## Optimization guardrails

1. Reference/golden execution stays exact and available.
2. Batching is enabled only where a family declares the relevant capability.
3. Result cache, template/slot cache and decoder KV/prefix cache remain
   separate lifecycles and metrics.
4. BF16/FP16, fused kernels and CUDA Graphs require parity records under
   `docs/records/`; otherwise the exact path remains the default.
5. New backends report unsupported capability instead of falling back to an
   incompatible readout.

## Current standalone components

The independent `jev_runtime.media`, `jev_runtime.calibration` and
`jev_runtime.parity` packages are now available without changing service or
legacy directories. Media normalization enforces a strict image/video MIME
allowlist, byte/pixel/frame bounds and SHA-256 identity; dimensions and frame
counts are supplied by an injectable probe, so video decoding is not required.
Calibration is pure Python temperature scaling with normalized-entropy
confidence and explicit version metadata. The parity runner executes or
compares reference/optimized outputs, reports choice and argmax agreement plus
probability drift, and exposes a configurable pass gate. These components are
library-only and are not wired into the HTTP service yet.

## Next increments

1. Attach `DecisionCapabilities` and `BatchKey` to family planning.
2. Add a token-logit family backed by an HTTP vLLM adapter with strict
   single-token label and complete-logprob validation.
3. Add Decision2 parity records before enabling any approximate kernel or
   decoder cache optimization.
4. Add independent encoder and diffusion families with family-specific
   readouts and benchmarks.
