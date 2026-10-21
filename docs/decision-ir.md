# Decision intermediate representation

This document defines the first compatible IR for the Jev/Decision runtime. It
is deliberately additive: current families continue to construct
`RenderedItem` and `RequestPlan`, while new families can attach the richer
metadata below.

## Flow

```text
DecisionRequest -> CompiledQuestion -> ReadoutPlan -> ExecutionBatch
                                      -> ReadoutResult -> DecisionAnswer
```

The scheduler sees only `ExecutionBatch`/`BatchKey`, cost and capabilities. It
does not inspect Vega, encoder or DiffusionGemma details.

## Types

The canonical Python definitions live in `vllm_srun/decision_ir.py`.

- `DecisionRequest`: validated model, state, questions, media and runtime
  options. It is transport-neutral and contains no tensors.
- `CompiledQuestion`: one validated question with stable candidate IDs and
  optional rendered input metadata.
- `ReadoutSlot`: one output location (token, head, encoder row or canvas
  position).
- `ReadoutPlan`: explicit readout mode, slots, labels and calibration metadata.
- `BatchKey`: immutable execution compatibility key. Different backend,
  readout mode, lane, shape or capability signature must not share a batch.
- `ExecutionBatch`: model/backend plus opaque tensors and a readout plan.
- `ReadoutResult`: raw logits/probabilities and diagnostics for one question.
- `DecisionAnswer`: API-neutral answer with confidence semantics and
  calibration version.
- `DecisionCapabilities`: public family capabilities and limits.
- `CalibrationSpec`: the meaning/version of probability calibration.

All metadata mappings are copied into immutable mapping proxies at construction
so a cache key or result cannot be changed by a caller after scheduling.

## Compatibility mapping

| IR | Existing runtime object |
| --- | --- |
| `CompiledQuestion` | `plugins.decisions.RenderedItem` |
| `ExecutionBatch` | scheduler `Batch` plus family forward tensors |
| `BatchKey` | the key used by `scheduler.batching.merged()` |
| `DecisionAnswer` | family `answer()` / `finish_surface()` result |
| request plan | `plugins.decisions.RequestPlan` / `SurfacePlan` |

The mapping is intentionally not a replacement. Existing answer probability
and confidence behavior remains family-owned because confidence types differ
(top-two margin, normalized entropy, variance, calibrated thresholds).

## Readout contract

Readouts are pure adapters over execution output:

```python
readout.execute(output, plan) -> sequence[ReadoutResult]
```

They must:

- preserve candidate order;
- reject missing or non-finite logits rather than invent probabilities;
- normalize probabilities only according to the declared plan;
- include raw backend/model diagnostics when available;
- never parse HTTP JSON or mutate a request.

The five planned modes are `next_token_logits`, `candidate_head`,
`encoder_head`, `diffusion_canvas` and `sequence_classification`.

## Capability rules

`supports_choice`, `supports_noul`, `supports_score` and media/context flags are
semantic capabilities. They are separate from accelerator capabilities and
from `LoadedModel.batch_invariant`/`packs_rows`. A scheduler may merge work
only when both its execution batch key and the family's batching flags allow
it. Unsupported options return a typed capability error.

## Confidence and calibration

`confidence_type` is required whenever confidence is returned. Examples are
`normalized_entropy`, `top_two_margin`, `normalized_variance` and
`calibrated_probability`. `calibration_version` identifies the exact policy;
absence means the output is uncalibrated, not that it is universally
comparable. A low value is diagnostic only and never silently changes the
choice.

## Security and parity

IR objects never contain executable package code. Model revision and hashes
remain owned by verified package metadata. Raw logits may be retained for
internal diagnostics, but public responses must pass the existing finite JSON
serializer and must not log sensitive state by default.
