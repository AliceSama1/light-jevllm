# Parallelism and context parallelism

light-jevllm now includes a CPU-testable parallel control plane inspired by the
current vLLM nightly architecture. It derives deterministic TP/PP/DP/EP group
metadata and lifecycle state, but does not create ProcessGroups or run NCCL,
Gloo, attention kernels, KV allocation, or collectives.

## Context parallel (CP)

CP is an execution-backend capability, not an HTTP trick. The safe default is
`ContextParallelPlan(mode="none", world_size=1)`. If an external vLLM backend
does not advertise a requested mode, phase, world size, model revision, or KV
layout, the request falls back to CP=1 with an observable reason.

The control plane carries only:

- absolute token ranges and positions;
- prefill/decode phase metadata;
- topology and rank descriptors;
- model revision and KV namespace/epoch;
- idempotency/deadline metadata.

It never sends or concatenates KV tensors. Decoder attention CP, MLA latent-KV
sharding, sequence parallelism, KV block ownership, and cross-rank collectives
must be implemented by the delegated vLLM worker group.

The IR types are in `jev_runtime/context_parallel.py` and are attached to
`BatchKey`/`ExecutionBatch` without changing the existing token-logit API.
`jev_runtime/parallel.py` provides the backend-neutral topology and lifecycle
contract.

A production CP backend must validate CP=1/2/4 parity, TTFT/TPOT/p99, tokens/s,
KV epoch/checksum handling, collective wait/bytes per rank, timeout/fallback,
and multimodal token-position alignment before enabling CP by default.
