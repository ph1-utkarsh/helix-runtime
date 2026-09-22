# Helix Project Charter

## Problem and motivation

Mature inference servers hide the interaction among KV memory, batching, prefill/decode scheduling, kernels, and latency objectives. Helix will implement a deliberately bounded runtime to produce experimentally defensible systems understanding.

## Research question and hypothesis

**Question:** Can a workload-adaptive scheduler improve throughput while satisfying explicit TTFT/TPOT SLOs more reliably than fixed batching and static chunk policies?

**Hypothesis:** Under mixed prompt/output lengths and controlled memory pressure, an online scheduler using queue state and KV pressure will deliver at least **15% more tokens/s** than the strongest implemented static policy while keeping both p95 TTFT and p95 TPOT within pre-registered targets on at least 95% of measurement windows.

The hypothesis is falsified if matched-workload improvements fall below the threshold, violate SLOs, or arise from correctness/quality changes.

## Target users

Inference/runtime engineers, ML systems researchers, and engineers learning production serving mechanisms from first principles.

## Success criteria

- Numerically validated loading, attention, sampling, autoregressive generation, and KV cache.
- Owned queue, scheduler, batching, paged KV allocator, eviction/prefix logic, and streaming engine.
- Matched comparisons with Hugging Face, vLLM, and SGLang where hardware permits.
- TTFT, TPOT, throughput, latency percentiles, utilization, memory, and KV metrics across registered workloads.
- Profile-guided kernel work with correctness tolerances and before/after measurements.
- Primary scheduler hypothesis, ablations, scaling limits, failure analysis, and clean reproduction.

## Non-goals

Feature parity with mature servers; implementing every model family; production multi-node serving; claiming faster-than-vLLM generally; writing CUDA/Triton kernels without a measured bottleneck.

## Constraints

Local Apple hardware supports semantic correctness but not CUDA/Triton evidence. CUDA GPU access and budget are unapproved. Start with one small decoder-only architecture and one tiny deterministic fixture. Target nine weeks after ForgeRL.

## Risks and kill criteria

Risks include numerical drift, unfair baselines, GPU measurement noise, allocator bugs, excessive model coverage, and hardware unavailability. Redesign if numerical parity cannot be reached, if CUDA access cannot support stable comparisons, if the scheduler contribution duplicates equivalent published work, or if the minimal runtime expands beyond the schedule before baseline evidence.

## Stage Gate 0 status

`DRAFT/PASS-CONDITIONAL`. The charter is sufficient for program planning, but it must be revalidated against current literature, actual CUDA access, workload/SLO values, and the lessons from ForgeRL before Helix activates.
