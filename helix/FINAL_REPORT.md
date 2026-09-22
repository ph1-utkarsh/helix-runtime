# Helix bounded local report

Helix implements its numerical and serving mechanisms rather than wrapping a server: stable softmax, RMSNorm, RoPE, causal attention, KV equivalence, a deterministic decoder/streamer, local HTTP/NDJSON serving, paged allocation, LRU prefix caching, fixed/adaptive schedulers, row-wise int8 quantization, and target-exact speculative decoding.

The corrected deterministic workload produced **0% adaptive throughput gain**, below the preregistered 15%; the scheduler hypothesis is unsupported. EXP-HEL-001 is excluded because it allowed causally impossible multi-token steps. On M1 Metal, explicit attention matched NumPy within `3.6e-7`; Metal was slower at lengths 16/64 and faster at 256 in nine-sample single-machine measurements. These are not CUDA, vLLM, SGLang, production-SLO, or general speed claims.

On the tiny fixture, int8 storage fell 71.9% with maximum reconstructed weight error .00279. Speculation exactly reproduced target greedy output at both 100% and 0% acceptance, but was slower than greedy in Python; the adversarial 0%-acceptance draft was about five times slower. This demonstrates why acceptance governs benefit, not a production speedup.

Reproduce core evidence with `python3 -m unittest discover -s helix -p 'test*.py'` and inspect immutable EXP-HEL-002/003 artifacts.
