# Project State

- Objective: build and measure a bounded LLM inference runtime from first principles.
- Status: ACTIVE; local CPU/Metal scope approved by user's 2026-09-21 completion directive.
- Architecture: intentionally undefined.
- Activation dependency: satisfied by documented program exception in `../PROGRAM_ROADMAP.md`.
- Hardware boundary: M1 Pro, 16 GB; no CUDA/Triton claims. Implement NumPy reference plus MLX Metal paths and compare only within this machine.
- Budget: zero paid calls/cloud spend.
- Implemented: numerical oracle, tiny decoder/streamer, local HTTP/NDJSON API, paged-KV allocator and LRU prefix cache, scheduler simulation, int8 quantization, speculative decoding, CPU benchmark, and explicit MLX Metal attention.
- Evidence: EXP-HEL-002 yields 0% adaptive gain; EXP-HEL-003 max numerical error is 3.6e-7. No CUDA, vLLM, or production-serving claim.
- EXP-HEL-005: int8 representation reduces fixture weight storage 71.9% at .00279 max error. Speculative output remains target-exact at both 100% and 0% acceptance; Python implementation is slower than greedy, with low acceptance worst.
