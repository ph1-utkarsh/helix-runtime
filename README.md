# Helix

**A small inference runtime rebuilt from first principles and measured on Apple Metal and CPU.**

Helix owns the mechanisms that are normally hidden behind an inference framework: numerically stable attention, RoPE, RMSNorm, KV caching, paged allocation, prefix reuse, continuous scheduling, streaming HTTP output, INT8 weights, and exact speculative decoding.

## Measured results

| Result | Measurement |
|---|---:|
| Metal vs NumPy maximum absolute error | **3.58×10⁻⁷** |
| INT8 weight-storage reduction | **71.875%** |
| Maximum quantized-weight error | **0.00279** |
| Metal speedup at sequence length 256 | **3.67×** |
| Speculative output parity | **100%** |
| Adaptive scheduler gain in frozen simulation | **0%** |

The negative results matter. Metal loses to NumPy on tiny inputs because launch overhead dominates. Speculative decoding is slower in the tiny Python fixture even at 100% acceptance, and becomes substantially worse with an adversarial draft. Helix records those boundaries instead of presenting an unsupported production-speed claim.

## Components

```text
HTTP NDJSON stream
        │
continuous scheduler ──► prefix-aware admission
        │
tiny decoder ──► stable attention / RoPE / RMSNorm
        │
paged KV cache ──► sharing / eviction
        │
INT8 weights + exact speculative verification
```

- `helix/runtime.py` — numerical reference primitives
- `helix/engine.py` — tiny decoder and streaming engine
- `helix/scheduler.py` — request simulation and paged KV allocation
- `helix/advanced.py` — prefix LRU, INT8 weights and speculative decoding
- `helix/http_api.py` — newline-delimited JSON streaming service
- `helix/experiments/` — immutable benchmark evidence

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m unittest discover -s helix -p 'test*.py' -v
python3 scripts/verify.py
python3 helix/benchmark.py --run-id EXP-LOCAL
python3 helix/advanced_benchmark.py --run-id EXP-ADV-LOCAL
```

The Metal benchmark additionally requires Apple Silicon and MLX:

```bash
pip install mlx
python3 helix/metal_benchmark.py --run-id EXP-METAL-LOCAL
```

## Evidence map

- `EXP-HEL-002`: corrected scheduler result; no adaptive gain
- `EXP-HEL-003`: explicit Metal-vs-CPU numerical and latency comparison
- `EXP-HEL-004`: seeded speculative pilot and its limitation
- `EXP-HEL-005`: quantization plus matched/adversarial speculative decoding
- `helix/FINAL_REPORT.md`: defensible claims and excluded CUDA scope

## Scope

Helix demonstrates correctness, mechanism ownership and honest local measurement. It does not claim CUDA-kernel parity with vLLM or SGLang. The repository intentionally labels Metal/CPU substitutions and fixture-scale conclusions.

## License

MIT © 2026 Sanjay Bisen.
