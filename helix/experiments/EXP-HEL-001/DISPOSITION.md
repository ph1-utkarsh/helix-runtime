# Invalid scheduler comparison

The static policy emitted up to eight sequential autoregressive tokens for a single request in one simulation tick, while the adaptive policy emitted one. This violates matched causal work and biases throughput. CPU attention timing remains descriptive, but the scheduler comparison is invalid and excluded. The corrected simulator permits one decode token per request per tick and is rerun under a new ID.
