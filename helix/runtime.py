"""Small, explicit NumPy inference primitives used as Helix's correctness oracle."""
import numpy as np


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float32)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    values = np.exp(shifted)
    return values / np.sum(values, axis=axis, keepdims=True)


def rms_norm(x, weight, eps=1e-6):
    x = np.asarray(x, dtype=np.float32)
    return x * np.asarray(weight, dtype=np.float32) / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def causal_attention(q, k, v, past=0):
    """Shapes [batch, heads, query/key length, head dimension]."""
    q, k, v = map(lambda a: np.asarray(a, dtype=np.float32), (q, k, v))
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
    qi = np.arange(q.shape[-2])[:, None] + past
    kj = np.arange(k.shape[-2])[None, :]
    scores = np.where(kj <= qi, scores, -np.inf)
    return softmax(scores, axis=-1) @ v


def rope(x, offset=0, theta=10000.0):
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] % 2: raise ValueError("RoPE dimension must be even")
    positions = np.arange(offset, offset + x.shape[-2], dtype=np.float32)
    inv = theta ** (-np.arange(0, x.shape[-1], 2, dtype=np.float32) / x.shape[-1])
    angles = positions[:, None] * inv[None, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x); out[..., 0::2] = even * np.cos(angles) - odd * np.sin(angles)
    out[..., 1::2] = even * np.sin(angles) + odd * np.cos(angles)
    return out


class KVCache:
    def __init__(self): self.k = None; self.v = None
    def append(self, k, v):
        axis = -2
        self.k = np.asarray(k) if self.k is None else np.concatenate((self.k, k), axis=axis)
        self.v = np.asarray(v) if self.v is None else np.concatenate((self.v, v), axis=axis)
        return self.k, self.v
