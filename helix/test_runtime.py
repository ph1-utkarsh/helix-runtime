import unittest
import numpy as np
from runtime import KVCache, causal_attention, rms_norm, rope, softmax


class RuntimeTests(unittest.TestCase):
    def test_softmax_stable(self):
        np.testing.assert_allclose(softmax([10000, 10000]), [.5, .5], rtol=1e-6)
    def test_rms(self):
        y = rms_norm([[3., 4.]], [1., 1.], eps=0)
        np.testing.assert_allclose(np.mean(y * y), 1, rtol=1e-6)
    def test_rope_offset_composes(self):
        x = np.arange(24, dtype=np.float32).reshape(1, 1, 3, 8)
        np.testing.assert_allclose(rope(x)[..., 2:3, :], rope(x[..., 2:3, :], offset=2), atol=1e-6)
    def test_cache_matches_full_attention(self):
        rng = np.random.default_rng(0); q = rng.normal(size=(1, 2, 5, 4)).astype(np.float32)
        k = rng.normal(size=q.shape).astype(np.float32); v = rng.normal(size=q.shape).astype(np.float32)
        full = causal_attention(q, k, v); cache = KVCache(); pieces = []
        for index in range(5):
            kc, vc = cache.append(k[..., index:index+1, :], v[..., index:index+1, :])
            pieces.append(causal_attention(q[..., index:index+1, :], kc, vc, past=index))
        np.testing.assert_allclose(np.concatenate(pieces, axis=-2), full, atol=1e-6)


if __name__ == "__main__": unittest.main()
