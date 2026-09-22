import json,threading,unittest
from urllib.request import Request,urlopen
import numpy as np
from advanced import Int8Weight,PrefixCache,speculative_generate
from engine import StreamingEngine,TinyDecoder
from http_api import serve

class AdvancedTests(unittest.TestCase):
    def test_quantization(self):
        weight=np.random.default_rng(0).normal(size=(8,16)).astype(np.float32); packed=Int8Weight.from_float(weight)
        self.assertLess(packed.bytes,weight.nbytes); self.assertLess(np.max(np.abs(weight-packed.dequantize())),.02)
    def test_prefix_lru(self):
        cache=PrefixCache(2); cache.put([1],"a"); cache.put([1,2],"b"); self.assertEqual(cache.longest([1,2,3])[1],"b")
        cache.put([9],"c"); self.assertIsNone(cache.longest([1])); self.assertEqual(cache.hit_rate,.5)
    def test_speculative_exact(self):
        target=TinyDecoder(seed=2); expected=list(target.generate([1,2],12)); actual,stats=speculative_generate(target,TinyDecoder(seed=3),[1,2],12,4)
        self.assertEqual(actual,expected); self.assertGreater(stats["proposed"],0)
    def test_http_stream(self):
        server=serve(StreamingEngine(TinyDecoder(seed=1))); thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        try:
            body=json.dumps({"prompt":[1,2],"max_new_tokens":3}).encode(); response=urlopen(Request(f"http://127.0.0.1:{server.server_port}/generate",body,{"Content-Type":"application/json"}),timeout=5)
            rows=[json.loads(line) for line in response]; self.assertEqual(len(rows),3)
        finally: server.shutdown(); server.server_close(); thread.join()

if __name__=="__main__": unittest.main()
