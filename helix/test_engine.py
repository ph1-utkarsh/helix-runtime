import unittest
from engine import StreamingEngine,TinyDecoder

class EngineTests(unittest.TestCase):
    def test_deterministic_and_bounded(self):
        a=list(StreamingEngine(TinyDecoder(seed=3)).stream([1,2],5)); b=list(StreamingEngine(TinyDecoder(seed=3)).stream([1,2],5))
        self.assertEqual(a,b); self.assertEqual(len(a),5); self.assertTrue(all(0<=x<32 for x in a))
    def test_empty_rejected(self):
        with self.assertRaises(ValueError): list(StreamingEngine(TinyDecoder()).stream([],1))
    def test_eos_stops(self):
        model=TinyDecoder(seed=1); first=next(model.generate([2],1)); self.assertEqual(len(list(model.generate([2],10,eos=first))),1)

if __name__=="__main__": unittest.main()
