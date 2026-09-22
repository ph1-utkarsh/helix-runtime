import unittest
from scheduler import OutOfPages, PagedKV, Request, simulate


class SchedulerTests(unittest.TestCase):
    def test_pages_share_and_release(self):
        cache = PagedKV(4, 4); self.assertEqual(len(cache.reserve("a", 5)), 2)
        cache.fork("a", "b"); cache.release("a"); self.assertEqual(cache.used_pages, 2)
        cache.release("b"); self.assertEqual(cache.used_pages, 0); self.assertEqual(len(cache.free), 4)
    def test_capacity(self):
        cache = PagedKV(1, 4); cache.reserve("a", 4)
        with self.assertRaises(OutOfPages): cache.reserve("b", 1)
    def test_simulation_conserves_output(self):
        jobs = [Request("a",0,10,4), Request("b",1,3,2)]
        for policy in ("static", "adaptive"):
            result = simulate(jobs, policy, budget=8, chunk=4)
            self.assertEqual(result["output_tokens"], 6)


if __name__ == "__main__": unittest.main()
