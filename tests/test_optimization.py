import unittest

from jev_runtime.optimization import (
    MemoryCurve,
    ParallelTopology,
    PrefixBlockPool,
    cuda_graph_eligibility,
)


class OptimizationPolicyTests(unittest.TestCase):
    def test_prefix_pool_refcount_and_capacity(self):
        pool = PrefixBlockPool(1)
        first = pool.acquire("a")
        self.assertEqual(pool.acquire("a").block_id, first.block_id)
        pool.release("a")
        pool.release("a")
        pool.acquire("b")
        with self.assertRaises(RuntimeError):
            pool.acquire("c")

    def test_graph_and_topology_are_declarative(self):
        self.assertFalse(cuda_graph_eligibility(dynamic_shapes=True, batch_size=1, max_batch_size=1).enabled)
        self.assertTrue(cuda_graph_eligibility(dynamic_shapes=False, batch_size=2, max_batch_size=4).enabled)
        self.assertEqual(ParallelTopology(2, 2, 3).world_size, 12)

    def test_memory_curve_interpolates(self):
        curve = MemoryCurve(((0, 10), (100, 110), (200, 210)))
        self.assertEqual(curve.estimate(50), 60)
        self.assertEqual(curve.estimate(300), 210)


if __name__ == "__main__":
    unittest.main()
