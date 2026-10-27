import unittest

from jev_runtime.context_parallel import (
    CPProtocolError,
    ContextParallelPlan,
    DelegationCapabilities,
    KVHandle,
    PhasePlan,
    choose_cp_plan,
)
from jev_runtime.parallel import ParallelLifecycle, ParallelTopology


class ParallelContractTests(unittest.TestCase):
    def test_official_style_rank_axes_and_groups_are_deterministic(self):
        topology = ParallelTopology(tensor_parallel=2, pipeline_parallel=2, data_parallel=2)
        self.assertEqual(topology.world_size, 8)
        self.assertEqual(topology.rank_axes(5)["tensor"], 1)
        group = topology.group("tp", 5)
        self.assertEqual(group.ranks, (4, 5))
        self.assertEqual(group.local_rank, 1)

    def test_lifecycle_is_idempotent_and_validates_topology(self):
        topology = ParallelTopology()
        lifecycle = ParallelLifecycle(topology)
        lifecycle.init(1)
        lifecycle.init(1)
        lifecycle.ensure_compatible(topology)
        self.assertTrue(lifecycle.initialized)
        lifecycle.destroy()
        with self.assertRaises(RuntimeError):
            lifecycle.init(1)

    def test_cp_falls_back_to_safe_single_context(self):
        requested = ContextParallelPlan(mode="cp", world_size=2, phase="prefill", owner="external_vllm")
        selected, reason = choose_cp_plan(requested, DelegationCapabilities("/v1/systemone"))
        self.assertFalse(selected.enabled)
        self.assertEqual(reason, "delegation_capability_missing")
        selected, reason = choose_cp_plan(
            requested,
            DelegationCapabilities("cp", parallel_modes=("cp",), world_size=2, phase_split=True),
        )
        self.assertTrue(selected.enabled)
        self.assertIsNone(reason)

    def test_kv_epoch_and_checksum_reject_stale_handles(self):
        handle = KVHandle("session", "prefix", (1, 2), 16, "bf16", "paged", epoch=2, checksum="abc")
        self.assertEqual(len(handle.fingerprint()), 64)
        handle.validate(session_id="session", epoch=2, checksum="abc")
        with self.assertRaises(CPProtocolError):
            handle.validate(session_id="session", epoch=1)
        with self.assertRaises(CPProtocolError):
            handle.validate(session_id="session", epoch=2, checksum="bad")

    def test_phase_plan_validates_nonnegative_work(self):
        self.assertEqual(PhasePlan(prefill_tokens=10, decode_steps=2).prefill_tokens, 10)
        with self.assertRaises(CPProtocolError):
            PhasePlan(prefill_tokens=-1)


if __name__ == "__main__":
    unittest.main()
