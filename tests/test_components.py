"""Component-layer unit tests; these policies and providers do not execute models."""
import unittest

from jev_runtime.cache import ContentCache, ResultCache, SizeAwareLRU, TemplateCache, FeatureCache
from jev_runtime.config import CacheConfig, MetricsConfig, RuntimeConfig
from jev_runtime.core import CostEstimate, Deadline, ExecutionOutput, HealthStatus, ModelIdentity
from jev_runtime.metrics import MemoryMetricsSink, PrometheusTextSink
from jev_runtime.plugins import Backend, Family, Readout, Registry
from jev_runtime.scheduling import AdmissionPolicy, BatchPolicy


class CoreConfigTests(unittest.TestCase):
    def test_typed_config_validation_and_legacy_mapping(self):
        config = RuntimeConfig("local", "http://127.0.0.1:8000", "upstream", workers=2)
        self.assertEqual(config.as_dict()["workers"], 2)
        self.assertEqual(RuntimeConfig.from_mapping(config.as_dict()), config)
        with self.assertRaises(ValueError):
            RuntimeConfig("", "http://localhost", "upstream")
        with self.assertRaises(ValueError):
            RuntimeConfig.from_mapping({"model": "x", "bad": True})
        self.assertEqual(CacheConfig().max_entries, 1024)
        self.assertEqual(MetricsConfig().namespace, "jev")

    def test_ir_extensions(self):
        identity = ModelIdentity("router", version="1", family="decision")
        deadline = Deadline.after_ms(500, now=10)
        self.assertEqual(deadline.remaining(now=10.25), 0.25)
        self.assertFalse(deadline.expired(now=10.25))
        self.assertTrue(deadline.expired(now=10.5))
        with self.assertRaises(ValueError):
            CostEstimate(input_tokens=-1)
        status = HealthStatus(True, details={"worker": "ready"})
        with self.assertRaises(TypeError):
            status.details["worker"] = "down"
        output = ExecutionOutput((1, 2), model=identity, cost=CostEstimate(compute_ms=2.5))
        self.assertEqual(output.model, identity)


class PluginTests(unittest.TestCase):
    def test_registry_and_structural_protocols(self):
        class TestFamily:
            def compile(self, request): return request
            def readout(self, items, output): return output
            def close(self): pass
        registry = Registry()
        registry.register("family", TestFamily)
        self.assertIsInstance(registry.create("family"), Family)
        self.assertEqual(registry.factories["family"], TestFamily)
        with self.assertRaises(ValueError):
            registry.register("family", TestFamily)
        with self.assertRaises(ValueError):
            registry.create("missing")
        self.assertTrue(isinstance(type("TestBackend", (), {"execute": lambda *a, **k: None, "close": lambda *a: None})(), Backend))
        self.assertTrue(isinstance(type("TestReadout", (), {"execute": lambda *a: None})(), Readout))


class SchedulingTests(unittest.TestCase):
    def test_admission_policy(self):
        policy = AdmissionPolicy(max_concurrent=2, queue_capacity=1)
        self.assertTrue(policy.decide(0).admitted)
        self.assertEqual(policy.decide(2, 0).reason, "queued")
        self.assertFalse(policy.decide(2, 1).admitted)
        with self.assertRaises(ValueError):
            AdmissionPolicy(max_concurrent=0)

    def test_batch_policy_is_declarative(self):
        self.assertEqual(BatchPolicy.reference().max_batch_size, 1)
        self.assertEqual(BatchPolicy.coalesced(8, 3).mode, "coalesced")
        with self.assertRaises(ValueError):
            BatchPolicy("reference", max_batch_size=4)
        with self.assertRaises(ValueError):
            BatchPolicy("execute")


class CacheTests(unittest.TestCase):
    def test_size_aware_lru_eviction_and_stats(self):
        cache = SizeAwareLRU(max_entries=2, max_size=5, size_of=len)
        cache.put("a", "aa")
        cache.put("b", "bb")
        self.assertEqual(cache.get("a"), "aa")
        cache.put("c", "ccc")
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("a"), "aa")
        stats = cache.stats()
        self.assertEqual((stats.entries, stats.size, stats.evictions), (2, 5, 1))
        cache.put("large", "too large")
        self.assertNotIn("large", cache)
        cache.pop("a")
        self.assertNotIn("a", cache)
        cache.clear()
        self.assertEqual(cache.stats().entries, 0)

    def test_replacement_and_invalid_weights(self):
        cache = SizeAwareLRU(max_entries=2, max_size=4, size_of=len)
        cache.put("a", "aa")
        cache.put("a", "oversized")
        self.assertEqual(cache.get("a"), "aa")
        cache.put("a", "b")
        self.assertEqual(cache.stats().size, 1)
        cache.put("b", "b")
        cache.put("c", "c")
        self.assertNotIn("a", cache)
        for weight in (-1, 1.5, True):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                SizeAwareLRU(size_of=lambda value: weight).put("x", None)

    def test_typed_cache_providers(self):
        for cache_type in (ResultCache, TemplateCache, ContentCache, FeatureCache):
            cache = cache_type(max_entries=1, max_size=1024)
            cache.put("key", {"answer": 1})
            self.assertEqual(cache.get("key"), {"answer": 1})


class MetricsTests(unittest.TestCase):
    def test_memory_and_prometheus_text(self):
        sink = MemoryMetricsSink()
        sink.increment("requests_total", labels={"route": "decision"})
        sink.increment("requests_total", 2, {"route": "decision"})
        sink.gauge("active", 1)
        snapshot = sink.snapshot()
        self.assertEqual(snapshot[("requests_total", (("route", "decision"),))], 3)
        self.assertEqual(snapshot[("active", ())], 1)
        text = PrometheusTextSink()
        text.increment("requests_total", labels={"route": "decision\napi"})
        text.gauge("active", 2)
        rendered = text.render()
        self.assertIn('# TYPE active gauge\nactive 2', rendered)
        self.assertIn('route="decision\\napi"', rendered)
        self.assertIn('# TYPE requests_total counter', rendered)
        with self.assertRaises(ValueError):
            text.increment("requests_total", -1)
        with self.assertRaises(ValueError):
            text.gauge("bad metric", 1)


if __name__ == "__main__":
    unittest.main()
