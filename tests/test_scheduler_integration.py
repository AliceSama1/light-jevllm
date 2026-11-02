import asyncio
import threading
import time
import unittest

from jev_runtime.scheduling.scheduler import BatchKey, RequestScheduler
from jev_runtime.token_logit import TokenLogitAdapter


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    async def execute(self, items, timeout=None):
        with self.lock:
            self.calls.append((list(items), timeout))
        await asyncio.sleep(0)
        return [{"yes": -0.0, "no": -1.0} for _ in items]


class SchedulerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.adapter = TokenLogitAdapter(lambda text: {"yes": [1], "no": [2]}[text])
        self.items = [self.adapter.compile("q", "choice", "prompt", ["yes", "no"])]

    def test_compatible_requests_are_sent_as_one_batch_and_split(self):
        backend = FakeBackend()
        scheduler = RequestScheduler(
            backend, capacity=8, window_ms=10, max_rows=8, max_cost=10000
        )
        key = BatchKey("m", "r", "t")
        results = []
        errors = []
        barrier = threading.Barrier(2)

        def run():
            try:
                barrier.wait()
                results.append(scheduler.execute(key, self.items, time.monotonic() + 2))
            except Exception as exc:  # pragma: no cover - assertion below
                errors.append(exc)

        threads = [threading.Thread(target=run) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        scheduler.close()
        self.assertFalse(errors)
        self.assertEqual(len(results), 2)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(len(backend.calls[0][0]), 2)
        self.assertEqual(scheduler.stats()["batches_executed"], 1)

    def test_incompatible_keys_do_not_share_batch(self):
        backend = FakeBackend()
        scheduler = RequestScheduler(
            backend, capacity=8, window_ms=1, max_rows=8, max_cost=10000
        )
        try:
            for revision in ("r1", "r2"):
                self.assertEqual(
                    len(scheduler.execute(BatchKey("m", revision, "t"), self.items, time.monotonic() + 2)),
                    1,
                )
        finally:
            scheduler.close()
        self.assertEqual(len(backend.calls), 2)

    def test_expired_request_is_rejected(self):
        backend = FakeBackend()
        scheduler = RequestScheduler(
            backend, capacity=2, window_ms=20, max_rows=8, max_cost=10000
        )
        try:
            with self.assertRaisesRegex(Exception, "deadline"):
                scheduler.execute(BatchKey("m", "r", "t"), self.items, time.monotonic() - 1)
        finally:
            scheduler.close()
        self.assertEqual(backend.calls, [])


if __name__ == "__main__":
    unittest.main()
