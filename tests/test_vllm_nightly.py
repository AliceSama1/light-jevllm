import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from jev_runtime.context_parallel import ContextParallelPlan, DelegationCapabilities, PhasePlan
from jev_runtime.parallel import ParallelTopology
from jev_runtime.vllm_nightly import (
    DelegationRequest,
    VLLMNightlyConnector,
    VLLMNightlyError,
    VLLMNightlyOptions,
)


class NightlyMock(BaseHTTPRequestHandler):
    capabilities = {
        "protocol_version": "light-jevllm.parallel.v1",
        "api": "mock-vllm-nightly",
        "model_revision": "rev1",
        "parallel_modes": ["cp"],
        "world_size": 2,
        "phase_split": True,
        "kv_transfer": True,
        "media_inputs": False,
    }
    execute_calls = []
    mode = "ok"

    def log_message(self, *_args):
        pass

    def do_GET(self):  # noqa: N802
        body = json.dumps(self.capabilities).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.execute_calls.append(payload)
        if self.mode == "version":
            response = {"protocol_version": "wrong", "result": "bad"}
        elif self.mode == "timeout":
            time.sleep(0.2)
            response = {"protocol_version": "light-jevllm.parallel.v1", "result": "late"}
        else:
            response = {
                "protocol_version": "light-jevllm.parallel.v1",
                "request_id": payload["request_id"],
                "result": {"ok": True},
                "fallback_reason": payload.get("fallback_reason"),
            }
        body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except OSError:
            pass


class NightlyConnectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        NightlyMock.execute_calls = []
        NightlyMock.mode = "ok"
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), NightlyMock)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        NightlyMock.execute_calls.clear()
        NightlyMock.mode = "ok"
        self.connector = VLLMNightlyConnector(
            VLLMNightlyOptions(
                f"http://127.0.0.1:{self.server.server_port}",
                timeout_seconds=0.5,
            )
        )

    def tearDown(self):
        self.connector.close()

    def request(self, cp):
        return DelegationRequest(
            request_id="req-1",
            sequence_id="seq-1",
            model="demo",
            revision="rev1",
            topology=ParallelTopology(context_parallel=cp.world_size),
            context_parallel=cp,
            phase=PhasePlan(prefill_tokens=128),
            payload={"input_tokens": 128},
            deadline_ms=500,
            idempotency_key="idem-1",
        )

    def test_capability_probe_and_cp_request(self):
        caps = asyncio_run(self.connector.capabilities())
        self.assertTrue(caps.supports("cp", 2))
        result = asyncio_run(self.connector.execute(self.request(ContextParallelPlan(mode="cp", world_size=2, owner="external_vllm")), capabilities=caps))
        self.assertTrue(result["result"]["ok"])
        self.assertEqual(NightlyMock.execute_calls[0]["context_parallel"]["mode"], "cp")

    def test_missing_capability_falls_back_to_cp_one_and_is_idempotent(self):
        requested = ContextParallelPlan(mode="cp", world_size=4, owner="external_vllm")
        caps = DelegationCapabilities("mock", parallel_modes=("cp",), world_size=2, phase_split=True)
        first = asyncio_run(self.connector.execute(self.request(requested), capabilities=caps))
        second = asyncio_run(self.connector.execute(self.request(requested), capabilities=caps))
        self.assertEqual(first, second)
        self.assertEqual(len(NightlyMock.execute_calls), 1)
        sent = NightlyMock.execute_calls[0]
        self.assertEqual(sent["context_parallel"]["mode"], "none")
        self.assertEqual(sent["context_parallel"]["world_size"], 1)
        self.assertEqual(sent["fallback_reason"], "delegation_capability_missing")

    def test_protocol_mismatch_and_timeout_are_rejected(self):
        NightlyMock.mode = "version"
        with self.assertRaises(VLLMNightlyError) as context:
            asyncio_run(self.connector.execute(self.request(ContextParallelPlan.safe())))
        self.assertEqual(context.exception.kind, "response")
        NightlyMock.mode = "timeout"
        with self.assertRaises(VLLMNightlyError) as context:
            asyncio_run(self.connector.execute(self.request(ContextParallelPlan.safe()), timeout=0.01))
        self.assertEqual(context.exception.kind, "timeout")


def asyncio_run(awaitable):
    import asyncio
    return asyncio.run(awaitable)


if __name__ == "__main__":
    unittest.main()
