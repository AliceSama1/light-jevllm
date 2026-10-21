import asyncio
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from jev_runtime.decision_ir import Candidate
from jev_runtime.token_logit import TokenLogitAdapter, TokenLogitError
from jev_runtime.vllm_http import VLLMHTTPBackend, VLLMHTTPOptions, VLLMUpstreamError


class Handler(BaseHTTPRequestHandler):
    response = {}
    delay = 0
    seen = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers["content-length"])
        Handler.seen.append(json.loads(self.rfile.read(length)))
        if Handler.delay:
            import time
            time.sleep(Handler.delay)
        body = json.dumps(Handler.response).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


class TokenLogitHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join()

    def setUp(self):
        Handler.delay = 0
        Handler.seen.clear()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.adapter = TokenLogitAdapter(lambda text: {"yes": [11], "no": [12]}[text])
        self.items = [self.adapter.compile("q", "choice", "Answer", ["yes", "no"])]

    def response(self, choices):
        Handler.response = {"choices": choices}

    def choice(self, index=0, values=None):
        return {"index": index, "logprobs": {"top_logprobs": [values or {"yes": -1.0, "no": -2.0}]}}

    def run_backend(self, timeout=1):
        return asyncio.run(VLLMHTTPBackend(VLLMHTTPOptions(self.url, timeout_seconds=timeout)).execute(self.items))

    def test_normal_and_batch_prompt(self):
        self.response([self.choice()])
        self.assertEqual(self.run_backend(), [{"yes": -1.0, "no": -2.0}])
        self.response([self.choice(0), self.choice(1)])
        items = self.items * 2
        self.assertEqual(asyncio.run(VLLMHTTPBackend(VLLMHTTPOptions(self.url)).execute(items))[1]["yes"], -1.0)
        self.assertIsInstance(Handler.seen[-1]["prompt"], list)

    def test_reordered_missing_nonfinite_and_indices(self):
        self.response([self.choice(1), self.choice(0)])
        self.assertEqual(asyncio.run(VLLMHTTPBackend(VLLMHTTPOptions(self.url)).execute(self.items * 2)), [{"yes": -1.0, "no": -2.0}, {"yes": -1.0, "no": -2.0}])
        self.response([self.choice(0)])
        self.assertEqual(self.run_backend(), [{"yes": -1.0, "no": -2.0}])
        for choices, message in [
            ([self.choice(0, {"yes": -1.0})], "omitted"),
            ([self.choice(0, {"yes": float("nan"), "no": -2.0})], "finite"),
            ([self.choice(0), self.choice(0)], "repeat"),
        ]:
            self.response(choices)
            with self.assertRaisesRegex(VLLMUpstreamError, message):
                self.run_backend()

    def test_errors_timeout_and_cancel(self):
        backend = VLLMHTTPBackend(VLLMHTTPOptions(self.url))
        async def failed(*_):
            return 503, {"error": "busy"}
        with self.assertRaisesRegex(VLLMUpstreamError, "HTTP 503"):
            asyncio.run(VLLMHTTPBackend(VLLMHTTPOptions(self.url), failed).execute(self.items))
        Handler.delay = 0.2
        with self.assertRaisesRegex(VLLMUpstreamError, "timed out"):
            self.run_backend(timeout=0.01)
        async def cancelled():
            task = asyncio.create_task(backend.execute(self.items))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        asyncio.run(cancelled())

    def test_compile_and_decode_contracts(self):
        with self.assertRaises(TokenLogitError):
            self.adapter.compile("q", "noul", "x", [Candidate("true"), Candidate("other")])
        result = self.adapter.decode(self.items[0], [2.0, 0.0])
        self.assertEqual(result["confidence_type"], "normalized_entropy")
        self.assertIsNone(result["calibration_version"])


if __name__ == "__main__":
    unittest.main()
