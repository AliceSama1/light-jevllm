import asyncio
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from jev_runtime.vllm_systemone import VLLMSystemOneBackend, VLLMSystemOneError, VLLMSystemOneOptions


class SystemOneUpstream(BaseHTTPRequestHandler):
    mode = "ok"
    response = None

    def log_message(self, *_args):
        pass

    def _reply(self, status, value):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.mode == "probe_bad":
            return self._reply(200, {"data": {}})
        self._reply(200, {"data": [{"id": "model", "capabilities": {"confidence_type": "upstream"}}]})

    def do_POST(self):
        self.__class__.seen = (self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        if self.mode == "timeout":
            time.sleep(0.2)
        if self.mode == "400":
            return self._reply(400, {"error": "bad request"})
        if self.mode == "429":
            return self._reply(429, {"error": "busy"})
        if self.mode == "500":
            return self._reply(500, {"error": "broken"})
        self._reply(200, self.response or {})


class SystemOneBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), SystemOneUpstream)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join()

    def setUp(self):
        SystemOneUpstream.mode = "ok"
        self.backend = VLLMSystemOneBackend(VLLMSystemOneOptions(
            f"http://127.0.0.1:{self.server.server_port}", model="model", timeout_seconds=.05))

    def tearDown(self):
        self.backend.close()

    def response(self, kind, probabilities, **extra):
        answer = {"type": kind, "probabilities": probabilities, "confidence": .75, **extra}
        SystemOneUpstream.response = {"model": "model", "answers": {"q": answer}, "usage": {"questions": 1}}

    def test_choice_noul_score_adaptation_and_native_confidence(self):
        for kind, extra in [
            ("choice", {"choice": "yes"}),
            ("noul", {"noul": .8}),
            ("score", {"score": 1.2, "legend": {"0": "low", "1": "high"}}),
        ]:
            self.response(kind, {"yes": .8, "no": .2} if kind == "choice" else {"true": .8, "false": .2} if kind == "noul" else {"0": .4, "1": .6}, **extra)
            result = asyncio.run(self.backend.execute({"state": "x", "questions": {"q": {"type": kind}}}))
            self.assertEqual(result["answers"]["q"]["type"], kind)
            self.assertEqual(result["answers"]["q"]["confidence"], .75)
            self.assertNotIn("confidence_type", result["answers"]["q"])

    def test_probe_and_http_error_classes(self):
        caps = asyncio.run(self.backend.probe_capabilities())
        self.assertTrue(caps.supports_choice)
        for mode, kind, status in (("400", "client_error", 400), ("429", "rate_limited", 429), ("500", "server_error", 500)):
            SystemOneUpstream.mode = mode
            with self.assertRaises(VLLMSystemOneError) as caught:
                asyncio.run(self.backend.execute({"state": "x", "questions": {"q": {}}}))
            self.assertEqual((caught.exception.kind, caught.exception.status), (kind, status))

    def test_timeout_and_incompatible_response(self):
        SystemOneUpstream.mode = "timeout"
        with self.assertRaisesRegex(VLLMSystemOneError, "timed out"):
            asyncio.run(self.backend.execute({"state": "x", "questions": {"q": {}}}))
        SystemOneUpstream.response = {"model": "model", "answers": {"q": {"type": "noul", "noul": .5, "probabilities": {"true": 1}}}, "usage": {}}
        with self.assertRaises(VLLMSystemOneError):
            asyncio.run(self.backend.execute({"state": "x", "questions": {"q": {}}}))
        SystemOneUpstream.mode = "probe_bad"
        with self.assertRaisesRegex(VLLMSystemOneError, "incompatible"):
            asyncio.run(self.backend.probe_capabilities())


if __name__ == "__main__":
    unittest.main()
