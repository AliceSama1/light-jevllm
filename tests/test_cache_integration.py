import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

from jev_runtime.service import Runtime


class CacheUpstream(BaseHTTPRequestHandler):
    calls = 0

    def do_POST(self):  # noqa: N802
        CacheUpstream.calls += 1
        length = int(self.headers["Content-Length"])
        self.rfile.read(length)
        body = json.dumps({
            "choices": [{"index": 0, "logprobs": {"top_logprobs": [{"yes": 0.0, "no": -1.0}]}}]
        }).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except OSError:
            pass

    def do_GET(self):  # noqa: N802
        body = b'{"data":[{"id":"upstream"}]}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


class CacheIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        CacheUpstream.calls = 0
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), CacheUpstream)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_result_cache_requires_identity_and_hits_without_upstream(self):
        runtime = Runtime({
            "model": "demo", "upstream_model": "upstream",
            "upstream_url": f"http://127.0.0.1:{self.server.server_port}",
            "workers": 1, "queue_capacity": 2,
            "result_cache": True, "deterministic_results": True,
            "model_revision": "rev1", "tokenizer_identity": "tok1",
        }, encode=lambda text: [1] if text == "yes" else [2])
        request = {"state": "x", "questions": {
            "q": {"type": "choice", "instructions": "pick", "criteria": {"yes": "Y", "no": "N"}}
        }}
        before = CacheUpstream.calls
        try:
            first = runtime.decide(request)
            second = runtime.decide(request)
        finally:
            runtime.close()
        self.assertEqual(first, second)
        self.assertEqual(CacheUpstream.calls - before, 1)


if __name__ == "__main__":
    unittest.main()
