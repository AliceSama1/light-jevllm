import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from jev_runtime.core.runtime import Runtime
from jev_runtime.api.server import Server


class CapabilityOnlyHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):  # noqa: N802
        body = json.dumps({
            "protocol_version": "light-jevllm.parallel.v1",
            "api": "mock",
            "parallel_modes": ["cp"],
            "world_size": 1,
            "phase_split": False,
            "kv_transfer": False,
            "media_inputs": False,
        }).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class NightlyRuntimeTests(unittest.TestCase):
    def test_nightly_backend_is_capability_only_until_family_is_registered(self):
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), CapabilityOnlyHandler)
        thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        thread.start()
        runtime = Runtime({
            "model": "nightly",
            "upstream_model": "nightly-model",
            "upstream_url": f"http://127.0.0.1:{upstream.server_port}",
            "backend": "vllm_nightly",
            "workers": 1,
            "queue_capacity": 1,
        })
        server = Server(("127.0.0.1", 0), runtime)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        try:
            req = Request(
                f"http://127.0.0.1:{server.server_port}/v1/models",
                headers={"Accept": "application/json"},
            )
            with urlopen(req, timeout=2) as response:
                models = json.loads(response.read())
            self.assertTrue(models["data"][0]["capabilities"]["supports_cp"])
            request = Request(
                f"http://127.0.0.1:{server.server_port}/v1/decisions",
                data=json.dumps({"state": "x", "questions": {"q": {"type": "noul", "instructions": "ok?"}}}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with self.assertRaises(HTTPError) as error:
                urlopen(request, timeout=2)
            self.assertEqual(error.exception.code, 501)
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join()
            upstream.shutdown()
            upstream.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
