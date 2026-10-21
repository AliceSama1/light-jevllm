import json
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from jev_runtime.service import Runtime, Server, Registry


class Upstream(BaseHTTPRequestHandler):
    mode = 'ok'
    seen = []
    entered = threading.Event()
    release = threading.Event()

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.reply(200, {'data': [{'id': 'upstream'}]} if self.mode != 'unready' else {'data': []})

    def reply(self, status, value):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except OSError:
            pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.seen.append(body)
        self.entered.set()
        if self.mode == 'block':
            self.release.wait(2)
        if self.mode == 'timeout':
            time.sleep(.15)
        if self.mode == 'error':
            return self.reply(500, {'error': 'broken'})
        prompts = body['prompt'] if isinstance(body['prompt'], list) else [body['prompt']]
        choices = []
        for i, prompt in enumerate(prompts):
            candidates = json.loads(prompt.split('\nAnswer with')[0])['candidates']
            values = {c['label']: -float(j) for j, c in enumerate(candidates)}
            if self.mode == 'missing':
                values.pop(next(iter(values)))
            choices.append({'index': i, 'logprobs': {'top_logprobs': [values]}})
        self.reply(200, {'choices': choices})


class ServiceTests(unittest.TestCase):
    def setUp(self):
        Upstream.mode = 'ok'
        Upstream.seen = []
        Upstream.entered.clear()
        Upstream.release.clear()
        self.upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
        self.upthread = threading.Thread(target=self.upstream.serve_forever)
        self.upthread.start()
        self.runtime = Runtime({'model': 'demo', 'upstream_model': 'upstream',
            'upstream_url': f'http://127.0.0.1:{self.upstream.server_port}',
            'workers': 1, 'queue_capacity': 0, 'timeout_seconds': .5},
            encode=lambda text: [sum((i+1)*ord(c) for i,c in enumerate(text))])
        self.server = Server(('127.0.0.1', 0), self.runtime)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()

    def tearDown(self):
        Upstream.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.upstream.shutdown()
        self.upstream.server_close()
        self.upthread.join()

    def call(self, path='/v1/decisions', body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = Request(f'http://127.0.0.1:{self.server.server_port}'+path, data=data,
                      headers={'Content-Type': 'application/json'})
        try:
            response = urlopen(req, timeout=3)
        except HTTPError as exc:
            response = exc
        with response:
            text = response.read().decode()
            return response.status, text if path == '/metrics' else json.loads(text)

    def request(self):
        return {'state': {'user': 'actual inbound context'}, 'questions': {
            'route': {'type': 'choice', 'instructions': 'Pick route', 'criteria': {'yes': 'accept', 'no': 'reject'}},
            'rating': {'type': 'score', 'instructions': 'Rate quality', 'criteria': ['low', 'high']},
            'valid': {'type': 'noul', 'instructions': 'Is valid?'}}}

    def test_real_http_e2e_alias_and_prompts(self):
        for endpoint in ('/v1/decisions', '/v1/systemone'):
            status, body = self.call(endpoint, self.request())
            self.assertEqual(status, 200, body)
            answers = body['answers']
            self.assertEqual(answers['route']['choice'], 'yes')
            self.assertAlmostEqual(answers['rating']['score'], 1/(1+__import__('math').e))
            self.assertIs(type(answers['valid']['noul']), float)
            self.assertAlmostEqual(answers['valid']['noul'], answers['valid']['probabilities']['true'])
            self.assertIsNone(answers['valid']['calibration_version'])
        payload = Upstream.seen[-1]
        self.assertEqual(payload['model'], 'upstream')
        self.assertEqual(len(payload['prompt']), 3)
        self.assertIn('actual inbound context', payload['prompt'][0])
        self.assertIn('Pick route', payload['prompt'][0])

    def test_schema_and_token_validation(self):
        invalid = [[], {}, dict(self.request(), extra=True)]
        for value in [True, 0, -1, '1']:
            invalid.append(dict(self.request(), options={'deadline_ms': value}))
        for q in [{'type': 'choice', 'instructions': '', 'criteria': {'a': None, 'b': None}},
                  {'type': 'noul', 'instructions': 'x', 'criteria': {'true': None}},
                  {'type': 'choice', 'instructions': 'x', 'criteria': {'a': None, 'b': None}, 'token_labels': {'a': 'z', 'b': 'z'}}]:
            invalid.append({'state': '', 'questions': {'q': q}})
        for body in invalid:
            self.assertEqual(self.call(body=body)[0], 400, body)
        self.runtime.family.adapter._encode = lambda text: [1, 2]
        self.assertEqual(self.call(body=self.request())[0], 400)
        self.assertEqual(Upstream.seen, [])

    def test_upstream_missing_failure_and_timeout(self):
        for mode, expected in [('missing', 502), ('error', 502), ('timeout', 504)]:
            Upstream.mode = mode
            body = dict(self.request(), options={'deadline_ms': 30})
            self.assertEqual(self.call(body=body)[0], expected)

    def test_admission_readiness_metrics_and_close(self):
        self.assertEqual(self.call('/health')[0], 200)
        Upstream.mode = 'unready'
        self.assertEqual(self.call('/health')[0], 503)
        self.assertEqual(self.call('/health/live')[0], 200)
        self.assertEqual(self.call('/v1/models')[1]['data'][0]['id'], 'demo')
        Upstream.mode = 'block'
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.call, '/v1/decisions', self.request())
            self.assertTrue(Upstream.entered.wait(1))
            self.assertEqual(self.call(body=self.request())[0], 429)
            self.assertIn('jev_active 1', self.call('/metrics')[1])
            close = pool.submit(self.runtime.close)
            self.assertEqual(self.call('/health')[0], 503)
            self.assertEqual(self.call(body=self.request())[0], 503)
            self.assertFalse(close.done())
            Upstream.release.set()
            self.assertEqual(first.result()[0], 200)
            close.result()
        self.assertTrue(self.runtime.backend.closed)
        self.assertTrue(self.runtime.family.closed)
        self.assertIn('jev_rejected_total 1', self.call('/metrics')[1])

    def test_registry(self):
        registry = Registry()
        registry.register('test', lambda: 42)
        self.assertEqual(registry.create('test'), 42)
        with self.assertRaises(ValueError):
            registry.register('test', object)
        with self.assertRaises(ValueError):
            registry.create('unknown')




class NativeServiceTests(ServiceTests):
    # Reuse HTTP helpers, not completion-specific test cases.
    def setUp(self):
        from test_vllm_systemone import SystemOneUpstream
        from jev_runtime.config import RuntimeConfig
        self.handler=SystemOneUpstream
        self.handler.mode='ok'
        self.handler.response={'model':'model','answers':{'q':{'type':'noul','noul':0.8,'probabilities':{'true':0.8,'false':0.2},'confidence':0.75}},'usage':{'questions':1,'tokens':7},'meta':{'native':True}}
        self.upstream=ThreadingHTTPServer(('127.0.0.1',0),self.handler)
        self.upthread=threading.Thread(target=self.upstream.serve_forever); self.upthread.start()
        self.runtime=Runtime(RuntimeConfig('demo',f'http://127.0.0.1:{self.upstream.server_port}','model',backend='vllm_systemone',policy_mode='disabled'))
        self.server=Server(('127.0.0.1',0),self.runtime)
        self.thread=threading.Thread(target=self.server.serve_forever); self.thread.start()

    def request(self):
        return {'model':'demo','state':'context','questions':{'q':{'type':'noul','instructions':'valid?'}}}

    def test_real_http_e2e_alias_and_prompts(self):
        for endpoint in ('/v1/systemone','/v1/decisions'):
            status,body=self.call(endpoint,self.request())
            self.assertEqual(status,200,body)
            self.assertEqual(body['answers'],self.handler.response['answers'])
            self.assertEqual(body['usage'],self.handler.response['usage'])
            self.assertEqual(body['meta'],{'native':True})
            self.assertEqual(body['model'],'demo')
            self.assertEqual(self.handler.seen[0],'/v1/systemone')
            self.assertEqual(self.handler.seen[1]['model'],'model')
            self.assertEqual(self.handler.seen[1]['questions'],self.request()['questions'])
        caps=self.call('/v1/models')[1]['data'][0]['capabilities']
        self.assertEqual(caps['confidence_type'],'upstream')
        self.assertFalse(caps['supports_request_batching'])
        text=self.call('/metrics')[1]
        self.assertIn('jev_request_questions_total 2',text)
        self.assertIn('jev_queue_depth 0',text)
        self.assertEqual(text,self.call('/metrics')[1])

    def test_upstream_missing_failure_and_timeout(self):
        for mode,status in [('400',400),('429',429),('500',502)]:
            self.handler.mode=mode
            self.assertEqual(self.call(body=self.request())[0],status)
        self.handler.mode='ok'
        self.handler.response['answers']['q']['noul']=True
        self.assertEqual(self.call(body=self.request())[0],502)

    def test_schema_and_token_validation(self):
        for request in [{},dict(self.request(),model='wrong'),dict(self.request(),questions={})]:
            self.assertEqual(self.call(body=request)[0],400)

    def test_admission_readiness_metrics_and_close(self):
        self.assertEqual(self.call('/health')[0],200)
        self.handler.mode='probe_bad'
        self.assertEqual(self.call('/v1/models')[0],503)
        self.runtime.close()
        self.assertEqual(self.call(body=self.request())[0],503)

if __name__ == '__main__':
    unittest.main()
