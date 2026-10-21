import json, socket, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from concurrent.futures import ThreadPoolExecutor
from ..core.errors import ServiceError
def strict_json(data):
    def pairs(entries):
        result={}
        for k,v in entries:
            if k in result: raise ValueError('duplicate JSON key')
            result[k]=v
        return result
    return json.loads(data,object_pairs_hook=pairs,parse_constant=lambda v: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
class Server(HTTPServer):
    allow_reuse_address=True
    def __init__(self,address,runtime):
        self.runtime=runtime; limit=runtime.workers+runtime.slots._initial_value+2; self.pool=ThreadPoolExecutor(max_workers=limit,thread_name_prefix='jev-http'); self.connections=threading.BoundedSemaphore(limit); super().__init__(address,Handler)
    def process_request(self,request,address):
        if not self.connections.acquire(False):
            try: request.settimeout(1); request.sendall(b'HTTP/1.0 429 Too Many Requests\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
            finally:self.shutdown_request(request)
            return
        self.pool.submit(self.process_connection,request,address)
    def process_connection(self,request,address):
        try: request.settimeout(self.runtime.timeout); self.finish_request(request,address)
        except ConnectionError: pass
        finally:self.shutdown_request(request); self.connections.release()
    def server_close(self):
        self.runtime.close(); super().server_close(); self.pool.shutdown(wait=True)
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def send(self,status,body,content_type='application/json'):
        data=body.encode() if isinstance(body,str) else json.dumps(body,allow_nan=False).encode(); self.send_response(status); self.send_header('Content-Type',content_type); self.send_header('Content-Length',str(len(data))); self.end_headers()
        try:self.wfile.write(data)
        except ConnectionError:pass
    def _error(self,status,message):
        try:self.send(status,{'error':{'message':message}})
        except ConnectionError:pass
    def do_GET(self):
        runtime=self.server.runtime
        try:
            if runtime.closed and self.path in ('/health','/v1/models'): return self._error(503,'runtime closing')
            if self.path=='/health/live': self.send(200,{'live':True})
            elif self.path=='/health':
                ready=runtime.ready(); self.send(200 if ready else 503,{'ready':ready})
            elif self.path=='/v1/models': self.send(200,{'object':'list','data':[{'id':runtime.model,'object':'model','family':('systemone' if runtime.backend_name=='vllm_systemone' else runtime.config.family),'backend':runtime.backend_name,'capabilities':runtime.capabilities()}]})
            elif self.path=='/metrics': self.send(200,runtime.metrics(),'text/plain; version=0.0.4')
            else:self._error(404,'not found')
        except ServiceError as exc:self._error(exc.status,exc.message)
        except Exception:self._error(500,'internal server error')
    def do_POST(self):
        runtime=self.server.runtime
        try:
            if self.path not in ('/v1/decisions','/v1/systemone'): raise ServiceError(404,'not found')
            if runtime.closed: raise ServiceError(503,'runtime closing')
            if self.headers.get('Transfer-Encoding') or len(self.headers.get_all('Content-Length',[]))!=1: raise ServiceError(400,'one Content-Length required; transfer encoding unsupported')
            if self.headers.get_content_type()!='application/json': raise ServiceError(400,'application/json required')
            size=int(self.headers['Content-Length'])
            if not 0<size<=runtime.max_request_bytes: raise ServiceError(413,'body size outside allowed range')
            raw=self.rfile.read(size)
            if len(raw)!=size: raise ValueError('incomplete request body')
            self.send(200,runtime.decide(strict_json(raw)))
        except (ValueError,UnicodeError) as exc:self._error(400,str(exc))
        except ServiceError as exc:
            with runtime.condition:
                runtime.counts['errors']+=1
                runtime.metrics_sink.increment(runtime.config.metrics_namespace+'_errors_total')
            self._error(exc.status,exc.message)
        except (TimeoutError,socket.timeout): self._error(408,'request body timeout')
        except Exception:
            with runtime.condition:
                runtime.counts['errors']+=1
                runtime.metrics_sink.increment(runtime.config.metrics_namespace+'_errors_total')
            self._error(500,'internal server error')
