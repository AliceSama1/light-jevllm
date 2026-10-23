import asyncio, json, math, threading, time, socket
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from ..plugins import Family, Backend, Registry, FAMILIES, BACKENDS
from urllib.request import urlopen
from ..decision_ir import Candidate
from ..token_logit import TokenLogitAdapter
from ..vllm_http import VLLMHTTPBackend, VLLMHTTPOptions, VLLMUpstreamError
from ..vllm_systemone import VLLMSystemOneBackend, VLLMSystemOneOptions, VLLMSystemOneError
from ..config import RuntimeConfig
from ..scheduling import AdmissionPolicy, BatchPolicy
from ..scheduling.scheduler import BatchKey, RequestScheduler
from ..metrics import MetricsSink, PrometheusTextSink
from ..cache.runtime import TTLCache, identity

from .errors import ServiceError, object_fields
from ..families.token_logit import TokenLogitFamily
FAMILIES.register('token_logit',TokenLogitFamily)
BACKENDS.register('vllm_http',VLLMHTTPBackend)
BACKENDS.register('vllm_systemone',VLLMSystemOneBackend)
class Runtime:
    def __init__(self, config, encode=None, metrics_sink: MetricsSink | None = None):
        typed = config if isinstance(config, RuntimeConfig) else RuntimeConfig.from_mapping(config)
        config = typed.as_dict()
        self.config = typed
        self.host=typed.host; self.port=typed.port
        self.workers=typed.workers; queue=typed.queue_capacity
        self.timeout=typed.timeout_seconds; self.close_timeout=typed.close_timeout_seconds
        self.max_request_bytes=typed.max_request_bytes
        if encode is None and typed.backend != 'vllm_systemone':
            from tokenizers import Tokenizer
            from pathlib import Path
            path=Path(config.get('tokenizer_json', ''))
            if not path.is_file(): raise ValueError('tokenizer_json must be an explicit local file')
            tokenizer=Tokenizer.from_file(str(path)); encode=lambda text: tokenizer.encode(text,add_special_tokens=False).ids
        self.model=typed.model
        self.backend_name=typed.backend
        self.family=FAMILIES.create(typed.family,encode, temperature=typed.temperature, calibration_version=typed.calibration_version) if typed.backend != 'vllm_systemone' else None
        if typed.backend == 'vllm_systemone':
            self.backend=BACKENDS.create(typed.backend, VLLMSystemOneOptions(typed.upstream_url, model=typed.upstream_model, timeout_seconds=self.timeout, max_concurrency=self.workers))
        else:
            self.backend=BACKENDS.create(typed.backend,VLLMHTTPOptions(typed.upstream_url,timeout_seconds=self.timeout,model=typed.upstream_model))
        self.admission_policy=AdmissionPolicy(self.workers, queue)
        self.batch_policy=BatchPolicy.reference() if typed.policy_mode == 'reference' else None
        self.metrics_sink=metrics_sink or PrometheusTextSink(); self.running=0; self.queued=0
        self.result_cache = TTLCache(typed.cache_entries, typed.cache_bytes, typed.cache_ttl_seconds) if typed.result_cache else None
        self.scheduler = None
        if typed.policy_mode == 'coalesced' and typed.backend == 'vllm_http':
            self.scheduler = RequestScheduler(
                self.backend,
                capacity=queue,
                window_ms=typed.batch_window_ms,
                max_rows=typed.batch_max_rows,
                max_cost=typed.batch_max_cost,
                observe=self._observe_scheduler,
            )
        self.slots=threading.BoundedSemaphore(self.workers+queue); self.execution=threading.BoundedSemaphore(self.workers); self.condition=threading.Condition(); self.closed=False; self.active=0; self.counts={'requests':0,'rejected':0,'errors':0}
        for name in self.counts: self.metrics_sink.increment(typed.metrics_namespace+'_'+name+'_total',0)
    def _observe_scheduler(self, name, value):
        self.metrics_sink.increment(self.config.metrics_namespace + '_' + name, value)

    def _cache_key(self, request):
        return identity({
            'model': self.model,
            'revision': self.config.model_revision,
            'tokenizer': self.config.tokenizer_identity,
            'temperature': self.config.temperature,
            'request': request,
        })

    def capabilities(self):
        if self.backend_name == 'vllm_systemone':
            try:
                caps=asyncio.run(self.backend.probe_capabilities(timeout=min(self.timeout, 2))).as_dict()
                caps.update(supports_request_batching=False, supports_images=False, policy_mode=self.config.policy_mode)
                return caps
            except VLLMSystemOneError as exc: raise ServiceError(503,'upstream capabilities unavailable') from exc
        return {'supports_noul': True, 'supports_choice': True, 'supports_score': True, 'confidence_type': 'normalized_entropy', 'supports_request_batching': self.scheduler is not None, 'supports_images': False, 'policy_mode': self.config.policy_mode}

    def ready(self):
        if self.closed:return False
        try:
            if self.backend_name == 'vllm_systemone':
                asyncio.run(self.backend.probe_capabilities(timeout=min(self.timeout,2)))
                return True
            with urlopen(self.backend.options.base_url.rstrip('/')+'/v1/models',timeout=min(self.timeout,2)) as response: data=json.loads(response.read(self.max_request_bytes+1))
            return isinstance(data,dict) and isinstance(data.get('data'),list) and any(isinstance(m,dict) and m.get('id')==self.backend.options.model for m in data['data'])
        except Exception:return False
    def decide(self, request):
        started=time.monotonic()
        with self.condition:
            self.counts['requests']+=1
            self.metrics_sink.increment(self.config.metrics_namespace+'_requests_total')
            if self.closed: raise ServiceError(503,'runtime closing')
            if not self.admission_policy.decide(min(self.active,self.workers), max(0,self.active-self.workers)).admitted or not self.slots.acquire(False): self.counts['rejected']+=1; self.metrics_sink.increment(self.config.metrics_namespace+'_rejected_total'); raise ServiceError(429,'admission full')
            self.active+=1; self.queued+=1
        acquired=False
        try:
            object_fields(request,('model','state','questions','options'),('state','questions'))
            if request.get('model',self.model)!=self.model: raise ValueError('unknown served model')
            self.metrics_sink.increment(self.config.metrics_namespace+'_request_bytes_total', len(json.dumps(request,allow_nan=False).encode()))
            self.metrics_sink.increment(self.config.metrics_namespace+'_request_questions_total', len(request['questions']) if isinstance(request['questions'],dict) else 0)
            options=request.get('options',{}); object_fields(options,('deadline_ms',)); deadline=options.get('deadline_ms',self.timeout*1000)
            if type(deadline) not in (int,float) or not math.isfinite(deadline) or deadline<=0: raise ValueError('deadline_ms must be finite and positive')
            expires=started+min(deadline/1000,self.timeout)
            if self.result_cache is not None:
                cached = self.result_cache.get(self._cache_key(request))
                if cached is not None:
                    self.metrics_sink.increment(self.config.metrics_namespace+'_cache_hits_total')
                    return cached
                self.metrics_sink.increment(self.config.metrics_namespace+'_cache_misses_total')
            if self.backend_name == 'vllm_systemone':
                questions=request['questions']
                if not isinstance(questions,dict) or not 1<=len(questions)<=64: raise ValueError('questions must contain 1..64 named questions')
                for name, question in questions.items():
                    if not isinstance(name,str) or not name.strip() or not isinstance(question,dict) or question.get('type') not in ('choice','noul','score'): raise ValueError('invalid question')
                    object_fields(question,('type','instructions','criteria'),('type','instructions'))
                    instructions=question['instructions']
                    if not isinstance(instructions,(str,dict,list)) or isinstance(instructions,str) and not instructions.strip(): raise ValueError('invalid instructions')
                    criteria=question.get('criteria')
                    if question['type']=='choice' and (not isinstance(criteria,dict) or not 2<=len(criteria)<=255 or any(not isinstance(k,str) or not k.strip() for k in criteria)): raise ValueError('invalid choice criteria')
                    if question['type']=='score' and (not isinstance(criteria,list) or not 2<=len(criteria)<=10): raise ValueError('invalid score criteria')
                    if question['type']=='noul' and 'criteria' in question and (not isinstance(criteria,dict) or set(criteria)!={'true','false'}): raise ValueError('invalid noul criteria')
                acquired=self._acquire(expires)
                if not acquired or time.monotonic()>=expires: raise ServiceError(504,'deadline exceeded')
                payload={k:v for k,v in request.items() if k!='options'}
                payload['model']=self.config.upstream_model
                result=asyncio.run(self.backend.execute(payload, timeout=max(0,expires-time.monotonic())))
                result['model']=self.model
                if time.monotonic()>=expires: raise ServiceError(504,'deadline exceeded')
                if self.result_cache is not None: self.result_cache.put(self._cache_key(request), result)
                return result
            items=self.family.compile(request)
            if self.scheduler is not None:
                key = BatchKey(
                    self.model,
                    self.config.model_revision,
                    self.config.tokenizer_identity,
                    'token_logit',
                )
                output = self.scheduler.execute(key, items, expires)
                if time.monotonic() >= expires: raise ServiceError(504,'deadline exceeded')
            else:
                acquired=self._acquire(expires)
                if not acquired or time.monotonic()>=expires: raise ServiceError(504,'deadline exceeded')
                output=asyncio.run(self.backend.execute(items,timeout=max(0,expires-time.monotonic())))
                if time.monotonic()>=expires: raise ServiceError(504,'deadline exceeded')
            try:
                answers = self.family.readout(items, output)
            except (ValueError, TypeError, KeyError) as exc:
                raise ServiceError(502, 'upstream readout contract invalid') from exc
            result = {'model':self.model,'answers':answers,'usage':{'questions':len(items)}}
            if self.result_cache is not None: self.result_cache.put(self._cache_key(request), result)
            return result
        except (ValueError,TypeError) as exc: raise ServiceError(400,str(exc)) from exc
        except VLLMSystemOneError as exc:
            status = 504 if exc.kind == 'timeout' else 429 if exc.status == 429 else 400 if exc.kind == 'client_error' else 502
            raise ServiceError(status, 'upstream request failed') from exc
        except VLLMUpstreamError as exc: raise ServiceError(504 if 'timed out' in str(exc) else 502,'upstream request failed') from exc
        finally:
            with self.condition:
                if acquired:
                    self.running-=1; self.execution.release()
                else: self.queued-=1
                self.active-=1; self.slots.release(); self.condition.notify_all()
    def _acquire(self, expires):
        started=time.monotonic()
        acquired=self.execution.acquire(timeout=max(0,expires-started))
        with self.condition:
            self.metrics_sink.increment(self.config.metrics_namespace+'_queue_wait_seconds_total', time.monotonic()-started)
            if acquired: self.queued-=1; self.running+=1
        return acquired
    def close(self):
        with self.condition:
            self.closed=True; end=time.monotonic()+self.close_timeout
            while self.active and time.monotonic()<end:self.condition.wait(end-time.monotonic())
        if self.scheduler is not None: self.scheduler.close(self.close_timeout)
        self.backend.close();
        if self.family is not None: self.family.close()
    def metrics(self):
        with self.condition:
            prefix=self.config.metrics_namespace
            for name,value in {'queue_depth':self.queued,'running':self.running,'active':self.active}.items():
                self.metrics_sink.gauge(prefix+'_'+name,value)
            batch_stats = getattr(self.scheduler, 'stats', lambda: {})()
            for name, value in batch_stats.items():
                self.metrics_sink.gauge(prefix+'_'+name, value)
            cache_stats = self.result_cache.store.stats() if self.result_cache is not None else None
            if cache_stats is not None:
                for name in ('entries', 'size', 'evictions', 'hits', 'misses'):
                    self.metrics_sink.gauge(prefix+'_cache_'+name, getattr(cache_stats, name))
            self.metrics_sink.gauge(prefix+'_closing',int(self.closed))
            self.metrics_sink.gauge(prefix+'_policy_mode',1,{'mode':self.config.policy_mode})
            render=getattr(self.metrics_sink,'render',None)
            text=render() if callable(render) else ''
            return text
