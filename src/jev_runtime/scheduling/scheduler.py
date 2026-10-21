"""Bounded FIFO request coalescer. Batch keys express compatibility, not identity.

One upstream call at a time; oldest request selects the next lane. Later compatible
requests can join it but cannot delay its window. Cancellation drops delivery;
stdlib HTTP calls already in flight finish at their bounded socket timeout.
"""
import asyncio
import time
import threading
from dataclasses import dataclass
from concurrent.futures import Future, TimeoutError
from ..core.errors import ServiceError

@dataclass(frozen=True)
class BatchKey:
    model: str
    revision: str
    tokenizer: str
    lane: str = 'token_logit'

@dataclass
class Work:
    key: BatchKey
    items: list
    expires: float
    enqueued: float
    future: Future
    cost: int

class RequestScheduler:
    def __init__(self, backend, *, capacity, window_ms, max_rows, max_cost, observe=lambda *a: None):
        self.backend = backend
        self.capacity, self.window = capacity, window_ms / 1000
        self.max_rows, self.max_cost, self.observe = max_rows, max_cost, observe
        self.condition = threading.Condition()
        self.pending = []
        self.closed = False
        self.thread = threading.Thread(target=self._run, name='jev-batcher', daemon=True)
        self.thread.start()

    def submit(self, key, items, expires):
        cost = sum(len(i.prompt.encode('utf-8')) for i in items)
        if len(items) > self.max_rows or cost > self.max_cost:
            raise ServiceError(400, 'request exceeds batch row/cost limit')
        future = Future()
        with self.condition:
            if self.closed: raise ServiceError(503, 'scheduler closing')
            if len(self.pending) >= self.capacity: raise ServiceError(429, 'scheduler full')
            self.pending.append(Work(key, items, expires, time.monotonic(), future, cost))
            self.condition.notify_all()
        return future

    def execute(self, key, items, expires):
        future = self.submit(key, items, expires)
        try: return future.result(timeout=max(0, expires-time.monotonic()))
        except TimeoutError as exc:
            future.cancel()
            with self.condition: self.condition.notify_all()
            raise ServiceError(504, 'deadline exceeded') from exc

    def _finish(self, work, value=None, error=None):
        # set_running_or_notify_cancel atomically claims delivery, not execution.
        if work.future.set_running_or_notify_cancel():
            if error is not None: work.future.set_exception(error)
            elif time.monotonic() >= work.expires: work.future.set_exception(ServiceError(504, 'deadline exceeded'))
            else: work.future.set_result(value)

    def _run(self):
        while True:
            with self.condition:
                while not self.pending and not self.closed: self.condition.wait()
                if self.closed: return
                first = self.pending[0]
                wait = min(first.enqueued+self.window, first.expires)-time.monotonic()
                if wait > 0:
                    self.condition.wait(wait)
                    continue
                batch, remaining, rows, cost = [], [], 0, 0
                for work in self.pending:
                    if work.future.cancelled(): continue
                    if work.expires <= time.monotonic():
                        self._finish(work, error=ServiceError(504, 'deadline exceeded'))
                    elif work.key == first.key and rows+len(work.items) <= self.max_rows and cost+work.cost <= self.max_cost:
                        batch.append(work); rows += len(work.items); cost += work.cost
                    else: remaining.append(work)
                self.pending = remaining
            if not batch: continue
            started = time.monotonic()
            for work in batch: self.observe('queue_wait_seconds_total', started-work.enqueued)
            self.observe('batches_total', 1)
            self.observe('batch_requests_total', len(batch))
            self.observe('batch_rows_total', rows)
            try:
                # A short deadline cannot poison a longer-lived peer's result.
                output = asyncio.run(self.backend.execute([i for w in batch for i in w.items], timeout=max(w.expires for w in batch)-time.monotonic()))
                if len(output) != rows: raise ServiceError(502, 'batch output size mismatch')
                offset = 0
                for work in batch:
                    self._finish(work, output[offset:offset+len(work.items)])
                    offset += len(work.items)
            except Exception as exc:
                for work in batch: self._finish(work, error=exc)
            finally: self.observe('forward_seconds_total', time.monotonic()-started)

    def close(self, timeout=5):
        with self.condition:
            self.closed = True
            for work in self.pending: self._finish(work, error=ServiceError(503, 'scheduler closing'))
            self.pending.clear()
            self.condition.notify_all()
        self.thread.join(timeout)
