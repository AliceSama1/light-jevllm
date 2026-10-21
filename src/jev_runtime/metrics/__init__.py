"""Thread-safe metrics with dependency-free Prometheus exposition."""
from __future__ import annotations
import math
import re
from threading import RLock
from typing import Mapping, Protocol, runtime_checkable

@runtime_checkable
class MetricsSink(Protocol):
    def increment(self, name: str, value: float = 1, labels: Mapping[str, str] | None = None) -> None: ...
    def gauge(self, name: str, value: float, labels: Mapping[str, str] | None = None) -> None: ...

class MemoryMetricsSink:
    def __init__(self):
        self._values: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._kinds: dict[str, str] = {}
        self._lock = RLock()

    def _record(self, name, value, labels, kind):
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z_:][a-zA-Z0-9_:]*", name):
            raise ValueError("invalid metric name")
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("metric value must be finite")
        if kind == "counter" and value < 0:
            raise ValueError("counter increments must be non-negative")
        labels = dict(labels or {})
        for key, text in labels.items():
            if not isinstance(key, str) or not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", key) or key.startswith("__"):
                raise ValueError("invalid label name")
            if not isinstance(text, str): raise ValueError("label values must be strings")
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            if name in self._kinds and self._kinds[name] != kind:
                raise ValueError("metric type cannot change")
            self._kinds[name] = kind
            self._values[key] = self._values.get(key, 0) + value if kind == "counter" else value

    def increment(self, name: str, value: float = 1, labels: Mapping[str, str] | None = None) -> None:
        self._record(name, value, labels, "counter")

    def gauge(self, name: str, value: float, labels: Mapping[str, str] | None = None) -> None:
        self._record(name, value, labels, "gauge")

    def snapshot(self):
        with self._lock: return dict(self._values)

class PrometheusTextSink(MemoryMetricsSink):
    @staticmethod
    def _escape(value: str) -> str:
        return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')

    def render(self) -> str:
        with self._lock:
            values, kinds = dict(self._values), dict(self._kinds)
        lines = []
        previous = None
        for (name, labels), value in sorted(values.items()):
            if name != previous:
                lines.append(f"# TYPE {name} {kinds[name]}")
                previous = name
            suffix = "{" + ",".join(f'{k}="{self._escape(v)}"' for k, v in labels) + "}" if labels else ""
            lines.append(f"{name}{suffix} {value:g}")
        return "\n".join(lines) + ("\n" if lines else "")
