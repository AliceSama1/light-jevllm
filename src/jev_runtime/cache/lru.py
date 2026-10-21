"""Thread-safe size-aware LRU cache providers."""
from __future__ import annotations
import sys
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Generic, Hashable, TypeVar, Callable, Any

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")

@dataclass(frozen=True)
class CacheStats:
    hits: int
    misses: int
    evictions: int
    entries: int
    size: int

class SizeAwareLRU(Generic[K, V]):
    def __init__(self, max_entries: int = 1024, max_size: int = 16 * 1024 * 1024, size_of: Callable[[V], int] | None = None):
        if type(max_entries) is not int or type(max_size) is not int or max_entries < 1 or max_size < 1:
            raise ValueError("cache limits must be positive integers")
        self.max_entries, self.max_size = max_entries, max_size
        self.size_of = size_of or self._default_size
        self._data: OrderedDict[K, tuple[V, int]] = OrderedDict()
        self._size = self._hits = self._misses = self._evictions = 0
        self._lock = RLock()
    @staticmethod
    def _default_size(value: Any) -> int:
        return max(1, sys.getsizeof(value))
    def get(self, key: K, default: Any = None) -> V | Any:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self._misses += 1
                return default
            self._data.move_to_end(key); self._hits += 1
            return item[0]
    def put(self, key: K, value: V) -> None:
        weight = self.size_of(value)
        if type(weight) is not int or weight < 0:
            raise ValueError("entry size must be a non-negative integer")
        with self._lock:
            old = self._data.get(key)
            if weight > self.max_size:
                return
            if old is not None:
                self._data.pop(key)
                self._size -= old[1]
            self._data[key] = (value, weight); self._size += weight
            while len(self._data) > self.max_entries or self._size > self.max_size:
                _, (_, removed) = self._data.popitem(last=False); self._size -= removed; self._evictions += 1
    def pop(self, key: K, default: Any = None) -> V | Any:
        with self._lock:
            item = self._data.pop(key, None)
            if item is None: return default
            self._size -= item[1]; return item[0]
    def clear(self) -> None:
        with self._lock: self._data.clear(); self._size = 0
    def stats(self) -> CacheStats:
        with self._lock: return CacheStats(self._hits, self._misses, self._evictions, len(self._data), self._size)
    def __contains__(self, key: object) -> bool:
        with self._lock: return key in self._data

class ResultCache(SizeAwareLRU[K, V]): pass
class TemplateCache(SizeAwareLRU[K, V]): pass
class ContentCache(SizeAwareLRU[K, V]): pass
class FeatureCache(SizeAwareLRU[K, V]): pass
