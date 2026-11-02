"""Copy-isolated bounded TTL cache with ordered, hashed identity."""
import copy
import hashlib
import json
import pickle
import time
from .lru import SizeAwareLRU

def identity(value):
    # No sorting: candidate/question order affects rendering and tie breaking.
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()).hexdigest()

class TTLCache:
    def __init__(self, entries, size, ttl):
        self.store = SizeAwareLRU(entries, size, size_of=lambda v: len(pickle.dumps(v)))
        self.ttl = ttl

    def get(self, key):
        value = self.store.get(key)
        if value is None: return None
        expires, result = value
        if expires <= time.monotonic():
            self.store.pop(key)
            return None
        return copy.deepcopy(result)

    def put(self, key, value):
        self.store.put(key, (time.monotonic()+self.ttl, copy.deepcopy(value)))

    def clear(self): self.store.clear()
