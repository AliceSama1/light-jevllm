"""Validate media bytes without decoding; probes may supply dimensions/frame count."""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Mapping

ALLOWED_MIME_TYPES = frozenset({
    "image/png", "image/jpeg", "image/webp",
    "video/mp4", "video/webm", "video/quicktime", "video/x-matroska",
    "video/mpeg", "video/ogg",
})

@dataclass(frozen=True)
class MediaLimits:
    max_bytes: int = 50 * 1024 * 1024
    max_pixels: int = 100_000_000
    max_frames: int = 300
    def __post_init__(self) -> None:
        if any(type(v) is not int or v <= 0 for v in (self.max_bytes, self.max_pixels, self.max_frames)):
            raise ValueError("media limits must be positive integers")

@dataclass(frozen=True)
class MediaMetadata:
    mime_type: str
    size_bytes: int
    digest: str
    width: int | None = None
    height: int | None = None
    frames: int | None = None

@dataclass(frozen=True)
class NormalizedMedia:
    data: bytes
    metadata: MediaMetadata

Probe = Callable[[bytes, str], Mapping[str, Any]]

class MediaNormalizer:
    def __init__(self, limits: MediaLimits | None = None, probe: Probe | None = None):
        self.limits = limits or MediaLimits()
        self.probe = probe

    def normalize(self, data: bytes | bytearray | memoryview, mime_type: str) -> NormalizedMedia:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("media data must be bytes-like")
        if not isinstance(mime_type, str) or mime_type.lower().strip() not in ALLOWED_MIME_TYPES:
            raise ValueError("unsupported media MIME type")
        payload = bytes(data)
        if len(payload) > self.limits.max_bytes:
            raise ValueError("media exceeds byte limit")
        mime = mime_type.lower().strip()
        info: Mapping[str, Any] = self.probe(payload, mime) if self.probe else {}
        width, height, frames = info.get("width"), info.get("height"), info.get("frames")
        for value, name in ((width, "width"), (height, "height"), (frames, "frames")):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer")
        if width is not None and height is not None and width * height > self.limits.max_pixels:
            raise ValueError("media exceeds pixel limit")
        if frames is not None and frames > self.limits.max_frames:
            raise ValueError("media exceeds frame limit")
        digest = hashlib.sha256(payload).hexdigest()
        return NormalizedMedia(payload, MediaMetadata(mime, len(payload), digest, width, height, frames))

def normalize_media(data: bytes | bytearray | memoryview, mime_type: str, *, limits: MediaLimits | None = None, probe: Probe | None = None) -> NormalizedMedia:
    return MediaNormalizer(limits, probe).normalize(data, mime_type)

__all__ = ["ALLOWED_MIME_TYPES", "MediaLimits", "MediaMetadata", "NormalizedMedia", "MediaNormalizer", "normalize_media"]
