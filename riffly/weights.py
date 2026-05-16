"""Lazy download and caching for pretrained Riffly checkpoints.

Aliases in :data:`WEIGHTS` map to a GitHub Release asset URL and the
sha256 of the file. On first use, ``Riffly("<alias>")`` fetches the asset
into the user's weights cache, verifies the hash, and loads it as a
normal ``.pt`` bundle. Subsequent calls reuse the cached file.

Cache layout (XDG-compliant; override with ``$RIFFLY_HOME``)::

    <cache>/weights/<alias>-<sha8>.pt

The hash-suffixed filename means a corrupted partial download cannot
shadow a verified file, and registry updates land at a new path
automatically.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path
from typing import TypedDict


class _WeightSpec(TypedDict):
    url: str
    sha256: str
    description: str


WEIGHTS: dict[str, _WeightSpec] = {
    "pop": {
        "url": "https://github.com/YannickGibson/riffly/releases/download/weights-v1/pop.pt",
        "sha256": "b78a5d29c082510bc1757d5c7bc15050ab2baeceddc99fd99a419a283647cfb9",
        "description": (
            "ConvVAE 32x32 (latent_dim=128, base_channels=48, dropout=0.1) trained "
            "300 epochs on the Pop subset of the ADL Piano MIDI Dataset (CC BY 4.0). "
            "Validation F1=0.65 at threshold=0.30."
        ),
    },
}


def is_alias(name: str) -> bool:
    """Whether ``name`` is a known checkpoint alias."""
    return name in WEIGHTS


def _cache_root() -> Path:
    if override := os.environ.get("RIFFLY_HOME"):
        return Path(override).expanduser() / "weights"
    if xdg := os.environ.get("XDG_CACHE_HOME"):
        return Path(xdg) / "riffly" / "weights"
    return Path.home() / ".cache" / "riffly" / "weights"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(alias: str) -> str:
    """Return a local file path to the cached checkpoint for ``alias``, downloading if needed.

    Verifies the sha256 every time — a cache hit with the wrong hash is
    treated as a miss and re-downloaded.
    """
    if alias not in WEIGHTS:
        raise KeyError(f"unknown weight alias: {alias!r}. known: {sorted(WEIGHTS)}")
    spec = WEIGHTS[alias]
    expected = spec["sha256"]
    cache = _cache_root()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{alias}-{expected[:8]}.pt"

    if target.exists():
        if _sha256_file(target) == expected:
            return str(target)
        target.unlink()  # stale or corrupted — fetch fresh

    tmp = target.with_suffix(".pt.partial")
    print(f"riffly: downloading {alias!r} from {spec['url']}")
    with urllib.request.urlopen(spec["url"]) as resp, open(tmp, "wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)

    got = _sha256_file(tmp)
    if got != expected:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"sha256 mismatch for {alias!r}: expected {expected}, got {got}. "
            f"The release asset may have been replaced or the download corrupted."
        )
    tmp.rename(target)
    return str(target)
