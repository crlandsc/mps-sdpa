"""Lazy-builds and loads the mpsgraph zero-copy extension.

First call compiles mpsgraph_zc.mm via torch.utils.cpp_extension.load(). The
binary is cached under $TORCH_EXTENSIONS_DIR (typically ~/.cache/torch_extensions).
Subsequent calls reuse the cache.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_ext = None
_load_error: Optional[str] = None


def _load():
    global _ext, _load_error
    if _ext is not None:
        return _ext
    if _load_error is not None:
        return None
    try:
        from torch.utils import cpp_extension
    except Exception as e:
        _load_error = f"torch cpp_extension unavailable: {e}"
        return None

    src = Path(__file__).parent / "mpsgraph_zc.mm"
    if not src.exists():
        _load_error = f"source not found: {src}"
        return None

    try:
        _ext = cpp_extension.load(
            name="mps_sdpa_zc_ext",
            sources=[str(src)],
            extra_cflags=[
                # Don't force -std=c++17: torch headers require c++20 (uses
                # std::unordered_map::contains). Torch's build passes c++20
                # already; we just inherit.
                "-ObjC++",
                "-fno-objc-arc",
                "-Wno-unused-function",
                "-Wno-unused-variable",
                "-Wno-deprecated-declarations",
            ],
            extra_ldflags=[
                "-framework", "Metal",
                "-framework", "MetalPerformanceShaders",
                "-framework", "MetalPerformanceShadersGraph",
                "-framework", "Foundation",
            ],
            verbose=False,
        )
    except Exception as e:
        _load_error = f"build failed: {type(e).__name__}: {e}"
        _ext = None
    return _ext


def get_ext():
    return _load()


def load_error() -> Optional[str]:
    return _load_error
