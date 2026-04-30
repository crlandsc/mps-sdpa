"""Concurrency tests for the graph caches.

Threat model: the `if key not in d: d[key] = build()` compound pattern races
under threading — two threads can both miss, both build, and the later store
wins. Graph construction has Metal side effects, so we want to serialize cache
misses. These tests verify the lock-around-build pattern, not concurrent kernel
execution (Metal command queues aren't thread-safe; callers are expected to
serialize kernel dispatch).
"""
from __future__ import annotations

import threading

import pytest
import torch

from mps_sdpa.backends import mpsgraph as _mg


def _can_run() -> bool:
    try:
        return _mg._AVAILABLE and torch.backends.mps.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _can_run(), reason="requires MPSGraph backend + MPS device")
def test_concurrent_cache_single_build_per_key():
    """Many threads racing on the same cache key produce exactly one entry."""
    _mg._graph_cache.clear()
    _mg._init_runtime()

    # Instrument _build_graph_unlocked to count how many times it's called per key.
    call_count = {"n": 0}
    real = _mg._build_graph_unlocked

    def counting_build(key, *args, **kw):
        call_count["n"] += 1
        return real(key, *args, **kw)

    with _mg._graph_cache_lock:
        # Pre-acquire lock to line up all workers at the gate.
        threads = []
        barrier = threading.Barrier(8)
        errors: list[BaseException] = []

        def worker():
            try:
                barrier.wait()
                # All threads call _build_graph for the same key.
                _mg._build_graph(1, 8, 1024, 1024, 64,
                                 _mg._MPS_DTYPE[torch.bfloat16], "none", dropout=False)
            except BaseException as e:
                errors.append(e)

        orig = _mg._build_graph_unlocked
        _mg._build_graph_unlocked = counting_build
        try:
            for _ in range(8):
                t = threading.Thread(target=worker)
                threads.append(t)
                t.start()
        finally:
            # lock released here by context manager
            pass
    # now let threads race
    for t in threads:
        t.join(timeout=10)
    _mg._build_graph_unlocked = orig

    assert not errors, f"workers raised: {errors}"
    # With the lock, we should build exactly once despite 8 racing callers.
    assert call_count["n"] == 1, (
        f"expected 1 build for shared key, got {call_count['n']} — cache races!"
    )
    # And the key is now present in the cache.
    # Key format: (dtype_val, B, H, Lq, Lkv, D, mask_kind, mask_shape, dropout)
    key = (_mg._MPS_DTYPE[torch.bfloat16], 1, 8, 1024, 1024, 64, "none", None, False)
    assert key in _mg._graph_cache


def test_lock_existence_and_type():
    """Sanity: the locks are actual Lock objects."""
    assert isinstance(_mg._graph_cache_lock, type(threading.Lock()))
    assert isinstance(_mg._bwd_graph_cache_lock, type(threading.Lock()))
    assert isinstance(_mg._runtime_lock, type(threading.Lock()))


@pytest.mark.skipif(not _can_run(), reason="requires MPSGraph backend + MPS device")
def test_init_runtime_reentrant():
    """_init_runtime must be idempotent and thread-safe (no errors, one init)."""
    errors: list[BaseException] = []

    def worker():
        try:
            _mg._init_runtime()
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not errors
    assert _mg._device is not None
    assert _mg._command_queue is not None
