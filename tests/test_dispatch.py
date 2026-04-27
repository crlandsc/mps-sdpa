"""Backend registry + dispatch tests."""
import pytest
import torch
from mps_sdpa import backends


def test_stock_always_available():
    assert "stock" in backends.available_backends()


def test_get_backend_returns_callable():
    fn = backends.get_backend("stock")
    assert callable(fn)


def test_get_backend_unknown_raises():
    with pytest.raises(KeyError):
        backends.get_backend("no_such_backend")


def test_register_backend_adds_to_registry():
    def dummy(q, k, v, **kw):
        return q
    backends.register_backend("dummy_test", dummy, available=True)
    assert "dummy_test" in backends.available_backends()
    assert backends.get_backend("dummy_test") is dummy


def test_register_unavailable_not_in_available_backends():
    def dummy(q, k, v, **kw): return q
    backends.register_backend("dummy_unavailable", dummy, available=False)
    assert "dummy_unavailable" not in backends.available_backends()
    with pytest.raises(RuntimeError, match="not available"):
        backends.get_backend("dummy_unavailable")


def test_register_unavailable_surfaces_reason():
    def dummy(q, k, v, **kw): return q
    backends.register_backend(
        "dummy_with_reason", dummy, available=False, reason="probe failed: macOS 14",
    )
    with pytest.raises(RuntimeError, match="probe failed: macOS 14"):
        backends.get_backend("dummy_with_reason")
    assert backends.backend_reason("dummy_with_reason") == "probe failed: macOS 14"


def test_register_available_clears_reason():
    def dummy(q, k, v, **kw): return q
    backends.register_backend("dummy_reclaim", dummy, available=False, reason="x")
    backends.register_backend("dummy_reclaim", dummy, available=True)
    assert backends.backend_reason("dummy_reclaim") is None


def test_mpsgraph_availability_probe_returns_tuple():
    """Probe function always returns (bool, reason-or-None)."""
    from mps_sdpa.backends import mpsgraph as _mg
    avail, reason = _mg._check_mpsgraph_sdpa_available()
    assert isinstance(avail, bool)
    assert reason is None or isinstance(reason, str)
    if not avail:
        assert reason, "unavailable backends must supply a reason"


def test_mpsgraph_missing_selector_marks_unavailable(monkeypatch):
    """Simulate macOS 14 (method absent): probe must return (False, reason)."""
    from mps_sdpa.backends import mpsgraph as _mg
    if not _mg._AVAILABLE:
        pytest.skip("mpsgraph import failed; selector path unreachable")

    class _FakeGraph:
        pass  # lacks scaledDotProductAttention* attrs

    class _FakeAlloc:
        def init(self):
            return _FakeGraph()

    class _FakeMPSGraph:
        @staticmethod
        def alloc():
            return _FakeAlloc()

    monkeypatch.setattr(_mg, "MPSGraph", _FakeMPSGraph)
    avail, reason = _mg._check_mpsgraph_sdpa_available()
    assert avail is False
    assert reason is not None
    assert "macOS 15" in reason or "missing method" in reason.lower()
