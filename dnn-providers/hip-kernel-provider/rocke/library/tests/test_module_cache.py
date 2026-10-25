# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Admission policy of benchmarks.common.module_cache, with HIP stubbed out.

What needs a GPU is loading the code object; what needs testing is *when* the
cache loads one -- which lookups reuse a module, which key separates two, and
what the byte budget does once it is reached. Those are host decisions, so
``KernelLauncher`` is replaced by a fake that counts its constructions.
"""

from __future__ import annotations

import pytest

from benchmarks.common.module_cache import ModuleCache

_SIG = [{"name": "A", "type": "ptr<f16, global>"}, {"name": "n", "type": "i32"}]
_OTHER_SIG = [{"name": "A", "type": "ptr<bf16, global>"}, {"name": "n", "type": "i32"}]


class _FakeLauncher:
    """Counts loads, and records whether it was unloaded."""

    loads = 0

    def __init__(self, *, hsaco, kernel_name, signature):
        type(self).loads += 1
        self.hsaco = hsaco
        self.kernel_name = kernel_name
        self.signature = signature
        self.unloaded = False

    def unload(self):
        self.unloaded = True


@pytest.fixture
def fake_hip(monkeypatch):
    """``ModuleCache`` builds launchers through this instead of HIP."""
    import rocke.runtime
    import rocke.runtime.launcher

    _FakeLauncher.loads = 0
    monkeypatch.setattr(rocke.runtime.launcher, "KernelLauncher", _FakeLauncher)
    monkeypatch.setattr(
        rocke.runtime, "synchronize_and_release", lambda *a, **k: None, raising=False
    )
    return _FakeLauncher


@pytest.fixture
def blobs(tmp_path):
    """Three distinct binaries of 1 kB each."""
    out = []
    for i in range(3):
        path = tmp_path / f"k{i}.hsaco"
        path.write_bytes(bytes([i]) * 1024)
        out.append(path)
    return out


def test_a_binary_is_loaded_once_however_many_shapes_ask(fake_hip, blobs):
    cache = ModuleCache()
    first = cache.launcher(hsaco_path=blobs[0], kernel_name="k", signature=_SIG)
    for _ in range(10):
        assert (
            cache.launcher(hsaco_path=blobs[0], kernel_name="k", signature=_SIG)
            is first
        )
    assert fake_hip.loads == 1
    assert (cache.hits, cache.misses) == (10, 1)


@pytest.mark.parametrize(
    "second",
    [
        pytest.param({"kernel_name": "other"}, id="entry-point"),
        pytest.param({"signature": _OTHER_SIG}, id="signature"),
    ],
)
def test_one_blob_two_entry_points_are_two_launchers(fake_hip, blobs, second):
    """The launcher bakes the entry point and the packer, so neither may be
    shared across a key that differs in them."""
    cache = ModuleCache()
    call = dict(hsaco_path=blobs[0], kernel_name="k", signature=_SIG)
    a = cache.launcher(**call)
    b = cache.launcher(**{**call, **second})
    assert a is not b
    assert fake_hip.loads == 2


def test_distinct_binaries_are_distinct_entries(fake_hip, blobs):
    cache = ModuleCache()
    held = [
        cache.launcher(hsaco_path=p, kernel_name="k", signature=_SIG) for p in blobs
    ]
    assert len({id(x) for x in held}) == 3
    assert len(cache) == 3
    assert cache.nbytes == 3 * 1024


def test_the_budget_stops_admitting_rather_than_evicting(fake_hip, blobs):
    """Every case walks the same candidate list in the same order, so evicting
    the entry the next case will ask for is the one policy guaranteed to give
    no hits. A full cache keeps what it has."""
    cache = ModuleCache(2 * 1024)
    first_pass = [
        cache.launcher(hsaco_path=p, kernel_name="k", signature=_SIG) for p in blobs
    ]
    assert len(cache) == 2
    second_pass = [
        cache.launcher(hsaco_path=p, kernel_name="k", signature=_SIG) for p in blobs
    ]
    # The two that fit are reused; the third is loaded again, as it would be
    # with no cache at all.
    assert second_pass[0] is first_pass[0]
    assert second_pass[1] is first_pass[1]
    assert second_pass[2] is not first_pass[2]
    assert fake_hip.loads == 4
    assert "budget" in cache.summary()


def test_a_budget_of_zero_loads_per_use(fake_hip, blobs):
    cache = ModuleCache(0)
    a = cache.launcher(hsaco_path=blobs[0], kernel_name="k", signature=_SIG)
    b = cache.launcher(hsaco_path=blobs[0], kernel_name="k", signature=_SIG)
    assert a is not b
    assert fake_hip.loads == 2
    assert len(cache) == 0


def test_clear_unloads_every_module(fake_hip, blobs, monkeypatch):
    drained = []
    import rocke.runtime

    monkeypatch.setattr(
        rocke.runtime,
        "synchronize_and_release",
        lambda *a, **k: drained.append(True),
        raising=False,
    )
    cache = ModuleCache()
    held = [
        cache.launcher(hsaco_path=p, kernel_name="k", signature=_SIG) for p in blobs
    ]
    cache.clear()
    assert drained, "a module must not be unloaded with launches possibly in flight"
    assert all(launcher.unloaded for launcher in held)
    assert len(cache) == 0 and cache.nbytes == 0


def test_an_unhashable_signature_is_not_a_crash(fake_hip, blobs):
    """A signature the key cannot be built from loads per use instead of
    raising: the cache is an optimisation, not a dependency."""
    cache = ModuleCache()
    weird = [{"name": "A", "shape": [1, 2]}]
    a = cache.launcher(hsaco_path=blobs[0], kernel_name="k", signature=weird)
    b = cache.launcher(hsaco_path=blobs[0], kernel_name="k", signature=weird)
    assert a is not b
    assert len(cache) == 0
