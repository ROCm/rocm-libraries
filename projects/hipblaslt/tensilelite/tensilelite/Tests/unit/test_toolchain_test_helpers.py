# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Contracts for shared capability data and parallel cache priming."""

import multiprocessing
import pickle
from types import SimpleNamespace

import pytest
import rocisa

import toolchain_test_helpers as helpers
from tensilelite.Common.Architectures import SUPPORTED_ISA


def test_full_map_is_shared_with_single_isa_helpers(isa_info_map, monkeypatch):
    assert isa_info_map is helpers.shared_full_isa_info_map()
    assert list(isa_info_map) == SUPPORTED_ISA
    for isa, info in isa_info_map.items():
        assert info is helpers.shared_isa_info_map(isa)[isa]
        with pytest.raises(TypeError, match="read-only"):
            info.asmCaps["SupportedISA"] = False
        with pytest.raises(TypeError, match="read-only"):
            info.archCaps = {}
    with pytest.raises(TypeError, match="read-only"):
        isa_info_map.clear()

    def unexpected_pool(*args):
        pytest.fail("native rocisa cache keys must be recognized as already primed")

    monkeypatch.setattr(multiprocessing, "get_context", unexpected_pool)
    helpers._prime_isa_cache(SUPPORTED_ISA, "/compiler")


@pytest.mark.parametrize(
    "copy_map", [helpers.copy_isa_info_map, lambda iim: pickle.loads(pickle.dumps(iim))]
)
def test_copy_is_mutable_without_changing_shared_map(isa_info_map, copy_map):
    copied = copy_map(isa_info_map)
    isa = SUPPORTED_ISA[0]
    original = dict(isa_info_map[isa].asmCaps)
    copied[isa].asmCaps["SupportedISA"] = "changed"
    copied[isa].archCaps = {}
    copied.pop(isa)
    assert dict(isa_info_map[isa].asmCaps) == original
    assert isa in isa_info_map


def test_parallel_prime_preserves_existing_and_unlisted_isas(monkeypatch):
    cached_isa, missing_isa, other_isa = (9, 4, 2), (9, 5, 0), (99, 0, 0)
    existing, other, probed = object(), object(), object()
    cache = {cached_isa: existing, other_isa: other}
    kernel = object()

    def replace_cache(data):
        cache.clear()
        cache.update(data)

    ri = SimpleNamespace(
        getData=lambda: dict(cache),
        setData=replace_cache,
        getKernel=lambda: kernel,
    )
    monkeypatch.setattr(rocisa.rocIsa, "getInstance", lambda: ri)
    monkeypatch.setattr(helpers.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(multiprocessing, "current_process", lambda: SimpleNamespace(daemon=False))

    # Two missing ISAs select the parallel branch. Existing entries must never
    # be sent to a child or discarded when its result is installed.
    second_missing = (11, 0, 0)

    class Pool:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def map(self, function, arguments):
            requested = [isa for chunk, compiler in arguments for isa in chunk]
            assert requested == [missing_isa, second_missing]
            return [{missing_isa: probed}, {second_missing: probed}]

    monkeypatch.setattr(
        multiprocessing, "get_context", lambda method: SimpleNamespace(Pool=lambda n: Pool())
    )
    helpers._prime_isa_cache([cached_isa, missing_isa, second_missing], "/compiler")
    assert cache == {
        cached_isa: existing,
        other_isa: other,
        missing_isa: probed,
        second_missing: probed,
    }
    assert ri.getKernel() is kernel


def test_warm_prime_does_not_start_processes(monkeypatch):
    isa = (9, 4, 2)
    ri = SimpleNamespace(getData=lambda: {isa: object()})
    monkeypatch.setattr(rocisa.rocIsa, "getInstance", lambda: ri)

    def unexpected_pool(*args):
        pytest.fail("a warm cache must not start probe processes")

    monkeypatch.setattr(multiprocessing, "get_context", unexpected_pool)
    helpers._prime_isa_cache([isa], "/compiler")


def test_daemon_prime_uses_serial_init(monkeypatch):
    calls = []
    ri = SimpleNamespace(getData=lambda: {}, init=lambda *args: calls.append(args))
    monkeypatch.setattr(rocisa.rocIsa, "getInstance", lambda: ri)
    monkeypatch.setattr(multiprocessing, "current_process", lambda: SimpleNamespace(daemon=True))
    helpers._prime_isa_cache([(9, 4, 2), (9, 5, 0)], "/compiler")
    assert calls == [((9, 4, 2), "/compiler", False), ((9, 5, 0), "/compiler", False)]
