# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Benchmark builds compile the helper kernels (Kernels.cpp) once, xnack-agnostic
(timing_runs/DECISIONS.md topic 6, A).

BenchmarkProblems.cmdLineArchsFor gives the bare architecture, as the library build does from
GPU_TARGETS, instead of both xnack variants; the runtime loads Kernels.so-000-<arch>.hsaco
first. A sanitizer build keeps xnack+ on the architectures that have it, as the library build
does for ASAN."""

from types import SimpleNamespace

import pytest

import Tensile.BenchmarkProblems as BP
import Tensile.Toolchain.Component as Component
from Tensile.Common import SemanticVersion
from Tensile.Common.Architectures import gfxToIsa, gfxVariantMap

pytestmark = pytest.mark.unit

XNACK_ARCHS = sorted(gfxVariantMap)


@pytest.fixture
def compilerFor(monkeypatch):
    """A real Compiler (its default arguments), without probing a compiler's version."""
    monkeypatch.setattr(Component, "_getVersion", lambda *_a: SemanticVersion(22, 0, 0))
    return lambda asan: Component.Compiler("amdclang++", "sha1", asan_build=asan)


def _targets(archs, compiler, archNames=None):
    return BP.cmdLineArchsFor(archNames, {gfxToIsa(a): None for a in archs}, compiler)


def test_xnack_architectures_are_covered():
    """Every architecture that used to get both xnack variants is now built once."""
    assert XNACK_ARCHS == ["gfx906", "gfx908", "gfx90a", "gfx942", "gfx950"]


@pytest.mark.parametrize("arch", XNACK_ARCHS + ["gfx1200", "gfx1250"])
def test_one_xnack_agnostic_target(compilerFor, arch):
    assert _targets([arch], compilerFor(False)) == [arch]


@pytest.mark.parametrize("arch", XNACK_ARCHS)
def test_sanitizer_build_keeps_xnack_on(compilerFor, arch):
    assert _targets([arch], compilerFor(True)) == [f"{arch}:xnack+"]


@pytest.mark.parametrize("arch", ["gfx1200", "gfx1250"])
def test_sanitizer_build_leaves_other_architectures_bare(compilerFor, arch):
    assert _targets([arch], compilerFor(True)) == [arch]


@pytest.mark.parametrize("asan", [False, True])
def test_requested_stepping_still_wins(compilerFor, asan):
    assert _targets(["gfx1250"], compilerFor(asan), ["gfx1250-strict"]) == ["gfx1250-strict"]


@pytest.mark.parametrize("asan, offloadArch", [(False, "gfx950"), (True, "gfx950:xnack+")])
def test_helper_compile_gets_one_offload_arch(compilerFor, monkeypatch, asan, offloadArch):
    """What the helper compile is actually given: one --offload-arch."""
    calls = []
    monkeypatch.setattr(Component, "_invoke", lambda args, desc="": calls.append(args))
    compiler = compilerFor(asan)

    compiler("include", _targets(["gfx950"], compiler), "Kernels.cpp", "Kernels.o")

    (args,) = calls
    assert [a for a in args if a.startswith("--offload-arch")] == [f"--offload-arch={offloadArch}"]


def test_benchmark_cache_key_changes_once(compilerFor):
    """The targets feed the --use-cache key: entries written for both xnack variants are not
    reused for the single target."""
    step = SimpleNamespace(**{attr: {} for attr in BP._CACHE_FIELDS.values()})
    old = BP._computeCacheKey(step, gfxVariantMap["gfx950"])
    assert BP._computeCacheKey(step, _targets(["gfx950"], compilerFor(False))) != old
