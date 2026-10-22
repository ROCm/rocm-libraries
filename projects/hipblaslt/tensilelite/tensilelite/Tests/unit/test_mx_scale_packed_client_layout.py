# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The benchmark client must allocate the MX scale layout its kernels read.

A micro-scaled GEMM carries one scale byte per 32 elements along the summation
dimension, so one row of scales needs ceil(K/32) bytes. Tensile's own generated
gfx950 kernels read a row pitch rounded up to a multiple of 8; an externally
captured kernel (every rocRoller one, for instance) reads the unrounded pitch,
which is what the hipBLASLt public API specifies and what a production caller
allocates.

The benchmark client used to allocate the padded layout for everything and hand
that buffer to a rocRoller kernel, which worked only because it also handed over
the matching padded stride -- a pairing that never occurs in production. These
tests lock in the replacement: the decision is made from the kernels in the run,
and it reaches the client over both the fresh and the cached config-writing path.
"""

import inspect
import os

from Tensile.BenchmarkProblems import _benchmarkProblemType, _readCacheIfValid
from Tensile.ClientWriter import writeClientConfigIni
from Tensile.CustomKernels import (
    customKernelNameOf,
    isExternalCustomKernel,
    mxScalesArePackedForSolutions,
)

_KERNEL_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "CustomKernels",
)


def _firstKernelIn(subdir):
    names = sorted(
        f[:-2] for f in os.listdir(os.path.join(_KERNEL_ROOT, subdir)) if f.endswith(".s")
    )
    assert names, f"no custom kernels under CustomKernels/{subdir}"
    return names[0]


def test_rocroller_kernels_are_external_captures():
    """Provenance, not the kernel name, is what marks a kernel external.

    Every rocRoller kernel declares Source.Origin in its embedded custom.config.
    That block is the same signal validateCustomKernelMetadata uses to decide
    which fields a kernel must carry, so keying the scale layout off it reuses an
    already-enforced property instead of pattern-matching on names.
    """
    assert isExternalCustomKernel(_firstKernelIn("rocroller")) is True


def test_tensile_generated_kernels_are_not_external_captures():
    assert isExternalCustomKernel(_firstKernelIn("tensile")) is False
    assert isExternalCustomKernel("") is False
    assert isExternalCustomKernel("no-such-kernel-anywhere") is False


def test_packed_only_when_every_solution_is_an_external_capture():
    """One scale buffer is shared by every kernel the run measures.

    So the unpadded layout is safe only if nothing in the run needs the padding.
    A single Tensile-generated kernel has to pull the whole run back to padded.
    """
    rocroller = {"CustomKernel": {"name": _firstKernelIn("rocroller")}}
    tensile = {"CustomKernel": {"name": _firstKernelIn("tensile")}}
    generated = {"SolutionIndex": 0}

    assert mxScalesArePackedForSolutions([rocroller]) is True
    assert mxScalesArePackedForSolutions([rocroller, rocroller]) is True

    assert mxScalesArePackedForSolutions([rocroller, tensile]) is False
    assert mxScalesArePackedForSolutions([rocroller, generated]) is False
    assert mxScalesArePackedForSolutions([]) is False
    assert mxScalesArePackedForSolutions(None) is False


def test_customKernelNameOf_reads_both_shapes():
    name = _firstKernelIn("rocroller")
    assert customKernelNameOf({"CustomKernel": {"name": name}}) == name
    assert customKernelNameOf({"CustomKernelName": name}) == name
    # A CustomKernel block Tensile synthesized for one of its own kernels is not
    # a hand-written custom kernel and must not be mistaken for one.
    assert customKernelNameOf({"CustomKernel": {"name": name, "generated": True}}) == ""
    assert customKernelNameOf({}) == ""


def test_writeClientConfigIni_accepts_and_emits_the_layout_decision():
    """The config the client reads must carry the decision, not re-derive it.

    The client cannot see the solution list, so Python has to tell it. Guarding
    the parameter and the emitted key together stops a refactor from keeping the
    argument while quietly dropping the line the client parses.
    """
    sig = inspect.signature(writeClientConfigIni)
    assert "mxScalesPacked" in sig.parameters
    assert sig.parameters["mxScalesPacked"].default is False, (
        "padded must stay the default so a caller that has not thought about "
        "scale layout keeps the layout Tensile's generated kernels require"
    )

    src = inspect.getsource(writeClientConfigIni)
    assert "param('mx-scales-packed', mxScalesPacked)" in src


def test_cached_benchmark_path_passes_the_layout_decision():
    """The run phase rewrites ClientParameters.ini from cached kernels.

    It has no solution list to re-derive the layout from, so the value is stored
    with the cache and replayed. Without this the build phase would allocate
    packed and the run phase -- the one that actually benchmarks -- would fall
    back to padded, silently restoring the pairing this change removes.
    """
    src = inspect.getsource(_benchmarkProblemType)

    assert '"MXScalesPacked": mxScalesArePackedForSolutions(solutions)' in src, (
        "the decision must be written into cache.yaml alongside the kernels"
    )
    assert 'cacheEntry.get("MXScalesPacked", False)' in src, (
        "the cached path must read the decision back out of cache.yaml"
    )
    assert "mxScalesPacked=cachedMXScalesPacked" in src, (
        "the cached path must forward the decision to writeClientConfigIni"
    )


def test_cache_reader_surfaces_the_layout_decision(tmp_path):
    """The cache reader hands back a filtered dict, not the raw cache.yaml.

    Writing MXScalesPacked into cache.yaml is useless if the reader drops it on
    the way out -- the run phase then falls back to padded while believing it
    replayed the build phase's choice. This is exactly how the first cut of this
    change failed, so it gets its own test.
    """
    import Tensile.LibraryIO as LibraryIO
    from Tensile.BenchmarkProblems import _CACHE_FIELDS, _KERNEL_NAMING_VERSION

    class _Step:
        pass

    step = _Step()
    cacheData = {"KernelNamingVersion": _KERNEL_NAMING_VERSION,
                 "CodeObjectFiles": ["a.co"],
                 "LibraryFile": "lib.yaml"}
    for field, attr in _CACHE_FIELDS.items():
        setattr(step, attr, [])
        cacheData[field] = []

    packedPath = str(tmp_path / "packed.yaml")
    LibraryIO.writeYAML(packedPath, {**cacheData, "MXScalesPacked": True})
    assert _readCacheIfValid(packedPath, step, "{path}")["MXScalesPacked"] is True

    # A cache written before this key existed predates the packed layout.
    legacyPath = str(tmp_path / "legacy.yaml")
    LibraryIO.writeYAML(legacyPath, cacheData)
    assert _readCacheIfValid(legacyPath, step, "{path}")["MXScalesPacked"] is False


def test_packed_and_preswizzled_scales_are_refused_together():
    """The client must reject the two contradictory scale declarations.

    ``mx-scales-packed`` says the scale buffer is a row-major array with no
    slack. ``mx-scale-format 1`` says it went through the gfx950 permutation,
    which is not a row-major array and has no row pitch. Both reach
    ``ClientProblemFactory`` as independent options, so a caller can ask for
    both.

    Letting either win silently produces a false pass rather than an error:
    every externally captured kernel is NoSwizzle, and the upload path keys off
    the selected kernel's format rather than the option, so the run uploads
    canonical scales and reports success for a pre-swizzle test that permuted
    nothing. Measured with the check removed, that is exactly what happens.
    """
    factory = os.path.join(
        os.path.dirname(os.path.dirname(_KERNEL_ROOT)),
        "client", "src", "ClientProblemFactory.cpp",
    )
    with open(factory, encoding="utf-8") as f:
        src = f.read()

    assert "if(m_mxScalesPacked && m_mxScaleHostPreSwizzle)" in src, (
        "the two scale-layout declarations must be checked against each other"
    )
    guard = src.split("if(m_mxScalesPacked && m_mxScaleHostPreSwizzle)", 1)[1][:400]
    assert "throw std::invalid_argument" in guard, (
        "the contradictory pair must be refused, not resolved by precedence"
    )
    for option in ("mx-scales-packed", "mx-scale-format"):
        assert option in guard, f"the error must name {option}"
