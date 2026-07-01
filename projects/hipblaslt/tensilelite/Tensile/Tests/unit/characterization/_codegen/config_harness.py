################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""CPU-only ``BenchmarkProblems`` config -> Solutions -> emit harness.

This is *not* a test module (no ``test_`` prefix, not collected). It exercises
the **config-driven** solution-generation surface that the logic-driven
:mod:`codegen_harness` does not touch:

    Tensile config YAML  ->  BenchmarkProcess (BenchmarkStructs)
                         ->  constructForkPermutations
                         ->  _generateForkedSolutions  ->  Solution(s)
                         ->  generateKernelObjectsFromSolutions  ->  kernel dict(s)
                         ->  processKernelSource  ->  assembly text

A Tensile ``BenchmarkProblems`` entry is a ``[ProblemType, ProblemSizeGroup]``
pair. The ``ForkParameters`` block is a cartesian product of single-element
value lists, so each fork permutation yields exactly one ``Solution`` (CPU-only;
no GPU, no benchmarking, no compile). We then hand the resulting ``Solution``
objects to the *same* emit path :mod:`codegen_harness` uses, so the emitted
assembly is canonicalized and warm-state-stable in exactly the same way.

Unlike a logic file (which pins its own ``ISA``/architecture), a benchmark
config under ``Tests/common`` is arch-agnostic: ``_generate_single_solution``
takes the ISA from ``next(iter(isaInfoMap.keys()))``. So here we build a
*single-arch* ISA-info map for a chosen architecture (default gfx942, which
supports the MFMA ``MatrixInstruction`` shapes the common gemm configs use) and
drive everything through it. Pass ``arch=`` to target another supported gfx.

Usage::

    from config_harness import emit_kernels_from_config
    results = emit_kernels_from_config(CONFIG_PATH)   # [(basename, src, err), ...]

The expensive toolchain build is cached process-wide (per arch).
"""

import contextlib
import copy
import functools

# Reuse the logic-driven harness for: assembler/toolchain construction, the
# canonicalize/warm-state emit, global-state isolation, and per-kernel rocisa
# init. Everything below only adds the *config -> solutions* front end.
import codegen_harness as _ch
from char_paths import resolve_tensile_path


# Default target architecture. gfx942 (IsaVersion(9, 4, 2)) supports the MFMA
# MatrixInstruction shapes used by the common gemm configs, and is a stable
# CPU-emit target. Override via ``arch=`` to characterize another gfx.
_DEFAULT_ARCH = "gfx942"


@functools.lru_cache(maxsize=None)
def _toolchain_for(arch):
    """Build ``(assembler, isaInfoMap)`` for a single ``arch`` (gfx name).

    Mirrors :func:`codegen_harness._toolchain` but restricts the ISA-info map to
    one architecture so ``_generate_single_solution``'s
    ``next(iter(isaInfoMap.keys()))`` deterministically selects it. Uses
    amdclang++; no GPU required. Cached per arch.
    """
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Toolchain.Validators import validateToolchain

    isa = gfxToIsa(arch)
    if isa is None:
        raise ValueError(f"Unrecognized gfx architecture: {arch!r}")
    cxx = validateToolchain("amdclang++")
    iim = makeIsaInfoMap([isa], cxx)
    # The assembler itself is arch-independent; reuse the shared cached build.
    assembler = _ch.get_assembler()
    return assembler, iim


@contextlib.contextmanager
def _isolated_globals_with_isa(isaInfoMap):
    """Isolate process-global parameter state, with ``validParameters["ISA"]``
    populated for the target ISA map.

    ``BenchmarkProcess`` validates fork/common parameters against
    ``validParameters`` (including the ``ISA`` entry that ``assignGlobalParameters``
    fills in). We must set it for our single-arch map *and* restore the prior
    state afterwards so this harness never leaks into unrelated unit tests
    (same contract as ``codegen_harness._isolated_globals``).
    """
    from Tensile.Common.GlobalParameters import globalParameters, assignGlobalParameters
    from Tensile.Common.ValidParameters import validParameters

    saved_gp = copy.deepcopy(dict(globalParameters))
    saved_vp = copy.deepcopy(dict(validParameters))
    try:
        # Populates validParameters["ISA"] and ROCm paths for this map.
        assignGlobalParameters({}, isaInfoMap)
        yield
    finally:
        globalParameters.clear()
        globalParameters.update(saved_gp)
        validParameters.clear()
        validParameters.update(saved_vp)


def _load_config(config_path):
    """Read a Tensile config YAML into a dict (GlobalParameters/BenchmarkProblems)."""
    from Tensile import LibraryIO

    return LibraryIO.read(str(resolve_tensile_path(config_path)))


def _solutions_from_config_unguarded(config_path, assembler, isaInfoMap, limit_solutions=None):
    """Build ``Solution`` objects from a config's first BenchmarkProblems entry.

    Walks the real config-driven path: ``BenchmarkProcess`` parses the
    ProblemType + ProblemSizeGroup, ``constructForkPermutations`` enumerates the
    fork cartesian product, and ``_generateForkedSolutions`` derives one
    ``Solution`` per permutation. CPU-only; nothing is compiled or run.

    ``limit_solutions`` caps the number of fork permutations fed to solution
    generation (keeps the rocisa per-process footprint bounded for big sweeps).
    """
    from Tensile.BenchmarkProblems import _generateForkedSolutions
    from Tensile.BenchmarkStructs import BenchmarkProcess, constructForkPermutations
    from Tensile.Common.Types import makeDebugConfig

    config = _load_config(config_path)
    benchmarkProblems = config["BenchmarkProblems"]
    if not benchmarkProblems:
        return []

    # Each BenchmarkProblems entry is [ProblemTypeConfig, ProblemSizeGroupConfig].
    problemTypeConfig, problemSizeGroupConfig = benchmarkProblems[0][0], benchmarkProblems[0][1]

    debugConfig = makeDebugConfig(config.get("GlobalParameters", {}))

    benchmarkProcess = BenchmarkProcess(problemTypeConfig, problemSizeGroupConfig, False)
    benchmarkStep = benchmarkProcess[0]

    if problemSizeGroupConfig.get("ForkParameters"):
        forkPermutations = constructForkPermutations(benchmarkStep.forkParams, benchmarkStep.paramGroups)
        perms = list(forkPermutations)
    else:
        perms = []

    if limit_solutions is not None:
        perms = perms[:limit_solutions]

    solutions = _generateForkedSolutions(
        benchmarkProcess.problemType,
        benchmarkStep.constantParams,
        perms,
        assembler,
        debugConfig,
        isaInfoMap,
    )
    return solutions


def solutions_from_config(config_path, arch=_DEFAULT_ARCH, limit_solutions=None):
    """Return fully-derived ``Solution`` objects for ``config_path`` (CPU-only).

    Runs under global-state isolation so it does not leak into other tests.
    """
    assembler, iim = _toolchain_for(arch)
    with _isolated_globals_with_isa(iim):
        return _solutions_from_config_unguarded(config_path, assembler, iim, limit_solutions)


def emit_kernels_from_config(config_path, limit=8, arch=_DEFAULT_ARCH, canonical=True, splitGSU=False):
    """Emit assembly for the kernels of a ``BenchmarkProblems`` config.

    Drives ``config -> BenchmarkProcess -> constructForkPermutations ->
    _generateForkedSolutions -> Solution(s)`` then emits each via the *same*
    path :mod:`codegen_harness` uses (``generateKernelObjectsFromSolutions`` +
    ``processKernelSource``), returning ``[(basename, source, err), ...]`` sorted
    by basename.

    ``err`` is the emitter return code (0 == ok). ``limit`` bounds both the
    number of fork permutations turned into solutions *and* the number of
    emitted kernels, so the rocisa per-process footprint stays small.
    """
    import rocisa  # noqa: F401  (ensures the singleton module is importable here)
    from Tensile.TensileCreateLibrary.Run import generateKernelObjectsFromSolutions
    from Tensile.KernelWriterAssembly import KernelWriterAssembly
    from Tensile.Common.Types import DebugConfig
    from Tensile.SolutionStructs.Naming import getKernelFileBase

    assembler, iim = _toolchain_for(arch)

    results = []
    with _isolated_globals_with_isa(iim):
        sols = _solutions_from_config_unguarded(config_path, assembler, iim, limit_solutions=limit)
        kernels = generateKernelObjectsFromSolutions(sols)
        if limit is not None:
            kernels = sorted(kernels, key=lambda k: getKernelFileBase(splitGSU, k))[:limit]
        kwa = KernelWriterAssembly(assembler, DebugConfig())

        # Steady-state warm-up (see codegen_harness for the rationale): the very
        # first emit in a process accumulates scheduler state, so emit one
        # throwaway kernel before recording results.
        if not _ch._WARMED and kernels:
            _emit_one(kwa, kernels[0], splitGSU, canonical)
            _ch._WARMED = True

        for kernel in kernels:
            results.append(_emit_one(kwa, kernel, splitGSU, canonical))

    results.sort(key=lambda t: t[0])
    return results


def _emit_one(kwa, kernel, splitGSU, canonical):
    """Emit a single kernel via the codegen_harness machinery.

    Reuses ``codegen_harness._init_rocisa_for`` (per-kernel rocisa init),
    ``_prepare_kernel`` (sets BaseName), and ``canonicalize_asm`` so the emitted
    text matches the logic-driven harness exactly.
    """
    from Tensile.TensileCreateLibrary.Run import processKernelSource

    ri = _ch._init_rocisa_for(kernel)
    data = ri.getData()
    outOptions = ri.getOutputOptions()
    base = _ch._prepare_kernel(kernel, splitGSU)
    res = processKernelSource(kwa, data, outOptions, splitGSU, kernel)
    src = res.src
    if canonical:
        src = _ch.canonicalize_asm(src)
    elif isinstance(src, (bytes, bytearray)):
        src = src.decode(errors="replace")
    return base, src, res.err


# --- in-file smoke runner ---------------------------------------------------
#
# NOT a pytest test (no ``test_`` prefix; guarded under __main__). Drives the
# harness on one small Tests/common gemm config and asserts >=1 kernel emits
# with err==0. Run in-container:
#
#   python config_harness.py [<config path>]
#
# Defaults to the small single-permutation fp32_nt gemm config relative to the
# Tensile package root.

_SMOKE_DEFAULT_CONFIG = "Tensile/Tests/common/gemm/fp32_nt.yaml"


def _smoke(config_path=_SMOKE_DEFAULT_CONFIG):
    results = emit_kernels_from_config(config_path, limit=8)
    n = len(results)
    err0 = all(r[2] == 0 for r in results)
    print("KERNELS", n, "ERR0", err0)
    assert n >= 1, f"expected >=1 kernel, got {n}"
    assert err0, f"expected all err==0, got {[r[2] for r in results]}"
    return results


if __name__ == "__main__":
    import sys

    cfg = sys.argv[1] if len(sys.argv) > 1 else _SMOKE_DEFAULT_CONFIG
    _smoke(cfg)
