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
import os
import re
import tempfile

import pytest

from Tensile.Tests.rocisa_test_state import preserve_rocisa_kernel_state

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
    with preserve_rocisa_kernel_state():
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


def emit_kernels_from_config(config_path, limit=8, arch=_DEFAULT_ARCH, canonical=True,
                             splitGSU=False, cluster_dim=None):
    """Emit assembly for the kernels of a ``BenchmarkProblems`` config.

    Drives ``config -> BenchmarkProcess -> constructForkPermutations ->
    _generateForkedSolutions -> Solution(s)`` then emits each via the *same*
    path :mod:`codegen_harness` uses (``generateKernelObjectsFromSolutions`` +
    ``processKernelSource``), returning ``[(basename, source, err), ...]`` sorted
    by basename.

    ``err`` is the emitter return code (0 == ok). ``limit`` bounds both the
    number of fork permutations turned into solutions *and* the number of
    emitted kernels, so the rocisa per-process footprint stays small.

    ``cluster_dim``, when given, keeps only the kernels of that ClusterDim. A
    config that sweeps several cluster shapes can then be pinned one shape at a
    time (the kernel name is a hash, so the shape is not recoverable from it).
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
        if cluster_dim is not None:
            want = list(cluster_dim)
            kernels = [k for k in kernels if list(k["ClusterDim"]) == want]
            assert kernels, f"config {config_path} has no ClusterDim={want} kernel"
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


_CLONE_TARGET_RE = re.compile(r"^label_([A-Za-z0-9]+)_target_\d+:", re.M)
_LABEL_RE = re.compile(r"^(label_\S+):")


def _count_cluster_barriers(src):
    """Return ``(signals, waits)`` cluster-scope ``-3`` counts, discounting the
    copies RegionClonePass duplicated into cloned bodies.

    A cloned region is emitted as ``label_<Clone>_label_<original>_<idx>`` bodies
    that converge on a ``label_<Clone>_target_<idx>`` join, and the clone ends in
    an unconditional branch to that join. Only one of the bodies runs on any given
    path, so a barrier inside one is a per-path copy of its original rather than an
    extra dynamic arrive/completion, and it must not be counted twice.
    """
    clone_names = set(_CLONE_TARGET_RE.findall(src))
    label = None
    signals = waits = 0
    for line in src.split("\n"):
        matched = _LABEL_RE.match(line)
        if matched:
            label = matched.group(1)
            continue
        if label is not None and any(
            label.startswith(f"label_{name}_label_") for name in clone_names
        ):
            continue
        if line.startswith("s_barrier_signal -3"):
            signals += 1
        elif line.startswith("s_barrier_wait -3"):
            waits += 1
    return signals, waits


def assert_cluster_barrier_balanced(src, base):
    """Cluster-scope split-barrier balance check shared by the gfx1250 StreamK
    cluster char tests. Every arrive (``s_barrier_signal -3``) must be consumed by
    a completion (``s_barrier_wait -3``) on every control-flow path. The prologue
    wave-0 arrive is consumed by exactly one of two mutually exclusive cluster
    waits: the last-iteration guard's zero-iteration skip-edge wait, or the
    first-load wait on the >=1-iteration fall-through. Both waits are emitted
    statically but only one executes on any given path, so the static wait count
    is exactly one greater than the signal count; every other arrive (including a
    config's dedicated prologue-prefetch handshake) is a self-contained arrive/wait
    pair. Any other imbalance would leave a cluster wait unpaired and stall the
    cluster waves.

    Barriers inside cloned bodies are discounted the same way (see
    ``_count_cluster_barriers``): InsertClusterBarrierPass anchors the Rule 3
    signal a fixed cycle lead ahead of its wait, which can place it in a loop-begin
    block that RegionClonePass duplicates, so one arrive can have several static
    copies of which exactly one runs.
    """
    n_signal, n_wait = _count_cluster_barriers(src)
    assert n_wait == n_signal + 1, (
        f"Kernel {base!r}: unexpected cluster barrier balance: "
        f"{n_signal} signal(-3) vs {n_wait} wait(-3) (expected wait == signal + 1, "
        "both counted outside cloned bodies)"
    )


def derive_states(config_path, arch=_DEFAULT_ARCH, limit_solutions=8):
    """Return the derived Solution ``state`` dicts for a config (CPU-only).

    Shared by the StreamK-cluster / Multicast unit suites, which all pin the
    derived solution state (Multicast / ClusterBarrier / StreamKMulticast) rather
    than emitted asm. Unwraps ``Solution._state`` when present.
    """
    sols = solutions_from_config(config_path, arch=arch, limit_solutions=limit_solutions)
    return [s._state if hasattr(s, "_state") else s for s in sols]


def assert_real_gfx1250_kernels(results):
    """Shared preamble check for the gfx1250 StreamK cluster char drivers.

    Every emitted kernel must be real gfx1250 assembly: >=1 kernel, all err==0, a
    non-trivial body (>50 lines), the gfx1250 target directive, and the ``Cijk_``
    kernel-name prefix. Returns ``results`` for further per-file dispatch.
    """
    assert len(results) >= 1, f"Expected >=1 kernel, got {len(results)}"
    bad = [(b, e) for (b, _s, e) in results if e != 0]
    assert not bad, f"Expected all err==0, got: {bad}"
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 50, (
            f"Kernel {base!r} emitted suspiciously short source"
        )
        assert ".amdgcn_target" in src, f"Kernel {base!r} missing .amdgcn_target"
        assert "gfx1250" in src, f"Kernel {base!r} missing gfx1250 target"
        assert base.startswith("Cijk_"), f"Kernel {base!r} has unexpected prefix"
    return results


def golden_digest(results):
    """Order-invariant ``{basename, err}`` digest shared by the syrupy goldens."""
    return sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )


_TARGET_RE = re.compile(r'^\.amdgcn_target\s+"amdgcn-amd-amdhsa--(\S+?)"', re.M)
_WAVE32_RE = re.compile(r"^\s*\.amdhsa_wavefront_size32\s+1", re.M)


@functools.lru_cache(maxsize=1)
@functools.lru_cache(maxsize=1)
def _guard_assembler():
    """Assembler for :func:`assert_assembles`, built with a real code-object version.

    ``codegen_harness`` builds its shared assembler with ``"default"``, which is
    harmless while nothing invokes it but which clang rejects outright
    (``invalid integral value 'default' in '-mcode-object-version=default'``).
    Build a separate one on Tensile's own default version instead of retargeting
    the shared assembler, whose ``code_object_version`` reaches signature codegen
    through ``Solution``.
    """
    from Tensile.Common.GlobalParameters import globalParameters
    from Tensile.Toolchain.Assembly import makeAssemblyToolchain
    from Tensile.Toolchain.Validators import validateToolchain, ToolchainDefaults

    coVersion = str(globalParameters["CodeObjectVersion"])
    if not coVersion.isdigit():
        coVersion = "4"
    cxx = validateToolchain("amdclang++")
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    return makeAssemblyToolchain(cxx, bundler, coVersion).assembler


def _assembler_or_reason():
    """Return ``(assembler, None)``, or ``(None, reason)`` if none is usable."""
    try:
        return _guard_assembler(), None
    except Exception as exc:  # noqa: BLE001 - any toolchain problem is a skip
        return None, f"ROCm assembler unavailable: {exc}"


def assert_assembles(src, base):
    """Assert the ROCm assembler accepts ``src``.

    The rest of the cluster assertions only pattern-match assembly *text*, so a
    kernel that names an SGPR the allocator never defined, or that puts two
    literals in one SOP2, still satisfies them and only breaks much later when a
    real build assembles it. Feeding the emitted text to the assembler closes
    that gap in the fast unit layer. Target and wavefront size come from the
    emitted directives so this stays arch-agnostic; skips when the toolchain has
    no assembler.
    """
    assembler, reason = _assembler_or_reason()
    if assembler is None:
        pytest.skip(reason)
    target = _TARGET_RE.search(src)
    assert target, f"Kernel {base!r} has no .amdgcn_target to assemble for"
    waveSize = 32 if _WAVE32_RE.search(src) else 64
    with tempfile.TemporaryDirectory() as tmpDir:
        srcPath = os.path.join(tmpDir, "kernel.s")
        with open(srcPath, "w") as fh:
            fh.write(src)
        try:
            assembler(target.group(1), waveSize, srcPath, os.path.join(tmpDir, "kernel.o"))
        except RuntimeError as exc:
            pytest.fail(f"Kernel {base!r} does not assemble for {target.group(1)}: {exc}")


def assert_split_multicast_masks(src, base):
    """Split topology: each operand carries its own mask on its own descriptor.

    B broadcasts along Cs and A along Ck; when Ck == 1 the A mask degenerates to
    the self bit but is still bound, so both attaches are expected either way.
    """
    assert "s[sgprtdmBGroup1], s[sgprtdmBGroup1], s[sgprMulticastMaskB]" in src, (
        f"Kernel {base!r} missing B-broadcast mask on the B descriptor"
    )
    assert "s[sgprtdmAGroup1], s[sgprtdmAGroup1], s[sgprMulticastMaskA]" in src, (
        f"Kernel {base!r} missing the A mask on the A descriptor"
    )


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
