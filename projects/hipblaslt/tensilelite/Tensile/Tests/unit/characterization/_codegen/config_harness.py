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


def _kernelend_has_gated_persist_wait(src):
    """True when KernelEnd waits persist USER ``-3`` (GW_End skipped SK_CloseLoop)."""
    lines = src.splitlines()
    kend = next((i for i, ln in enumerate(lines)
                 if ln.startswith("label_KernelEnd:")), None)
    if kend is None:
        return False
    end = next(
        (i for i, ln in enumerate(lines[kend + 1 :], start=kend + 1)
         if ln.startswith("s_endpgm") or ln.startswith("label_ASM_End:")),
        None,
    )
    window = lines[kend:end]
    return (any("s_barrier_wait -3" in ln for ln in window)
            and not any("s_barrier_signal -3" in ln for ln in window)
            and any("wait last -3 at persist close" in ln for ln in window))


def assert_kernelend_gated_persist_wait(src, base):
    """GW_End skips SK_CloseLoop; KernelEnd must wait persist USER ``-3`` if arrived.

    Not wait-only (flag==0 skips; pads already ``s_endpgm``). SK_CloseLoop
    already waited and cleared the flag.
    """
    assert _kernelend_has_gated_persist_wait(src), (
        f"Kernel {base!r}: GW_End branches to KernelEnd skipping SK_CloseLoop; "
        f"KernelEnd must gated-wait persist USER -3 before s_endpgm"
    )


def assert_kernelend_tdm_drain(src, base):
    """KernelEnd must ``s_wait_tensorcnt 0`` (not a 500-char lookbehind).

    Drain in-flight TDM at KernelEnd so the next kernel does not start with
    leftover tensor ops. tensorcnt-only: a full tdmWait also waits vlcnt=0
    and can stall persist overlap. ``s_wait_xcnt 0`` is forbidden.
    Persist close must not tdmWait.
    """
    lines = src.splitlines()
    kend = next((i for i, ln in enumerate(lines)
                 if ln.startswith("label_KernelEnd:")), None)
    assert kend is not None, f"Kernel {base!r}: missing label_KernelEnd"
    end = next(
        (i for i, ln in enumerate(lines[kend + 1 :], start=kend + 1)
         if ln.startswith("s_endpgm") or ln.startswith("label_ASM_End:")),
        None,
    )
    assert end is not None, (
        f"Kernel {base!r}: missing s_endpgm after label_KernelEnd"
    )
    window = "\n".join(lines[kend:end])
    assert "s_wait_tensorcnt 0" in window, (
        f"Kernel {base!r}: KernelEnd must s_wait_tensorcnt 0 before s_endpgm "
        f"(deferred path must mirror functionEnd tdmWait). window={window!r}"
    )
    assert "s_wait_xcnt 0" not in window, (
        f"Kernel {base!r}: KernelEnd must not s_wait_xcnt 0 "
        f"(s_wait_xcnt 0 at KernelEnd stalls persist overlap)"
    )


def assert_cluster_barrier_balanced(src, base):
    """Cluster-scope split-barrier balance check shared by the gfx1250 StreamK
    cluster char tests. Every arrive (``s_barrier_signal -3``) must be consumed by
    a completion (``s_barrier_wait -3``) on every control-flow path.

    The prologue round is a self-contained arrive/wait (every launched member
    including pads signal+wait, then pads ``s_endpgm``). The first-load wait and
    zero-iteration skip wait are not emitted for that round. Every other arrive
    (including a config's dedicated prologue-prefetch handshake and later loop
    ``-3``) is also a self-contained arrive/wait pair, so the static wait count
    equals the signal count. Any other imbalance would leave a cluster wait
    unpaired and stall the cluster waves.

    Barriers inside cloned bodies are discounted the same way (see
    ``_count_cluster_barriers``): InsertClusterBarrierPass anchors the Rule 3
    signal a fixed cycle lead ahead of its wait, which can place it in a loop-begin
    block that RegionClonePass duplicates, so one arrive can have several static
    copies of which exactly one runs.
    """
    n_signal, n_wait = _count_cluster_barriers(src)
    # ForceDPOnly=0 persist PGR2: skipPGR2 and LDS1 load-path arrives are
    # mutually exclusive; persist close waits once. Static instruction count
    # would otherwise see +1 extra signal.
    if "PGR2 skipPGR2: wait last -3 at persist close" in src:
        n_signal -= 1
    # LC==0 ZeroIter and LC>0 skipPGR2 / LDS1 are also mutually exclusive;
    # persist PGR2 defers ZeroIter wait to the same persist-close wait.
    if "PGR2 ZeroIter: arrive now; wait last -3 at persist close" in src:
        n_signal -= 1
    # GW_End long-branches to KernelEnd and skips SK_CloseLoop. A gated
    # persist wait is emitted at both sites (dynamically exclusive).
    if _kernelend_has_gated_persist_wait(src):
        n_wait -= 1
    assert n_wait == n_signal, (
        f"Kernel {base!r}: unexpected cluster barrier balance: "
        f"{n_signal} signal(-3) vs {n_wait} wait(-3) (expected wait == signal, "
        "both counted outside cloned bodies)"
    )


def assert_skip_pgr2_skip_path_handshake(src, base):
    """The LC==1 skipPGR2 fall-through must complete the same ``-3`` round as
    the LDS1 load path.

    ForceDPOnly=0 [Cs,Ck] SK peers in one cluster can disagree on LoopCounterL
    (partial vs remainder K). Handshake only inside the load-path guard leaves
    LC==1 peers skipping ``s_barrier_signal/wait -3`` while others wait.

    ForceDPOnly=0 persist PGR2 arrives in the skip window and waits at
    persist close (WAVEDONE leftover). ForceDPOnly=1 keeps wait in-window.
    """
    lines = src.splitlines()
    skip_idx = join_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("label_skipPGR2_1:"):
            skip_idx = i
        elif ln.startswith("label_skipPGR2_2:") and skip_idx is not None:
            join_idx = i
            break
    assert skip_idx is not None, f"Kernel {base!r} missing label_skipPGR2_1"
    assert join_idx is not None, (
        f"Kernel {base!r} missing label_skipPGR2_2 after skipPGR2_1"
    )
    window = lines[skip_idx:join_idx]
    # SIA4 may drop the Tensile comment and insert s_wait_tensorcnt around
    # the handshake; the arrive (and wait, unless deferred to persist close)
    # is the contract.
    assert any("s_barrier_signal -3" in w for w in window), (
        f"Kernel {base!r} skipPGR2 LC==1 path missing s_barrier_signal -3"
    )
    defer_close = "PGR2 skipPGR2: wait last -3 at persist close" in src
    if defer_close:
        assert not any("s_barrier_wait -3" in w for w in window), (
            f"Kernel {base!r}: persist PGR2 skipPGR2 must not wait in-window "
            f"(wait last -3 at persist close). window={window!r}"
        )
        close_idx = next((i for i, ln in enumerate(lines)
                          if ln.startswith("label_SK_CloseLoop:")), None)
        assert close_idx is not None, f"Kernel {base!r} missing label_SK_CloseLoop"
        end = next(
            (i for i, ln in enumerate(lines[close_idx + 1 :], start=close_idx + 1)
             if ln.startswith("label_KernelEnd:") or ln.startswith("s_endpgm")),
            None,
        )
        close_win = lines[close_idx:end]
        assert any("s_barrier_wait -3" in ln for ln in close_win), (
            f"Kernel {base!r}: persist PGR2 skipPGR2 must s_barrier_wait -3 "
            f"at persist close. window={close_win!r}"
        )
    else:
        assert any("s_barrier_wait -3" in w for w in window), (
            f"Kernel {base!r} skipPGR2 LC==1 path missing s_barrier_wait -3"
        )


def assert_skip_pgr2_leftover_tdm_drain(src, base):
    """After skipPGR2 -3 arrive (and in-window wait if not persist-close),
    drain TDM so leftover cannot outlive persist / next kernel.

    SIA=0 skip window has no wait_tensorcnt; leftover F8 MX descriptors parked
    at skipPGR2 hang the next kernel's persist-DP -3. Drain after the skip -3
    (not on it) and before WG -1. KernelEnd wait-only cluster -3 is
    membership-unbalanced (pads already s_endpgm) and is not this close.
    Persist PGR2 waits USER -3 at persist close; leftover TDM is still
    tensorcnt-only after the skip arrive.
    """
    lines = src.splitlines()
    skip_idx = join_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("label_skipPGR2_1:"):
            skip_idx = i
        elif ln.startswith("label_skipPGR2_2:") and skip_idx is not None:
            join_idx = i
            break
    assert skip_idx is not None, f"Kernel {base!r} missing label_skipPGR2_1"
    assert join_idx is not None, f"Kernel {base!r} missing label_skipPGR2_2"
    window = lines[skip_idx:join_idx]
    waits = [i for i, w in enumerate(window) if "s_barrier_wait -3" in w]
    sigs = [i for i, w in enumerate(window) if "s_barrier_signal -3" in w]
    assert sigs, f"Kernel {base!r} skipPGR2 missing s_barrier_signal -3"
    after = skip_idx + (waits[0] if waits else sigs[0]) + 1
    end = next(
        (i for i, ln in enumerate(lines[after:], start=after)
         if "ds_load" in ln or "tensor_load_to_lds" in ln),
        None,
    )
    assert end is not None, (
        f"Kernel {base!r} missing ds_load / tensor_load_to_lds after skipPGR2 -3"
    )
    drain = lines[after:end]
    prologue_drained = any("s_wait_tensorcnt" in ln for ln in lines[:skip_idx])
    if not prologue_drained:
        assert any("s_wait_tensorcnt" in ln for ln in drain), (
            f"Kernel {base!r}: skipPGR2 leftover TDM must drain after skip -3 "
            f"and before LDS/TDM consume when the prologue has not drained. "
            f"drain={drain!r}"
        )


def assert_zero_iter_prefetch_handshake_preserves_scc(src, base):
    """LC==0 skip-prefetch must complete the PGR>=2 prefetch ``-3`` without
    clobbering SCC for ``longBranchScc1``.

    SIA4 may drop standalone Tensile comment0 lines; the skip/signal labels
    and the SCC save/restore comments on real instructions are the contract.
    """
    assert "label_SKMC_ZeroIterSkipHS" in src, (
        f"Kernel {base!r}: missing LC==0 skip-prefetch -3 pairing "
        f"(label_SKMC_ZeroIterSkipHS)"
    )
    assert "label_SKMC_ZeroIterSignal" in src, (
        f"Kernel {base!r}: missing LC==0 skip-prefetch elect-signal "
        f"(label_SKMC_ZeroIterSignal)"
    )
    assert "save checkLastIter SCC (1 iff LC==0)" in src, (
        f"Kernel {base!r}: missing checkLastIter SCC save before LC==0 handshake"
    )
    assert "restore SCC for longBranchScc1 (LC==0)" in src, (
        f"Kernel {base!r}: missing SCC restore for longBranchScc1 after LC==0 handshake"
    )
    if "PGR2 skipPGR2: wait last -3 at persist close" in src:
        lines = src.splitlines()
        zskip = next((i for i, ln in enumerate(lines)
                      if ln.startswith("label_SKMC_ZeroIterSkipHS")), None)
        start = next(
            (i for i, ln in enumerate(lines)
             if "LC>0: skipPGR2 / LDS1 handshake is the matching round" in ln
             and (zskip is None or i < zskip)),
            None,
        )
        assert start is not None and zskip is not None and start < zskip, (
            f"Kernel {base!r}: persist PGR2 ZeroIter must skipHS after LC>0 branch"
        )
        window = lines[start:zskip]
        assert any("s_barrier_signal -3" in w for w in window), (
            f"Kernel {base!r}: persist PGR2 ZeroIter missing s_barrier_signal -3"
        )
        assert not any("s_barrier_wait -3" in w for w in window), (
            f"Kernel {base!r}: persist PGR2 ZeroIter must not wait in-window "
            f"(wait last -3 at persist close). window={window!r}"
        )
        assert "PGR2 ZeroIter: arrive now; wait last -3 at persist close" in src, (
            f"Kernel {base!r}: persist PGR2 ZeroIter must set persist-open flag"
        )


def assert_persist_open_until_wavedone(src, base):
    """Persist-open flag must survive until persist wait; pads must not skip it.

    WAVEDONE with USER ``-3`` still open leaves ``signal_count != 0`` for the
    next kernel. Pads ``s_endpgm`` only after the prologue ``-3`` wait, before
    persist-open. SK tile math must not dst-write the persist-open SGPR
    between persist-open and persist wait. Dedicated PersistDpMcOpen is
    ``s_mov 0/1``; skipPGR2 load/skip/ZeroIter can each set the flag.
    """
    lines = src.splitlines()
    or_is = [
        i for i, ln in enumerate(lines)
        if "arrive now; wait last -3 at persist close" in ln
        and ("s_or_b32" in ln or "s_mov_b32" in ln)
    ]
    assert or_is, (
        f"Kernel {base!r}: missing persist-open OR/Mov "
        f"(arrive now; wait last -3 at persist close)"
    )
    m = re.search(r"s_(?:or|mov)_b32 s(\d+),", lines[or_is[0]])
    assert m, f"Kernel {base!r}: persist-open dst not sN in {lines[or_is[0]]!r}"
    idx = m.group(1)
    first_or = or_is[0]

    def _is_persist_close_wait(i, ln):
        if "s_barrier_wait -3" not in ln:
            return False
        window = "\n".join(lines[max(0, i - 4): i + 1])
        return (
            "wait last -3 at persist close" in window
            or "complete last -3 before persist re-entry" in ln
        )

    wait_i = next(
        (i for i, ln in enumerate(lines) if _is_persist_close_wait(i, ln)),
        None,
    )
    assert wait_i is not None and wait_i > first_or, (
        f"Kernel {base!r}: persist wait must follow persist-open OR"
    )
    dst_re = re.compile(
        r"^\s*(?:s_mov_b32|s_cmov_b32|s_cselect_b32|s_add_u32|s_sub_u32|"
        r"s_mul_i32|s_lshl_b32|s_lshr_b32|v_readfirstlane_b32)\s+s%s," % idx
    )
    # Dedicated PersistDpMcOpen is s_mov 0/1 (skipPGR2 load vs skip vs ZeroIter
    # can each set the flag). Packed-bit AND/OR of the same SGPR is also the
    # persist-open flag, not SK tile math.
    allowed_flag = re.compile(r"^\s*s_mov_b32 s%s, [01]\b" % idx)
    allowed_and_or = re.compile(
        r"^\s*s_(?:and|or)_b32 s%s, s%s, (?:1|2)\b" % (idx, idx)
    )
    for i, ln in enumerate(lines[first_or + 1 : wait_i], start=first_or + 2):
        if "arrive now; wait last -3 at persist close" in ln:
            continue
        if allowed_flag.search(ln) or allowed_and_or.search(ln):
            continue
        if dst_re.search(ln):
            raise AssertionError(
                f"Kernel {base!r}: SK tile math must not dst-write persist-open "
                f"s{idx} between persist-open and persist wait. line {i}: {ln}"
            )
    pad_endpgm = [
        i for i, ln in enumerate(lines)
        if ln.startswith("s_endpgm")
        and "padded work-group" in ln
    ]
    assert pad_endpgm, f"Kernel {base!r}: missing pad s_endpgm after prologue -3"
    for pi in pad_endpgm:
        assert pi < first_or, (
            f"Kernel {base!r}: pad s_endpgm at {pi} is after persist-open "
            f"(must exit on idle prologue -3, not skip persist wait)"
        )
        window = lines[max(0, pi - 8):pi]
        assert any("s_barrier_wait -3" in w for w in window), (
            f"Kernel {base!r}: pad s_endpgm must follow prologue -3 wait. "
            f"window={window!r}"
        )
    for i, ln in enumerate(lines[first_or:], start=first_or + 1):
        if not ln.startswith("s_endpgm"):
            continue
        pre = "\n".join(lines[max(first_or, i - 40):i])
        assert "s_barrier_wait -3" in pre and (
            "wait last -3 at persist close" in pre
            or "complete last -3 before persist re-entry" in pre
        ), (
            f"Kernel {base!r}: s_endpgm after persist-open must be KernelEnd "
            f"after gated persist wait (not a pad/no-work skip). line {i}: {ln}"
        )


def assert_pgr1_persist_dp_close_wait(src, base):
    """PGR1 persist-DP must wait USER ``-3`` at persist close, not only graWorkGroup.

    Arrive at graWorkGroup then GEMM/TDM, wait at SK_CloseLoop among remaining
    WGs (pads already ``s_endpgm``). Waiting only at graWorkGroup leaves
    ``signal_count != 0`` at WAVEDONE. Continue-SK must not wait-only. GW_End
    long-branches to KernelEnd and skips SK_CloseLoop, so KernelEnd also
    gated-waits (not wait-only).
    """
    lines = src.splitlines()
    close_idx = next((i for i, ln in enumerate(lines)
                      if ln.startswith("label_SK_CloseLoop:")), None)
    assert close_idx is not None, f"Kernel {base!r} missing label_SK_CloseLoop"
    end = next(
        (i for i, ln in enumerate(lines[close_idx + 1 :], start=close_idx + 1)
         if ln.startswith("label_KernelEnd:") or ln.startswith("s_endpgm")),
        None,
    )
    assert end is not None, (
        f"Kernel {base!r}: missing KernelEnd / s_endpgm after SK_CloseLoop"
    )
    window = lines[close_idx:end]
    assert any("s_barrier_wait -3" in ln for ln in window), (
        f"Kernel {base!r}: PGR1 persist-DP must s_barrier_wait -3 at persist "
        f"close before re-entry / KernelEnd. window={window!r}"
    )
    assert any("PGR1 persist-DP: wait last -3 at persist close" in ln for ln in window), (
        f"Kernel {base!r}: persist close must wait last persist-DP -3 "
        f"(graWorkGroup arrive is not a WAVEDONE-idle close)"
    )
    assert not any("s_barrier_signal -3" in ln for ln in window), (
        f"Kernel {base!r}: persist close must not arrive -3 "
        f"(pads / owner-fixup / already-WAVEDONE). window={window!r}"
    )
    assert "PGR1 persist-DP: arrive now; wait last -3 at persist close" in src, (
        f"Kernel {base!r}: PGR1 persist-DP must arrive in graWorkGroup and "
        f"defer the wait to persist close"
    )
    # PGR1 arrive-only in graWorkGroup; wait at persist close.
    or_i = next(
        (i for i, ln in enumerate(lines)
         if "PGR1 persist-DP: arrive now; wait last -3 at persist close" in ln
         and ("s_or_b32" in ln or "s_mov_b32" in ln)),
        None,
    )
    assert or_i is not None, (
        f"Kernel {base!r}: missing PGR1 persist-open OR/Mov"
    )
    skip_mc_i = next(
        (i for i, ln in enumerate(lines[or_i + 1 :], start=or_i + 1)
         if ln.startswith("label_SK_SkipPassMulticast")),
        None,
    )
    assert skip_mc_i is not None, (
        f"Kernel {base!r}: missing label_SK_SkipPassMulticast after persist-open"
    )
    post = lines[or_i + 1:skip_mc_i]
    assert any("s_barrier_signal -3" in ln for ln in post), (
        f"Kernel {base!r}: persist-open must arrive USER -3 in graWorkGroup. "
        f"post={post!r}"
    )
    assert not any("s_barrier_wait -3" in ln for ln in post), (
        f"Kernel {base!r}: persist-DP arrive must not wait in graWorkGroup "
        f"(wait is at persist close). post={post!r}"
    )
    assert "PGR1 persist-DP: wait last -3 at persist close" in src, (
        f"Kernel {base!r}: missing persist-close wait of last persist-DP -3"
    )
    assert "PGR2 skipPGR2: wait last -3 at persist close" not in src, (
        f"Kernel {base!r}: PGR1 has no skipPGR2 persist-close wait"
    )
    assert_kernelend_gated_persist_wait(src, base)
    assert_kernelend_tdm_drain(src, base)
    assert ".set sgprPersistDpMcOpen, UNDEF" not in "\n".join(
        lines[:close_idx]
    ), (
        f"Kernel {base!r}: PersistDpMcOpen must stay live past endSummation "
        f"until persist close / KernelEnd"
    )
    assert_persist_open_until_wavedone(src, base)


def assert_pgr2_persist_prefetch_close_wait(src, base):
    """PGR2 persist skipPGR2 / LDS1 must wait USER ``-3`` at persist close.

    graWorkGroup still waits the per-pass DP round (must be idle before this
    later ``-3``). Arrive at skipPGR2 / LDS1 then GEMM/TDM, wait at
    SK_CloseLoop among remaining WGs (pads already ``s_endpgm``). Waiting
    only at skipPGR2 leaves ``signal_count != 0`` at WAVEDONE. Continue-SK
    must not wait-only. GW_End long-branches to KernelEnd and skips
    SK_CloseLoop, so KernelEnd also gated-waits (not wait-only).
    """
    lines = src.splitlines()
    close_idx = next((i for i, ln in enumerate(lines)
                      if ln.startswith("label_SK_CloseLoop:")), None)
    assert close_idx is not None, f"Kernel {base!r} missing label_SK_CloseLoop"
    end = next(
        (i for i, ln in enumerate(lines[close_idx + 1 :], start=close_idx + 1)
         if ln.startswith("label_KernelEnd:") or ln.startswith("s_endpgm")),
        None,
    )
    assert end is not None, (
        f"Kernel {base!r}: missing KernelEnd / s_endpgm after SK_CloseLoop"
    )
    window = lines[close_idx:end]
    assert any("s_barrier_wait -3" in ln for ln in window), (
        f"Kernel {base!r}: PGR2 skipPGR2 must s_barrier_wait -3 at persist "
        f"close before re-entry / KernelEnd. window={window!r}"
    )
    assert any("PGR2 skipPGR2: wait last -3 at persist close" in ln for ln in window), (
        f"Kernel {base!r}: persist close must wait last skipPGR2 -3 "
        f"(skipPGR2 in-window wait is not a WAVEDONE-idle close)"
    )
    assert not any("s_barrier_signal -3" in ln for ln in window), (
        f"Kernel {base!r}: persist close must not arrive -3 "
        f"(pads / owner-fixup / already-WAVEDONE). window={window!r}"
    )
    assert "PGR2 skipPGR2: arrive now; wait last -3 at persist close" in src, (
        f"Kernel {base!r}: PGR2 skipPGR2 must arrive at prefetch and "
        f"defer the wait to persist close"
    )
    assert "PGR2 skipPGR2: wait last -3 at persist close" in src, (
        f"Kernel {base!r}: missing persist-close wait of last skipPGR2 -3"
    )
    assert "PGR1 persist-DP: wait last -3 at persist close" not in src, (
        f"Kernel {base!r}: PGR>=2 must wait per-pass persist-DP -3 in "
        f"graWorkGroup; persist-DP close wait is PGR1-only"
    )
    assert_kernelend_gated_persist_wait(src, base)
    assert_kernelend_tdm_drain(src, base)
    assert ".set sgprPersistDpMcOpen, UNDEF" not in "\n".join(
        lines[:close_idx]
    ), (
        f"Kernel {base!r}: PersistDpMcOpen must stay live past endSummation "
        f"until persist close / KernelEnd"
    )
    assert_persist_open_until_wavedone(src, base)


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
