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
assembly is canonicalized in exactly the same way.

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
import hashlib
import json
import os
import re
import tempfile
from collections import Counter

import pytest

from Tensile.Tests.rocisa_test_state import preserve_rocisa_kernel_state

# Reuse the logic-driven harness for: assembler/toolchain construction, the
# canonicalized emit, global-state isolation, and per-kernel rocisa
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


def _benchmark_problem_selection(problem):
    """Return the fields that determine solution generation for one entry.

    ``BenchmarkFinalParameters`` contains runtime problem sizes and benchmark
    arguments. Those values do not participate in solution derivation or kernel
    emission, so they must not invalidate a set-cover selector.
    """
    problem_type, problem_size_group = problem[:2]
    generation_parameters = {
        key: value
        for key, value in problem_size_group.items()
        if key != "BenchmarkFinalParameters"
    }
    return [problem_type, generation_parameters]


def benchmark_problem_fingerprint(problem):
    """Return a short identifier for an entry's solution-generation inputs."""
    encoded = json.dumps(
        _benchmark_problem_selection(problem), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def benchmark_problem_fingerprints(config_path):
    """Return ``(index, fingerprint)`` pairs for every problem group."""
    config = _load_config(config_path)
    return [
        (index, benchmark_problem_fingerprint(problem))
        for index, problem in enumerate(config.get("BenchmarkProblems", []))
    ]


def _select_benchmark_problem(benchmark_problems, config_path, problem_index, fingerprint):
    if not benchmark_problems:
        return None
    if fingerprint is not None:
        available = [benchmark_problem_fingerprint(problem) for problem in benchmark_problems]
        matches = [
            problem
            for problem in benchmark_problems
            if benchmark_problem_fingerprint(problem) == fingerprint
        ]
        if not matches:
            raise ValueError(
                f"BenchmarkProblems fingerprint {fingerprint!r} does not exist in "
                f"{config_path}; available fingerprints: {available}"
            )
        return matches[0]
    if not 0 <= problem_index < len(benchmark_problems):
        raise IndexError(
            f"BenchmarkProblems index {problem_index} is out of range for "
            f"{config_path} ({len(benchmark_problems)} entries)"
        )
    return benchmark_problems[problem_index]


def _solutions_from_config_unguarded(
    config_path,
    assembler,
    isaInfoMap,
    limit_solutions=None,
    problem_index=0,
    problem_fingerprint=None,
    expected_fork_count=None,
):
    """Build ``Solution`` objects from one selected BenchmarkProblems entry.

    Walks the real config-driven path: ``BenchmarkProcess`` parses the
    ProblemType + ProblemSizeGroup, ``constructForkPermutations`` enumerates the
    fork cartesian product, and ``_generateForkedSolutions`` derives one
    ``Solution`` per permutation. CPU-only; nothing is compiled or run.

    ``limit_solutions`` caps the number of fork permutations fed to solution
    generation (keeps the rocisa per-process footprint bounded for big sweeps).
    ``problem_fingerprint`` selects an entry by content and takes precedence
    over ``problem_index``. ``expected_fork_count`` checks the complete
    permutation count before the cap is applied.
    """
    from Tensile.BenchmarkProblems import _generateForkedSolutions
    from Tensile.BenchmarkStructs import BenchmarkProcess, constructForkPermutations
    from Tensile.Common.Types import makeDebugConfig

    config = _load_config(config_path)
    benchmarkProblems = config["BenchmarkProblems"]
    selected_problem = _select_benchmark_problem(
        benchmarkProblems, config_path, problem_index, problem_fingerprint
    )
    if selected_problem is None:
        return []

    # Each BenchmarkProblems entry is [ProblemTypeConfig, ProblemSizeGroupConfig].
    problemTypeConfig, problemSizeGroupConfig = selected_problem[:2]

    debugConfig = makeDebugConfig(config.get("GlobalParameters", {}))

    benchmarkProcess = BenchmarkProcess(problemTypeConfig, problemSizeGroupConfig, False)
    benchmarkStep = benchmarkProcess[0]

    if problemSizeGroupConfig.get("ForkParameters"):
        forkPermutations = constructForkPermutations(benchmarkStep.forkParams, benchmarkStep.paramGroups)
        perms = list(forkPermutations)
    else:
        perms = []

    if expected_fork_count is not None:
        assert len(perms) == expected_fork_count, (
            f"expected {expected_fork_count} fork permutations for {config_path}, "
            f"got {len(perms)}"
        )

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


def solutions_from_config(
    config_path,
    arch=_DEFAULT_ARCH,
    limit_solutions=None,
    problem_index=0,
    problem_fingerprint=None,
    expected_fork_count=None,
):
    """Return fully-derived ``Solution`` objects for ``config_path`` (CPU-only).

    Runs under global-state isolation so it does not leak into other tests.
    """
    assembler, iim = _toolchain_for(arch)
    with _isolated_globals_with_isa(iim):
        return _solutions_from_config_unguarded(
            config_path,
            assembler,
            iim,
            limit_solutions,
            problem_index,
            problem_fingerprint,
            expected_fork_count,
        )


def emit_kernels_from_config(config_path, limit=8, arch=_DEFAULT_ARCH, canonical=True,
                             splitGSU=False, cluster_dim=None, problem_index=0,
                             problem_fingerprint=None, expected_fork_count=None):
    """Emit assembly for the kernels of a ``BenchmarkProblems`` config.

    Drives ``config -> BenchmarkProcess -> constructForkPermutations ->
    _generateForkedSolutions -> Solution(s)`` then emits each via the *same*
    path :mod:`codegen_harness` uses (``generateKernelObjectsFromSolutions`` +
    ``processKernelSource``), returning ``[(basename, source, err), ...]`` sorted
    by basename.

    ``err`` is the emitter return code (0 == ok). ``limit`` bounds both the
    number of fork permutations turned into solutions *and* the number of
    emitted kernels, so the rocisa per-process footprint stays small.

    ``expected_fork_count`` checks the full selected search size before
    ``limit`` is applied.

    ``cluster_dim``, when given, keeps only the kernels of that ClusterDim. A
    config that sweeps several cluster shapes can then be pinned one shape at a
    time (the kernel name is a hash, so the shape is not recoverable from it).

    ``problem_fingerprint`` selects a stable problem-group identity and takes
    precedence over the positional ``problem_index``.
    """
    import rocisa  # noqa: F401  (ensures the singleton module is importable here)
    from Tensile.TensileCreateLibrary.Run import generateKernelObjectsFromSolutions
    from Tensile.KernelWriterAssembly import KernelWriterAssembly
    from Tensile.Common.Types import DebugConfig
    from Tensile.SolutionStructs.Naming import getKernelFileBase

    assembler, iim = _toolchain_for(arch)

    results = []
    with _isolated_globals_with_isa(iim):
        sols = _solutions_from_config_unguarded(
            config_path,
            assembler,
            iim,
            limit_solutions=limit,
            problem_index=problem_index,
            problem_fingerprint=problem_fingerprint,
            expected_fork_count=expected_fork_count,
        )
        kernels = generateKernelObjectsFromSolutions(sols)
        if cluster_dim is not None:
            want = list(cluster_dim)
            kernels = [k for k in kernels if list(k["ClusterDim"]) == want]
            assert kernels, f"config {config_path} has no ClusterDim={want} kernel"
        if limit is not None:
            kernels = sorted(kernels, key=lambda k: getKernelFileBase(splitGSU, k))[:limit]
        kwa = KernelWriterAssembly(assembler, DebugConfig())

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


def _assert_real_kernels(results, arch, *, require_all_ok=True):
    """Check that successful results contain non-trivial target assembly."""
    assert results, "expected at least one emitted kernel"
    errors = [(base, err) for base, _source, err in results if err != 0]
    if require_all_ok:
        assert not errors, f"expected all kernels to emit successfully, got {errors}"
    for base, src, err in results:
        if err != 0:
            continue
        if isinstance(src, (bytes, bytearray)):
            src = src.decode(errors="replace")
        src = src or ""
        assert len(src.splitlines()) > 50, f"kernel {base!r}: suspiciously short assembly"
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert arch in src, f"kernel {base!r}: wrong arch in assembly"
        assert base.startswith("Cijk_"), f"kernel {base!r}: unexpected prefix"
    return results


def assert_real_gfx1250_kernels(results):
    """Shared source check for the gfx1250 StreamK cluster drivers.

    Every emitted kernel must be real gfx1250 assembly: >=1 kernel, all err==0, a
    non-trivial body (>50 lines), the gfx1250 target directive, and the ``Cijk_``
    kernel-name prefix. Returns ``results`` for further per-file dispatch.
    """
    return _assert_real_kernels(results, "gfx1250")


def golden_digest(results):
    """Return the order-invariant kernel-identity and status saved result."""
    return sorted(
        ({"basename": base, "err": err} for base, _source, err in results),
        key=lambda item: item["basename"],
    )


def assert_config_emits(
    config_path,
    arch,
    *,
    limit=8,
    validate_source=False,
    problem_index=0,
    problem_fingerprint=None,
    expected_fork_count=None,
    expected_statuses=None,
    required_source_patterns=(),
):
    """Emit one configuration once and check its declared observable behavior.

    ``required_source_patterns`` is a sequence of ``(description, regex)`` pairs.
    Each expression must occur in at least one successfully emitted kernel. This
    lets a test name the instruction or source structure it exists to protect
    without recording the complete compiler-dependent assembly.

    ``expected_statuses`` is an exact ``{return_code: count}`` mapping for the
    bounded emitted sample. If omitted, every emitted kernel must succeed.
    """
    results = emit_kernels_from_config(
        config_path,
        limit=limit,
        arch=arch,
        problem_index=problem_index,
        problem_fingerprint=problem_fingerprint,
        expected_fork_count=expected_fork_count,
    )
    assert results, f"expected >=1 kernel, got {len(results)}"
    actual_statuses = Counter(err for _base, _source, err in results)
    if expected_statuses is None:
        errors = [(base, err) for base, _src, err in results if err != 0]
        assert not errors, f"expected all kernels to emit successfully, got {errors}"
    else:
        assert actual_statuses == Counter(expected_statuses), (
            f"expected emitter statuses {dict(Counter(expected_statuses))}, "
            f"got {dict(actual_statuses)}"
        )

    successful_sources = []
    for base, src, err in results:
        if err != 0:
            continue
        if isinstance(src, (bytes, bytearray)):
            src = src.decode(errors="replace")
        src = src or ""
        successful_sources.append(src)
    if validate_source:
        _assert_real_kernels(results, arch, require_all_ok=False)

    combined_source = "\n".join(successful_sources)
    for description, pattern in required_source_patterns:
        assert re.search(pattern, combined_source, re.MULTILINE), (
            f"emitted assembly is missing {description}: /{pattern}/"
        )
    return results


def assert_config_rejects(config_path, arch, monkeypatch, capsys, expected_rejections):
    """Derive a configuration serially and check its exact rejection multiset.

    This helper is consumed by the dependent terminal characterization PR,
    whose designed zero-survivor cases exercise validation through the config
    front end rather than through a pre-built solution state.
    """
    import Tensile.BenchmarkProblems as benchmark_problems

    def serial_map(function, objects, *_args, **_kwargs):
        return [function(*args) for args in objects]

    monkeypatch.setattr(benchmark_problems, "ParallelMap2", serial_map)
    solutions = solutions_from_config(config_path, arch=arch)
    rejection_counts = Counter(
        line.strip()
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("reject:")
    )
    assert not solutions, f"expected 0 surviving solutions, got {len(solutions)}"
    assert rejection_counts == Counter(expected_rejections)


_TARGET_RE = re.compile(r'^\.amdgcn_target\s+"amdgcn-amd-amdhsa--(\S+?)"', re.M)
_WAVE32_RE = re.compile(r"^\s*\.amdhsa_wavefront_size32\s+1", re.M)


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


_SYMBOL_BREADCRUMB_RE = re.compile(
    r"([sv](?:\d+|\[\d+:\d+\])) was (\w+(?:\+-?\d+)*) \((?:unresolved \.set|split)\)"
)
# Same operand and the same ``was``, any reason: catches a reason string the
# allocator grows that the pattern above does not know, which would otherwise
# leave the operand numeric and fail an assertion for a reason nobody can see.
_BREADCRUMB_SHAPE_RE = re.compile(r"[sv](?:\d+|\[\d+:\d+\]) was \S+ \([^)]*\)")
_OPERAND_RE = re.compile(r"(?<!\w)[sv](?:\d+|\[\d+:\d+\])(?!\w)")


def with_symbol_names(src):
    """Return ``src`` with allocator-renumbered operands spelled by name again.

    ``StinkyTofuRegisterAllocation >= 2`` compacts SGPRs, and an operand it moves
    loses its symbolic name whenever the name's ``.set`` cannot be resolved --
    which is most of them, because TensileLite ends a name's scope with a second
    ``.set NAME, UNDEF``. So ``s[sgprMulticastMaskA]`` prints as ``s0``. The
    instruction is unchanged; only the spelling is, and every assertion that
    matches assembly text by name stops matching.

    The allocator records what it renamed in a trailing ``s0 was
    sgprMulticastMaskA (unresolved .set)`` note, so the original spelling is
    recoverable: substitute each renamed operand back on the line that documents
    it. With no such notes -- allocation off, or an arch that does not emit them
    -- the text is returned byte-identical, so an assertion reads the same in
    both modes.

    Match names against this view and assemble the real text
    (``assert_assembles``), which is the only check that can tell the two apart.
    """
    out = []
    for line in src.splitlines(keepends=True):
        code, sep, note = line.partition("//")
        crumbs = _SYMBOL_BREADCRUMB_RE.findall(note)
        shapes = len(_BREADCRUMB_SHAPE_RE.findall(note))
        assert shapes == len(crumbs), (
            f"unrecognized allocator breadcrumb, so this operand would stay "
            f"numeric and every name-matching assertion on it would fail "
            f"silently: {note.strip()!r}"
        )
        if not crumbs:
            out.append(line)
            continue

        # One physical register can carry two names on one line, e.g. a
        # destination the allocator moved to where a source's symbol also
        # landed. The allocator writes its notes in operand order (destinations
        # then sources, and one note per distinct fact), so consume them in that
        # order rather than substituting a name everywhere it fits -- which
        # would spell the second operand with the first operand's symbol.
        pending = {}
        for physical, name in crumbs:
            pending.setdefault(physical, []).append(name)

        def spell(match):
            names = pending.get(match.group(0))
            if not names:
                return match.group(0)
            # A single note covers every occurrence: it is either the only use
            # or two uses the allocator deduplicated because they agree.
            name = names.pop(0) if len(names) > 1 else names[0]
            span = re.fullmatch(r"([sv])\[(\d+):(\d+)\]", match.group(0))
            if span:
                width = int(span.group(3)) - int(span.group(2))
                return f"{span.group(1)}[{name}:{name}+{width}]"
            return f"{match.group(0)[0]}[{name}]"

        out.append(_OPERAND_RE.sub(spell, code) + sep + note)
    return "".join(out)


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
