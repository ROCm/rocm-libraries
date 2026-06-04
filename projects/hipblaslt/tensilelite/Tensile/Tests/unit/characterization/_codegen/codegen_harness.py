################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""Shared CPU-only codegen-emit harness for the characterization suites.

This is *not* a test module (no ``test_`` prefix, not collected). It drives the
real TensileLite assembly emitter end-to-end **without a GPU**:

    logic YAML  ->  parseLibraryLogicFile  ->  Solution(s)
                ->  generateKernelObjectsFromSolutions  ->  kernel dict(s)
                ->  KernelWriterAssembly.getSourceFileString  ->  assembly text

Only the *emit* is exercised here; assembling to a code object (amdclang++) and
running on hardware are deliberately out of scope. The emitted text is
deterministic given a pinned toolchain/ISA once random label suffixes are
canonicalized (see :func:`canonicalize_asm`), which makes it an ideal golden
target for the codegen surface (``KernelWriterAssembly``, ``KernelWriter``,
``Components/*``, ``Asm*``).

Usage from a suite::

    from codegen_harness import emit_kernels_from_logic
    results = emit_kernels_from_logic(LOGIC_PATH)      # [(basename, canon_src, err), ...]

The expensive toolchain/cap-map build is cached process-wide so many suites
share one construction.
"""

import contextlib
import copy
import functools
import re
import shutil

# --- assembly canonicalization ---------------------------------------------

# The emitter tags branch/loop labels with a random 16-char [A-Z0-9] suffix
# (e.g. ``label_NoBranch_T8JHFHKM7BO5OHXW``). That suffix is the *only* source
# of run-to-run nondeterminism in the emitted text. We map each distinct random
# token to a stable sequential id by first-appearance order, which preserves the
# label<->reference correspondence while removing the randomness.
_RANDOM_LABEL_SUFFIX = re.compile(r"_[A-Z0-9]{16}\b")

# The WMMA matrix-reuse operand (`... matrix_a_reuse` / `matrix_b_reuse`) is a
# performance hint whose *presence* is rendered from rocisa's internal MMA
# scheduler state, which carries across kernels in a process. So the same kernel
# emits the hint or not depending on emit order / warm-state — a benign,
# correctness-neutral churn (like register numbering). We strip it so goldens
# key on the stable structure. Documented as a pinned finding in resistance.md.
_MATRIX_REUSE = re.compile(r" matrix_[ab]_reuse\b")

# Set once the emitter has been driven through one throwaway kernel (see
# emit_kernels_from_logic) so every returned emit is the warm steady-state.
_WARMED = False


def canonicalize_asm(text):
    """Return ``text`` with random label suffixes replaced by stable ids.

    Deterministic and order-preserving: the Nth *distinct* random suffix seen
    becomes ``_LBL{N}`` everywhere it appears, so a label definition and its
    branch targets stay consistent.
    """
    if text is None:
        return None
    if isinstance(text, (bytes, bytearray)):
        text = text.decode(errors="replace")
    text = _MATRIX_REUSE.sub("", text)
    mapping = {}

    def _repl(m):
        tok = m.group(0)
        if tok not in mapping:
            mapping[tok] = f"_LBL{len(mapping)}"
        return mapping[tok]

    return _RANDOM_LABEL_SUFFIX.sub(_repl, text)


# --- toolchain / cap-map (cached) ------------------------------------------


@functools.lru_cache(maxsize=1)
def _toolchain():
    """Build (assembler, isaInfoMap) once. Uses amdclang++; no GPU required."""
    from Tensile.Common.Architectures import SUPPORTED_ISA
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Toolchain.Assembly import makeAssemblyToolchain
    from Tensile.Toolchain.Validators import validateToolchain, ToolchainDefaults

    cxx = validateToolchain("amdclang++")
    iim = makeIsaInfoMap(SUPPORTED_ISA, cxx)
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    assembler = makeAssemblyToolchain(cxx, bundler, "default").assembler
    return assembler, iim


def get_assembler():
    return _toolchain()[0]


def get_isa_info_map():
    return _toolchain()[1]


# --- global-state isolation -------------------------------------------------
#
# Parsing a logic file / deriving a Solution mutates process-global state
# (``globalParameters`` and ``validParameters["ISA"]`` via
# ``assignGlobalParameters``). Left unrestored, that leaks into unrelated tests
# (e.g. ``test_validateParameterTypes`` compares a precomputed type map against
# a fresh one over ``validParameters``). Wrap every emit in this context so the
# harness is a no-op on shared state.


@contextlib.contextmanager
def _isolated_globals():
    from Tensile.Common.GlobalParameters import globalParameters
    from Tensile.Common.ValidParameters import validParameters

    saved_gp = copy.deepcopy(dict(globalParameters))
    saved_vp = copy.deepcopy(dict(validParameters))
    try:
        yield
    finally:
        globalParameters.clear()
        globalParameters.update(saved_gp)
        validParameters.clear()
        validParameters.update(saved_vp)


def _init_rocisa_for(kernel):
    """Initialize the rocIsa singleton for ``kernel``'s own ISA + wavefront.

    The singleton is process-global; whichever arch emitted last sets its
    state. Re-initializing per kernel makes each emit independent of order, so
    the goldens are stable whether a suite runs alone or after others.
    """
    import rocisa

    isa = tuple(kernel["ISA"])
    wavefront = kernel["WavefrontSize"]
    asmpath = shutil.which("amdclang++") or "/usr/bin/amdclang++"
    ri = rocisa.rocIsa.getInstance()
    ri.init(isa, asmpath)
    ri.setKernel(isa, wavefront)
    return ri


# --- solution / kernel emit -------------------------------------------------


def _solutions_from_logic_unguarded(logic_path):
    import Tensile.LibraryIO as L

    asm = get_assembler()
    lib = L.parseLibraryLogicFile(str(logic_path), asm, False, False, False, get_isa_info_map(), False)
    sols = lib.solutions
    return list(sols.values()) if isinstance(sols, dict) else list(sols)


def solutions_from_logic(logic_path):
    """Parse a logic YAML into a list of fully-derived ``Solution`` objects.

    Runs under global-state isolation so it does not leak into other tests.
    """
    with _isolated_globals():
        return _solutions_from_logic_unguarded(logic_path)


def _prepare_kernel(kernel, splitGSU=False):
    """Set the per-kernel fields ``writeSolutionsAndKernels`` sets before emit."""
    from Tensile.SolutionStructs.Naming import getKernelFileBase

    base = getKernelFileBase(splitGSU, kernel)
    kernel.duplicate = False
    kernel["BaseName"] = base
    return base


def emit_kernels_from_logic(logic_path, splitGSU=False, canonical=True, limit=None):
    """Emit assembly for every unique kernel produced by ``logic_path``.

    Returns a list of ``(basename, source, err)`` tuples, sorted by basename for
    stable ordering. ``source`` is canonicalized assembly text when
    ``canonical`` is True (the default). ``err`` is the emitter return code
    (0 == ok); a nonzero ``err`` is itself real covered behavior worth pinning.

    ``limit`` caps the number of kernels emitted (after a stable sort by kernel
    name) — used to draw a few representative kernels from very large tuned
    logic files (e.g. the StreamK corpus) without emitting thousands.
    """
    import rocisa
    from Tensile.TensileCreateLibrary.Run import (
        generateKernelObjectsFromSolutions,
        processKernelSource,
    )
    from Tensile.KernelWriterAssembly import KernelWriterAssembly
    from Tensile.Common.Types import DebugConfig
    from Tensile.SolutionStructs.Naming import getKernelFileBase

    asm = get_assembler()

    def _emit(kwa, kernel):
        ri = _init_rocisa_for(kernel)  # independent of any prior arch's state
        data = ri.getData()
        outOptions = ri.getOutputOptions()
        base = _prepare_kernel(kernel, splitGSU)
        res = processKernelSource(kwa, data, outOptions, splitGSU, kernel)
        src = res.src
        if canonical:
            src = canonicalize_asm(src)
        elif isinstance(src, (bytes, bytearray)):
            src = src.decode(errors="replace")
        return base, src, res.err

    results = []
    with _isolated_globals():
        sols = _solutions_from_logic_unguarded(logic_path)
        kernels = generateKernelObjectsFromSolutions(sols)
        if limit is not None:
            # stable subset by kernel name so the cap is deterministic
            kernels = sorted(kernels, key=lambda k: getKernelFileBase(splitGSU, k))[:limit]
        kwa = KernelWriterAssembly(asm, DebugConfig())

        # Steady-state warm-up: the emitter accumulates process-global scheduler
        # state (e.g. WMMA matrix-reuse tracking) so the very first emit in a
        # process differs from all subsequent ones. Production always emits in
        # the warm state; emit one throwaway kernel once so every *returned*
        # result is the stable steady-state, independent of suite ordering.
        global _WARMED
        if not _WARMED and kernels:
            _emit(kwa, kernels[0])
            _WARMED = True

        for kernel in kernels:
            results.append(_emit(kwa, kernel))

    results.sort(key=lambda t: t[0])
    return results


def emit_one(logic_path, index=0, canonical=True):
    """Convenience: emit a single kernel's (basename, source, err)."""
    return emit_kernels_from_logic(logic_path, canonical=canonical)[index]


def emit_helpers_from_logic(logic_path):
    """Emit the *helper* kernels (beta-only, conversion, activation-enum, …) for
    the solutions in ``logic_path``.

    These are the HIP C++ kernels written into Kernels.cpp alongside the asm
    GEMM kernels — a separate emit path (``KernelWriterBetaOnly`` /
    ``KernelWriterConversion`` / ``KernelWriterModules``) from the assembly one.
    Returns a sorted list of ``(name, err)`` tuples. The source is HIP C++ (no
    MMA scheduler state) but we still key the golden on identity + return code
    for consistency with the asm suites.
    """
    from Tensile.TensileCreateLibrary.Run import (
        generateKernelObjectsFromSolutions,
        generateKernelHelperObjects,
    )

    asm = get_assembler()
    out = []
    with _isolated_globals():
        sols = _solutions_from_logic_unguarded(logic_path)
        kernels = generateKernelObjectsFromSolutions(sols)
        khos = generateKernelHelperObjects(kernels, str(asm.path), get_isa_info_map())
        for ko in khos:
            name = ko.getKernelName()
            err, _src = ko.getSourceFileString()
            ko.getHeaderFileString()  # exercise header emit too
            out.append((name, err))
    return sorted(out)
