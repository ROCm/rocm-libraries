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

import functools
import re

# --- assembly canonicalization ---------------------------------------------

# The emitter tags branch/loop labels with a random 16-char [A-Z0-9] suffix
# (e.g. ``label_NoBranch_T8JHFHKM7BO5OHXW``). That suffix is the *only* source
# of run-to-run nondeterminism in the emitted text. We map each distinct random
# token to a stable sequential id by first-appearance order, which preserves the
# label<->reference correspondence while removing the randomness.
_RANDOM_LABEL_SUFFIX = re.compile(r"_[A-Z0-9]{16}\b")


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


# --- solution / kernel emit -------------------------------------------------


def solutions_from_logic(logic_path):
    """Parse a logic YAML into a list of fully-derived ``Solution`` objects."""
    import Tensile.LibraryIO as L

    asm = get_assembler()
    lib = L.parseLibraryLogicFile(str(logic_path), asm, False, False, False, get_isa_info_map(), False)
    sols = lib.solutions
    return list(sols.values()) if isinstance(sols, dict) else list(sols)


def _prepare_kernel(kernel, splitGSU=False):
    """Set the per-kernel fields ``writeSolutionsAndKernels`` sets before emit."""
    from Tensile.SolutionStructs.Naming import getKernelFileBase

    base = getKernelFileBase(splitGSU, kernel)
    kernel.duplicate = False
    kernel["BaseName"] = base
    return base


def emit_kernels_from_logic(logic_path, splitGSU=False, canonical=True):
    """Emit assembly for every unique kernel produced by ``logic_path``.

    Returns a list of ``(basename, source, err)`` tuples, sorted by basename for
    stable ordering. ``source`` is canonicalized assembly text when
    ``canonical`` is True (the default). ``err`` is the emitter return code
    (0 == ok); a nonzero ``err`` is itself real covered behavior worth pinning.
    """
    import rocisa
    from Tensile.TensileCreateLibrary.Run import (
        generateKernelObjectsFromSolutions,
        processKernelSource,
    )
    from Tensile.KernelWriterAssembly import KernelWriterAssembly
    from Tensile.Common.Types import DebugConfig

    asm = get_assembler()
    sols = solutions_from_logic(logic_path)
    kernels = generateKernelObjectsFromSolutions(sols)

    kwa = KernelWriterAssembly(asm, DebugConfig())
    data = rocisa.rocIsa.getInstance().getData()
    outOptions = rocisa.rocIsa.getInstance().getOutputOptions()

    results = []
    for kernel in kernels:
        base = _prepare_kernel(kernel, splitGSU)
        res = processKernelSource(kwa, data, outOptions, splitGSU, kernel)
        src = res.src
        if canonical:
            src = canonicalize_asm(src)
        elif isinstance(src, (bytes, bytearray)):
            src = src.decode(errors="replace")
        results.append((base, src, res.err))

    results.sort(key=lambda t: t[0])
    return results


def emit_one(logic_path, index=0, canonical=True):
    """Convenience: emit a single kernel's (basename, source, err)."""
    return emit_kernels_from_logic(logic_path, canonical=canonical)[index]
