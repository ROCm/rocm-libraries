# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared helpers for unit tests that emit a kernel's assembly on the CPU and read it."""

import re
import shutil


def emit_asm(isa_info_map, assembler, params):
    """(solution, assembly text or None) of the kernel ``params`` describes. CPU-only; no GPU is touched.

    The text is None when Solution rejects ``params``.
    """
    import rocisa
    from tensilelite.Common.Types import DebugConfig
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly
    from tensilelite.SolutionStructs.Naming import getKernelFileBase
    from tensilelite.SolutionStructs.Solution import Solution
    from tensilelite.create_library.run import generateKernelObjectsFromSolutions, processKernelSource
    from tensilelite.Tests.rocisa_test_state import preserve_rocisa_kernel_state

    sol = Solution(params, False, True, False, assembler, isa_info_map)
    if not sol.get("Valid"):
        return sol, None
    with preserve_rocisa_kernel_state():
        kwa = KernelWriterAssembly(assembler, DebugConfig())
        pieces = []
        for kernel in generateKernelObjectsFromSolutions([sol]):
            ri = rocisa.rocIsa.getInstance()
            ri.init(tuple(kernel["ISA"]), shutil.which("amdclang++") or "/usr/bin/amdclang++")
            ri.setKernel(tuple(kernel["ISA"]), kernel["WavefrontSize"])
            kernel.duplicate = False
            kernel["BaseName"] = getKernelFileBase(False, kernel)
            src = processKernelSource(kwa, ri.getData(), ri.getOutputOptions(), False, kernel).src
            if isinstance(src, (bytes, bytearray)):
                src = src.decode(errors="replace")
            pieces.append(src or "")
    return sol, "\n".join(pieces)


def assert_every_vector_preload_barriers(asm):
    """Each vector preload barriers between its last label_Load_Bias and its first store."""
    preloads = [s for s in re.split(r"store (?:bias|scaleAlpha)\b", asm)[:-1] if "buffer_load" in s]
    assert preloads
    for preload in preloads:
        assert "s_barrier_wait" in preload[preload.rindex("label_Load_Bias"):]
