# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Toolchain-free direct HSACO emission for the first qualified GEMM slice."""

from __future__ import annotations

import time

from ..core.arch import ArchTarget
from ..core.ir_print import print_ir
from ..core.ir_serialize import serialize
from ..core.passes import PassStats
from ..instances.gfx1151.wmma_gemm import WmmaGemmSpec, build_wmma_gemm
from ..runtime.code_object import (
    build_direct_gfx1151_wmma_gemm_hsaco,
)
from .compile import KernelArtifact

__all__ = ["compile_gfx1151_wmma_gemm_direct"]


def compile_gfx1151_wmma_gemm_direct(
    *,
    capture_ir_text: bool = True,
    name: str = "rocke_direct_wmma_gemm",
    block_x_is_m: bool = True,
) -> KernelArtifact:
    """Compile the gfx1151 f16 RCR WMMA GEMM with the direct Rust backend.

    This strict proof slice supports runtime M/N/K values that are positive
    multiples of 16. It has one wave32 block per 16x16 output tile; ``name`` and
    ``block_x_is_m`` select the generated symbol and MN/NM grid mapping. It does
    not lower through LLVM and does not load COMGR, invoke a toolchain, or
    consult a code-object cache.
    """
    started = time.perf_counter()
    spec = WmmaGemmSpec(name=name, block_x_is_m=block_x_is_m)
    kernel = build_wmma_gemm(spec, arch="gfx1151")
    built = time.perf_counter()
    serialized_ir = serialize(kernel)
    ir_text = print_ir(kernel) if capture_ir_text else ""
    printed = time.perf_counter()
    hsaco, code_object_timings, _compiler = build_direct_gfx1151_wmma_gemm_hsaco(
        serialized_ir
    )
    finished = time.perf_counter()
    return KernelArtifact(
        kernel=kernel,
        ir_text=ir_text,
        llvm_text="",
        hsaco=hsaco,
        timings={
            "ir_opt": 0.0,
            "ir_build": (built - started) * 1000.0,
            "ir_print": (printed - built) * 1000.0,
            "ir_lower_llvm": 0.0,
            "codeobj_bc": 0.0,
            "codeobj_relocatable": 0.0,
            "codeobj_executable": code_object_timings.executable * 1000.0,
            "direct_hsaco": (finished - printed) * 1000.0,
            "total": (finished - started) * 1000.0,
        },
        pass_stats=PassStats(),
        isa=ArchTarget.from_gfx("gfx1151").isa_triple,
    )
