# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""LDS barrier placement in the interleaved GEMM demo.

Pure CPU (build the kernel, inspect the IR op order). The cooperative LDS stores mean a wave
overwrites cells that other waves read, so every overwrite of a buffer must follow a workgroup
barrier that comes after all waves' reads of it.
"""

from __future__ import annotations

from rocke.helpers.tiling.kernels.tiling_gemm_interleaved_demo import (
    build_interleaved_gemm,
)


def test_reg_prefetch_fences_the_prologue_read_before_trip_0_overwrites_it() -> None:
    # reg_prefetch reads tile 0 from LDS buffer 0 in the prologue; trip 0 then stores tile 2 into
    # buffer 0. Without a barrier between them, a fast wave can overwrite tile 0 before a slow
    # wave has read it.
    kernel, _ = build_interleaved_gemm(
        256,
        256,
        256,
        arch="gfx90a",
        tile_m=128,
        tile_n=128,
        waves_m=2,
        waves_n=2,
        reg_prefetch=True,
    )
    names = [op.name for op in kernel.body.ops]
    loop = names.index("scf.for")
    last_read = max(
        i for i, n in enumerate(names[:loop]) if n.startswith("tile.smem_load")
    )
    assert "tile.sync_lds_only" in names[last_read + 1 : loop], names[last_read:loop]
