# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Workflow harness for the gfx1151 transposed-QK WMMA FMHA (swapqk) kernel.

Single import surface for the build / verify / tune / benchmark loop, decoupled
from the unified ``benchmark`` CLI driver.

Production kernel (winning knobs baked in):

    from builders.gfx1151.attention.harness import (
        WmmaFmhaSwapQKSpec, build_wmma_fmha_swapqk_fwd, wmma_fmha_swapqk_fwd_grid,
    )
    spec = WmmaFmhaSwapQKSpec(head_size=128, num_query_heads=24)
    kdef = build_wmma_fmha_swapqk_fwd(spec)          # -> KernelDef

Verify + time one config on-device:

    from builders.gfx1151.attention.harness import verify_and_time_swapqk, Shape
    r = verify_and_time_swapqk(spec.to_cfg(), Shape(1, 24, 24, 2048, 2048, 128, False))

Cross-machine board workflow (compile on any host, run on the gfx1151 board):
the dense kernel compiles host-side (comgr targets gfx1151 regardless of the
build host's GPU) but must EXECUTE on gfx1151, so the unified ``benchmark`` driver
splits the two phases via ``--emit <dir>`` (compile hsaco, no GPU needed) and
``--prebuilt <dir>`` (load + verify + time on the board). ``--warmup`` /
``--iters`` scale the timing loop (dense attention is O(L^2): use small iters +
``--no-verify`` for long sequences). Config fields are swept with
``--grid FIELD=v1,v2`` / pinned with ``--set FIELD=v``. See ``README.md``.
"""

from __future__ import annotations

# Production kernel API (from the kernels/ tree).
from kernels.gfx1151.wmma_fmha_swapqk import (  # noqa: F401
    SwapQKCfg,
    WmmaFmhaSwapQKSpec,
    build_wmma_fmha_swapqk,
    build_wmma_fmha_swapqk_fwd,
    is_valid_spec,
    swapqk_grid,
    wmma_fmha_swapqk_fwd_grid,
)

# Long-sequence production candidates.
from kernels.gfx1151.wmma_fmha_swapqk_persistent import (  # noqa: F401
    PersistentCfg,
    build_wmma_fmha_persistent,
    persistent_grid,
)
from kernels.gfx1151.wmma_fmha_multiwave import (  # noqa: F401
    MultiWaveCfg,
    build_wmma_fmha_multiwave,
    multiwave_grid,
)

# Unified verify+time driver (per-kernel wrappers) and the shape descriptor.
from .benchmark import (  # noqa: F401
    KERNELS,
    Shape,
    verify_and_time,
    verify_and_time_multiwave,
    verify_and_time_persistent,
    verify_and_time_swapqk,
)

__all__ = [
    "WmmaFmhaSwapQKSpec",
    "build_wmma_fmha_swapqk_fwd",
    "wmma_fmha_swapqk_fwd_grid",
    "is_valid_spec",
    "SwapQKCfg",
    "build_wmma_fmha_swapqk",
    "swapqk_grid",
    "PersistentCfg",
    "build_wmma_fmha_persistent",
    "persistent_grid",
    "MultiWaveCfg",
    "build_wmma_fmha_multiwave",
    "multiwave_grid",
    "KERNELS",
    "Shape",
    "verify_and_time",
    "verify_and_time_swapqk",
    "verify_and_time_persistent",
    "verify_and_time_multiwave",
]
