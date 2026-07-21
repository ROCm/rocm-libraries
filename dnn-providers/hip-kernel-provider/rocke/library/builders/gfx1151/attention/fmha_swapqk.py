# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Back-compat shim: the transposed-QK WMMA FMHA kernel now lives in
``kernels.gfx1151.wmma_fmha_swapqk`` (it graduated from research builder to
production kernel). This module re-exports the full research API so the tuning
harness (``sq_tune``, ``rocprof_swapqk_runner``) keeps importing from here.

New code should import the production entry point directly:

    from kernels.gfx1151.wmma_fmha_swapqk import (
        WmmaFmhaSwapQKSpec, build_wmma_fmha_swapqk_fwd, wmma_fmha_swapqk_fwd_grid,
    )
"""

from __future__ import annotations

from kernels.gfx1151.wmma_fmha_swapqk import (  # noqa: F401
    SwapQKCfg,
    WmmaFmhaSwapQKSpec,
    build_wmma_fmha_swapqk,
    build_wmma_fmha_swapqk_fwd,
    is_valid_spec,
    swapqk_grid,
    wmma_fmha_swapqk_fwd_grid,
)
