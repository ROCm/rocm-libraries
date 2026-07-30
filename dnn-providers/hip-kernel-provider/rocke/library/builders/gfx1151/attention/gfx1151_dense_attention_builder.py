# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Single import surface for the gfx1151 transposed-QK WMMA FMHA (swapqk) kernel.

``swapqk`` is the production dense-attention forward kernel for gfx1151 (Strix
Halo, RDNA3.5). Build it from the frozen spec, which bakes the swept +
hardware-validated knobs:

    from builders.gfx1151.attention.gfx1151_dense_attention_builder import (
        WmmaFmhaSwapQKSpec, build_wmma_fmha_swapqk_fwd, wmma_fmha_swapqk_fwd_grid,
    )
    spec = WmmaFmhaSwapQKSpec(head_size=128, num_query_heads=24)
    kdef = build_wmma_fmha_swapqk_fwd(spec)          # -> KernelDef
    grid = wmma_fmha_swapqk_fwd_grid(spec, seqlen_q=16384, batch=1)

The default spec takes V pre-transposed as ``[B, H, D, S]`` -- relay a row-major
tensor with :func:`swapqk_transpose_v`, or pass ``v_transposed=False`` to build
against ``[B, S, H, D]`` at a measured cost of ~20% more cycles at L=4096.

Verify + time one shape on a gfx1151 board (also supports the compile-here /
run-there split via ``--emit`` / ``--prebuilt``)::

    PYTHONPATH=python python3 -m \
        builders.gfx1151.attention.wmma_fmha_swapqk_verify \
        --seqlen-q 2048 --seqlen-k 2048 --head-size 128 --heads 24 --batch 1

:class:`SwapQKCfg` / :func:`build_wmma_fmha_swapqk` expose every knob for A/B
work, including the documented dead-ends; each field records what it measured.
See ``README.md`` and ``ALGORITHM.md``.
"""

from __future__ import annotations

from kernels.gfx1151.wmma_fmha_swapqk import (  # noqa: F401
    SwapQKCfg,
    WmmaFmhaSwapQKSpec,
    build_wmma_fmha_swapqk,
    build_wmma_fmha_swapqk_fwd,
    is_valid_spec,
    swapqk_grid,
    swapqk_transpose_v,
    wmma_fmha_swapqk_fwd_grid,
)

__all__ = [
    # production API (frozen spec, winning knobs baked in)
    "WmmaFmhaSwapQKSpec",
    "build_wmma_fmha_swapqk_fwd",
    "wmma_fmha_swapqk_fwd_grid",
    "is_valid_spec",
    "swapqk_transpose_v",
    # research API (every knob)
    "SwapQKCfg",
    "build_wmma_fmha_swapqk",
    "swapqk_grid",
]
