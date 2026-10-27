# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Single import surface for the gfx1151 transposed-QK WMMA FMHA (swapqk) kernel.

``swapqk`` is the production dense-attention forward kernel for gfx1151
(RDNA3.5). There is one config type, :class:`SwapQKCfg`, whose defaults are the
swept + hardware-validated winner, so the production build is just::

    from builders.gfx1151.attention.gfx1151_dense_attention_builder import (
        SwapQKCfg, build_wmma_fmha_swapqk, swapqk_grid,
    )
    cfg = SwapQKCfg(head_size=128, num_query_heads=24)
    kdef = build_wmma_fmha_swapqk(cfg)                 # -> KernelDef
    grid = swapqk_grid(cfg, seqlen_q=16384, batch=1)

Every other field is an opt-in lever for A/B work, including the documented
dead-ends; each one records what it measured. Use :func:`is_valid_spec` as the
cheap static gate before building.

The default config takes V pre-transposed as ``[B, H, D, S]`` -- relay a
row-major tensor with :func:`swapqk_transpose_v`, or pass ``v_transposed=False``
to build against ``[B, S, H, D]`` at a measured cost of ~20% more cycles at
L=4096.

Verify + time one shape on a gfx1151 board (also supports the compile-here /
run-there split via ``--emit`` / ``--prebuilt``)::

    PYTHONPATH=python python3 -m \
        builders.gfx1151.attention.wmma_fmha_swapqk_verify \
        --seqlen-q 2048 --seqlen-k 2048 --head-size 128 --heads 24 --batch 1

See ``README.md`` and ``ALGORITHM.md``.
"""

from __future__ import annotations

from kernels.gfx1151.wmma_fmha_swapqk import (  # noqa: F401
    SwapQKCfg,
    build_wmma_fmha_swapqk,
    is_valid_spec,
    swapqk_grid,
    swapqk_transpose_v,
)

__all__ = [
    "SwapQKCfg",
    "is_valid_spec",
    "build_wmma_fmha_swapqk",
    "swapqk_grid",
    "swapqk_transpose_v",
]
