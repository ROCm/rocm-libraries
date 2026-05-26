# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Validation helpers for gemm_decode tile/trait configurations.

The full sweep matrix lands in P1+. P0 only ships this stub so the directory
layout matches `tile_engine/ops/gemm/` and so callers can already import the
expected entry points without breaking once codegen is wired in.
"""

from __future__ import annotations

from typing import Any, Dict


WARP_SIZE = 64  # gfx9; gfx11/12 wave32 paths land in P3+.


_PER_TENSOR_LAYOUTS = {"per_tensor", "PerTensor"}
_UNSCALED_LAYOUTS   = {"void", "unscaled", ""}
_BLOCK2D_PREFIXES   = ("block2d", "Block2D")


def _is_unscaled(value: Any) -> bool:
    return str(value or "").lower() in {v.lower() for v in _UNSCALED_LAYOUTS}


def _is_per_tensor(value: Any) -> bool:
    return str(value or "") in _PER_TENSOR_LAYOUTS


def _is_block2d(value: Any) -> bool:
    return str(value or "").startswith(_BLOCK2D_PREFIXES)


def is_tile_config_valid(tile: Dict[str, Any]) -> bool:
    """Return True iff the per-config entry satisfies the P0/P0b invariants.

    The same checks are enforced at runtime by
    `GemmDecodeUniversalKernel::IsSupportedArgument`; this Python copy lets the
    instance builder prune the sweep before invoking the HIP compiler.
    """
    vec = int(tile.get("vector_size", 0))
    if vec <= 0:
        return False
    lanes = int(tile.get("lanes_per_output", WARP_SIZE))
    if lanes != WARP_SIZE:
        return False
    if int(tile.get("m_per_warp", 1)) != 1:
        return False
    if int(tile.get("n_per_warp", 1)) != 1:
        return False
    if int(tile.get("warps_per_block", 1)) != 1:
        return False
    if tile.get("output_axis", "smallM") != "smallM":
        return False
    if bool(tile.get("use_packed_fp32", False)):
        return False  # alternate gfx950 numerics path lands in P4.
    return True


def is_trait_combination_valid(trait: Dict[str, Any]) -> bool:
    if trait.get("pipeline", "decode") != "decode":
        return False
    if trait.get("scheduler", "intrawave") != "intrawave":
        return False
    if trait.get("epilogue", "default") not in {"default", "atomic_add"}:
        return False
    split_k = int(trait.get("split_k", 1))
    if split_k not in {1, 2, 4, 8}:
        return False
    if bool(trait.get("persistent", False)):
        return False  # persistent-loop variants land in P4.

    x_scale = trait.get("x_scale_layout", "void")
    w_scale = trait.get("w_scale_layout", "void")
    # Allowed combinations:
    #   (unscaled, unscaled)         P0
    #   (PerTensor, PerTensor)       P0b
    #   (Block2D<.,.>, Block2D<.,.>) P1   (handled by the blockscale family)
    if _is_unscaled(x_scale) and _is_unscaled(w_scale):
        pass
    elif _is_per_tensor(x_scale) and _is_per_tensor(w_scale):
        pass
    elif _is_block2d(x_scale) and _is_block2d(w_scale):
        # The universal family does not own blockscale; the blockscale
        # family validator should accept these instead.
        return False
    else:
        return False

    return True


def k_alignment_ok(K: int, vector_size: int, split_k: int) -> bool:
    return K % (WARP_SIZE * vector_size * split_k) == 0


__all__ = [
    "is_tile_config_valid",
    "is_trait_combination_valid",
    "k_alignment_ok",
    "WARP_SIZE",
]
