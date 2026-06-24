# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Validation helpers for gemm_decode tile/trait configurations.

Mirrors `GemmDecodeUniversalKernel::IsSupportedArgument` so the instance builder
can prune the sweep before invoking the HIP compiler. Covers the register-tile
knobs, the XCD chiplet swizzle, and the P0 wvSplitKQ recipe levers
(warps_per_block / stage_a_in_lds / stream_b / persistent).
"""

from __future__ import annotations

from typing import Any, Dict


WARP_SIZE = 64  # gfx9; gfx11/12 wave32 paths land in P3+.


_PER_TENSOR_LAYOUTS = {"per_tensor", "PerTensor"}
_PER_TOKEN_LAYOUTS  = {"per_token", "PerToken"}
_UNSCALED_LAYOUTS   = {"void", "unscaled", ""}
_BLOCK2D_PREFIXES   = ("block2d", "Block2D")


def _is_unscaled(value: Any) -> bool:
    return str(value or "").lower() in {v.lower() for v in _UNSCALED_LAYOUTS}


def _is_per_tensor(value: Any) -> bool:
    return str(value or "") in _PER_TENSOR_LAYOUTS


def _is_per_token(value: Any) -> bool:
    return str(value or "") in _PER_TOKEN_LAYOUTS


def _is_block2d(value: Any) -> bool:
    return str(value or "").startswith(_BLOCK2D_PREFIXES)


# Canonical chiplet defaults that the instance headers fall back to when the
# swizzle is off. Used to dedupe the sweep: with chiplet_swizzle=False the
# chunk_size / num_xcds template params are unused, so every (chunk, num_xcds)
# pair would otherwise enumerate an identical kernel.
_CHIPLET_DEFAULT_NUM_XCDS = 8
_CHIPLET_DEFAULT_CHUNK     = 8
_CHIPLET_VALID_NUM_XCDS    = {1, 8}
_CHIPLET_VALID_CHUNK       = {4, 8, 16, 32, 64}


def _is_chiplet_swizzle_valid(tile: Dict[str, Any]) -> bool:
    """Validate (and dedupe) the XCD-aware workgroup-swizzle knobs.

    `chiplet_swizzle` toggles the remap; `chiplet_num_xcds` / `chiplet_chunk_size`
    are only meaningful when it is on, so when it is off we accept only the
    canonical defaults to avoid emitting duplicate kernels.
    """
    swizzle = bool(tile.get("chiplet_swizzle", False))
    num_xcds = int(tile.get("chiplet_num_xcds", _CHIPLET_DEFAULT_NUM_XCDS))
    chunk = int(tile.get("chiplet_chunk_size", _CHIPLET_DEFAULT_CHUNK))
    if num_xcds not in _CHIPLET_VALID_NUM_XCDS:
        return False
    if chunk not in _CHIPLET_VALID_CHUNK:
        return False
    if not swizzle:
        # Off: collapse to the single canonical (num_xcds, chunk) instance.
        return num_xcds == _CHIPLET_DEFAULT_NUM_XCDS and chunk == _CHIPLET_DEFAULT_CHUNK
    # On: a single-XCD remap is a no-op, so require a real multi-die config.
    return num_xcds > 1


# P0 wvSplitKQ recipe levers (all compile-time Problem flags). gfx9 fat-WG
# occupancy tops out around 16 warps/block; 1 means the plain warp-per-output
# path. The multi-warp values share the A row through LDS.
_VALID_WARPS_PER_BLOCK = {1, 4, 8, 16}


def _is_recipe_combo_valid(tile: Dict[str, Any]) -> bool:
    """Validate (and dedupe) the P0 wvSplitKQ-recipe levers.

    The four levers are compile-time Problem flags:
      - warps_per_block : warps cooperating on one output tile (LDS A-share)
      - stage_a_in_lds  : stage the shared A row(s) through LDS (WD-OPT-21)
      - stream_b        : non-temporal / cache-bypassing B loads
      - persistent      : 1 WG/CU fat-WG launch that loops over the tile space

    These mirror the kernel's compile-time static_asserts; we additionally scope
    them to the decode band (m_per_warp == n_per_warp == 1) so the levers — which
    target the skinny M=1..4 shapes where register tiling is off — don't multiply
    the register-tile matrix they never benefit.
    """
    wpb = int(tile.get("warps_per_block", 1))
    if wpb not in _VALID_WARPS_PER_BLOCK:
        return False
    stage_a_in_lds = bool(tile.get("stage_a_in_lds", False))
    stream_b = bool(tile.get("stream_b", False))
    persistent = bool(tile.get("persistent", False))
    m_per_warp = int(tile.get("m_per_warp", 1))
    n_per_warp = int(tile.get("n_per_warp", 1))

    multi_warp = wpb > 1
    decode_band = m_per_warp == 1 and n_per_warp == 1

    # Kernel static_assert: a multi-warp workgroup cooperates on one output tile,
    # so A-in-LDS staging only makes sense when there is more than one warp.
    if stage_a_in_lds and not multi_warp:
        return False
    # Sweep scoping: every recipe lever targets the decode band, so reject any
    # recipe combined with a register tile (mp*np > 1) to bound the matrix.
    any_recipe = multi_warp or stage_a_in_lds or stream_b or persistent
    if any_recipe and not decode_band:
        return False
    return True


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
    if vec not in {8, 16}:
        return False
    m_per_warp = int(tile.get("m_per_warp", 1))
    n_per_warp = int(tile.get("n_per_warp", 1))
    # M-tile register reuse / B-reuse (A4): each warp computes a
    # kMPerWarp x kNPerWarp tile, loading every B row once and reusing it
    # across the kMPerWarp A rows held in VGPRs. The runtime grid is
    # ceil(M / m_per_warp) with a masked tail, so any M is valid.
    if m_per_warp not in {1, 2, 4, 8}:
        return False
    # N-tile register reuse (A1): one warp emits n_per_warp adjacent columns by
    # loading the shared A row once. The runtime N % n_per_warp == 0 check is in
    # IsSupportedArgument; here we just bound the compile-time fan-out.
    if n_per_warp not in {1, 2, 4}:
        return False
    # Cap the combined register tile (kMPerWarp * kNPerWarp accumulators plus
    # operands) so codegen never emits configs that spill VGPRs. mp*np <= 16
    # matches the R1 bench sweep (§15.C).
    if m_per_warp * n_per_warp > 16:
        return False
    if not _is_recipe_combo_valid(tile):
        return False
    if tile.get("output_axis", "smallM") != "smallM":
        return False
    if bool(tile.get("use_packed_fp32", False)):
        return False  # alternate gfx950 numerics path lands in P4.
    if not _is_chiplet_swizzle_valid(tile):
        return False
    return True


def is_trait_combination_valid(trait: Dict[str, Any], family: str = "universal") -> bool:
    """Validate a trait config row.

    `family` selects which scale-layout combinations are accepted:
      - "universal" : (unscaled, unscaled), (PerTensor, PerTensor),
                      (PerToken, PerTensor)                          (P0/P0b/P3)
      - "blockscale": (Block2D<.,.>, Block2D<.,.>)                  (P1)
    """
    if trait.get("pipeline", "decode") != "decode":
        return False
    if trait.get("scheduler", "intrawave") != "intrawave":
        return False
    if trait.get("epilogue", "default") not in {"default", "atomic_add"}:
        return False
    split_k = int(trait.get("split_k", 1))
    if split_k not in {1, 2, 4, 8}:
        return False

    x_scale = trait.get("x_scale_layout", "void")
    w_scale = trait.get("w_scale_layout", "void")

    if family == "universal":
        if _is_unscaled(x_scale) and _is_unscaled(w_scale):
            pass
        elif _is_per_tensor(x_scale) and _is_per_tensor(w_scale):
            pass
        elif _is_per_token(x_scale) and _is_per_tensor(w_scale):
            # Per-token activation quant: X is an [M] scale vector, W per-tensor.
            pass
        else:
            return False
    elif family == "blockscale":
        if not (_is_block2d(x_scale) and _is_block2d(w_scale)):
            return False
    else:
        raise ValueError(f"Unknown family for is_trait_combination_valid: {family!r}")

    return True


def k_alignment_ok(K: int, vector_size: int, split_k: int) -> bool:
    return K % (WARP_SIZE * vector_size * split_k) == 0


__all__ = [
    "is_tile_config_valid",
    "is_trait_combination_valid",
    "k_alignment_ok",
    "WARP_SIZE",
]
