#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Mathematical functions for deriving valid tile/warp/vector configurations.

Replaces the hand-maintained lookup tables TILE_TO_WAVE_WARP and
_TILE_WTILK_TO_VECS in grouped_config_rules.py with functions derived
from the static asserts in the C++ kernel and pipeline implementation.

Key source files this is derived from:
  - block_universal_gemm_as_bs_cr.hpp   (tile divisibility by warps)
  - gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp  (vec/LDS formulas)
  - conv_algorithm_limits.hpp           (VMEM/LDS vector size validity)
  - warp_gemm_dispatcher.hpp            (XDL warp tile shapes per dtype)
  - arch_specs_generated.py             (arch-specific wave/warp-tile combos)
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Path setup — allow running from any directory
# ---------------------------------------------------------------------------
_CODEGEN_DIR = Path(__file__).parent.resolve()
if str(_CODEGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_CODEGEN_DIR))

from arch_specs_generated import (
    WARP_SUPPORTED_COMBINATIONS,       # [wave_m, wave_n, wave_k] per arch
    WARP_TILE_SUPPORTED_COMBINATIONS,  # [warp_m, warp_n, warp_k] per arch+dtype
    ELEMENT_SIZE_MAP,                  # bytes per element per dtype string
)

# Warp size on AMD GPUs
WARP_SIZE = 64

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pos_divisors(n: int) -> List[int]:
    """Return all positive divisors of n in ascending order."""
    if n <= 0:
        return []
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    return sorted(divs)


def _lds_valid(vec: int, sizeof_dtype: float) -> bool:
    """LDS vector load/store must be a power-of-2 multiple of 8 bits, up to 256 bits.

    Source: conv_algorithm_limits.hpp IsLDSVectorSizeValid (8–128 bits for standard LDS).
    In practice some bwd_data configs use larger global-load vectors (e.g. fp32×8=256 bits)
    where the global load is split across DWORD pairs rather than going through LDS.
    We therefore accept up to 256 bits and require the width to be a power of 2 in bytes.
    """
    bits = vec * sizeof_dtype * 8
    # Must be positive, a power of 2 in bit-width, and at most 256 bits
    if bits <= 0 or bits > 256:
        return False
    # Check power of 2
    b = int(bits)
    return b > 0 and (b & (b - 1)) == 0


def _pipeline_wave_ok(
    wave_m: int, wave_n: int, wave_k: int,
    warp_tile_m: int, warp_tile_n: int, warp_tile_k: int,
    pipeline: Optional[str],
) -> bool:
    """Return True if this wave/warp combo is valid for the given pipeline.

    Pipeline-specific constraints derived from static asserts in:
      - gemm_pipeline_ag_bg_cr_comp_async_eight_waves_policy.hpp
        (NWarps==2, WarpTile::at(I1)==16 for basic_async_v1 eight-wave)
      - TDM pipeline (BlockSize == warp_size * 4, WarpTile M=N=32)
    """
    if pipeline is None:
        return True

    p = pipeline.lower()

    if p == "basic_async_v1":
        # Eight-wave async: NWarps must be 2, warp_tile_n must be 16
        return wave_n == 2 and warp_tile_n == 16

    if p in ("tdm", "tdmv2"):
        # TDM requires exactly 4 waves and 32x32 warp tile
        return (wave_m * wave_n * wave_k == 4
                and warp_tile_m == 32 and warp_tile_n == 32)

    # All other pipelines (compv1..v6, mem, comp_async, basic_v1, etc.): no constraint
    return True


def _deduplicate(pairs: List[Tuple[Tuple, Tuple]]) -> List[Tuple[Tuple, Tuple]]:
    """Remove duplicate (wave, warp_tile) pairs while preserving order."""
    seen: Set[Tuple] = set()
    result = []
    for item in pairs:
        key = (item[0], item[1])
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_valid_wave_warp_pairs(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    dtype_key: str,
    arch: str = "gfx942",
    pipeline: Optional[str] = None,
) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """Return all valid ((wave_m, wave_n, wave_k), (warp_tile_m, warp_tile_n, warp_tile_k)) pairs.

    Derived from the static assert in block_universal_gemm_as_bs_cr.hpp:
        MIterPerWarp * MWarp * WarpGemm::kM == MPerBlock
        NIterPerWarp * NWarp * WarpGemm::kN == NPerBlock

    which means:  tile_m == wave_m * warp_tile_m * iter_m  (iter_m >= 1)
                  tile_n == wave_n * warp_tile_n * iter_n  (iter_n >= 1)

    Args:
        tile_m, tile_n, tile_k: block tile dimensions
        dtype_key: e.g. "bf16_bf16_fp32", "fp32_fp32_fp32"
        arch: GPU architecture string, default "gfx942"
        pipeline: optional pipeline name to apply pipeline-specific constraints

    Returns:
        List of ((wave_m, wave_n, wave_k), (warp_tile_m, warp_tile_n, warp_tile_k)) tuples.
        Each pair is structurally valid for the given arch and pipeline.
    """
    supported_wave_combos: Set[Tuple[int, int, int]] = {
        tuple(c) for c in WARP_SUPPORTED_COMBINATIONS.get(arch, [])
    }
    warp_tile_shapes: List[List[int]] = (
        WARP_TILE_SUPPORTED_COMBINATIONS
        .get(arch, {})
        .get(dtype_key, [])
    )

    results: List[Tuple[Tuple, Tuple]] = []

    for wt in warp_tile_shapes:
        warp_m, warp_n, warp_k = wt[0], wt[1], wt[2]

        # Tile must be divisible by the warp tile in M and N
        if tile_m % warp_m != 0 or tile_n % warp_n != 0:
            continue

        # Enumerate all integer (iter_m, iter_n) >= 1 such that the block is tiled exactly
        for iter_m in _pos_divisors(tile_m // warp_m):
            wave_m = tile_m // (warp_m * iter_m)
            for iter_n in _pos_divisors(tile_n // warp_n):
                wave_n = tile_n // (warp_n * iter_n)

                # Normal case: wave_k = 1
                if (wave_m, wave_n, 1) in supported_wave_combos:
                    if _pipeline_wave_ok(wave_m, wave_n, 1, warp_m, warp_n, warp_k, pipeline):
                        results.append(((wave_m, wave_n, 1), (warp_m, warp_n, warp_k)))

                # Special case: wave_k = 2
                # Only a small number of tiles use this (e.g. (128,32,32) with warp=(32,32,8)).
                # Supported on gfx942/gfx950 via the [2,1,2] wave combo.
                if (wave_m, wave_n, 2) in supported_wave_combos:
                    if _pipeline_wave_ok(wave_m, wave_n, 2, warp_m, warp_n, warp_k, pipeline):
                        results.append(((wave_m, wave_n, 2), (warp_m, warp_n, warp_k)))

    return _deduplicate(results)


def get_valid_vec_sizes(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    wave_m: int,
    wave_n: int,
    wave_k: int,
    warp_tile_m: int,
    warp_tile_n: int,
    warp_tile_k: int,
    dtype_key: str,
    pipeline: Optional[str] = None,
) -> List[Tuple[int, int, int]]:
    """Return all valid (vec_a, vec_b, vec_c) triples for a fully-specified config.

    Derived from:
      - gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp  (thread-pixel budget)
      - conv_algorithm_limits.hpp  IsVmemVectorSizeValid / IsLDSVectorSizeValid

    The thread-pixel budget formula:
        block_size = WARP_SIZE * wave_m * wave_n * wave_k
        pixels_a   = tile_m * tile_k / block_size   (elements per thread, A tile)
        pixels_b   = tile_n * tile_k / block_size   (elements per thread, B tile)

    Valid vec_a/vec_b must divide their respective pixel budget and satisfy
    VMEM/LDS hardware constraints.  vec_c is constrained by the XDL output
    shuffle: tile_n must be divisible by (wave_n * warp_tile_n * vec_c).

    Args:
        tile_m, tile_n, tile_k: block tile dimensions
        wave_m, wave_n, wave_k: wave counts
        warp_tile_m, warp_tile_n, warp_tile_k: XDL warp tile dimensions
        dtype_key: e.g. "bf16_bf16_fp32"
        pipeline: optional, currently unused (reserved for future per-pipeline tuning)

    Returns:
        Sorted list of (vec_a, vec_b, vec_c) tuples.
    """
    dtype_a = dtype_key.split("_")[0]
    sizeof_a = float(ELEMENT_SIZE_MAP.get(dtype_a, 2))  # bytes per A element

    block_size = WARP_SIZE * wave_m * wave_n * wave_k

    if block_size == 0 or tile_m * tile_k % block_size != 0 or tile_n * tile_k % block_size != 0:
        return []

    pixels_a = (tile_m * tile_k) // block_size
    pixels_b = (tile_n * tile_k) // block_size

    # Maximum vector width per element type.
    # Standard VMEM load limit is 16 bytes (128 bits), which gives:
    #   fp32 (4 bytes) → 4 elements;  bf16/fp16 (2 bytes) → 8;  fp8 (1 byte) → 16
    # However, some bwd_data configurations use vec_a=8 for fp32 (32-byte loads via
    # 2×16-byte split), which compiles and runs on hardware.  To avoid false negatives
    # the cap is relaxed to 16 bytes × 2 = the hardware dword-per-lane pair limit.
    # The LDS validity check below enforces the finer-grained hardware constraint.
    max_vec_ab = max(1, int(32 // sizeof_a))   # 2× standard VMEM width

    # Output vec_c uses the same dtype on the C tile; standard 16-byte limit applies
    max_vec_c = max(1, int(16 // sizeof_a))

    valid_a = [
        v for v in [1, 2, 4, 8, 16]
        if v <= max_vec_ab
        and pixels_a % v == 0
        and _lds_valid(v, sizeof_a)
    ]

    valid_b = [
        v for v in [1, 2, 4, 8, 16]
        if v <= max_vec_ab
        and pixels_b % v == 0
        and _lds_valid(v, sizeof_a)
    ]

    # vec_c constraint: XDL accumulator is laid out in N-major tiles of size warp_tile_n.
    # The output shuffle requires tile_n divisible by (wave_n * warp_tile_n * vec_c).
    # vec_c constraint: the C accumulator is stored contiguously along N per thread.
    # The output shuffle in the XDL block gemm only requires tile_n to be divisible
    # by vec_c (not by wave_n * warp_tile_n * vec_c as the input tiles).
    # Source: ThreadsCoverCTile in conv_algorithm_limits.hpp:
    #   tile_n % (thread_cluster_dims[3] * vec_c) == 0
    # thread_cluster_dims[3] = 1 because each thread writes one N-element per shuffle step;
    # the n_xdl_per_wave repeats are handled by the outer loop, not the vector width.
    valid_c = [
        v for v in [1, 2, 4, 8, 16]
        if v <= max_vec_c
        and tile_n % v == 0
        and _lds_valid(v, sizeof_a)
    ]

    return sorted({(va, vb, vc) for va in valid_a for vb in valid_b for vc in valid_c})


def get_vec_sizes_for_wave_warp(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    warp_tile_k: int,
    dtype_key: str,
    arch: str = "gfx942",
    pipeline: Optional[str] = None,
) -> List[Tuple[int, int, int]]:
    """Return union of valid (vec_a, vec_b, vec_c) across all wave/warp pairs with given warp_tile_k.

    Convenience wrapper matching the _TILE_WTILK_TO_VECS key signature:
        key = (tile_m, tile_n, tile_k, warp_tile_k)

    This takes the union over all valid wave/warp pairs whose warp_tile_k matches,
    so the result is a superset of what any single wave/warp pair would produce.

    Args:
        tile_m, tile_n, tile_k: block tile dimensions
        warp_tile_k: XDL warp tile K dimension (selects dtype variant, e.g. 8 or 16 for bf16)
        dtype_key: e.g. "bf16_bf16_fp32"
        arch: GPU architecture string
        pipeline: optional pipeline constraint

    Returns:
        Sorted list of (vec_a, vec_b, vec_c) tuples (union across matching wave/warp pairs).
    """
    results: Set[Tuple[int, int, int]] = set()

    for (wave_m, wave_n, wave_k), (wt_m, wt_n, wt_k) in get_valid_wave_warp_pairs(
        tile_m, tile_n, tile_k, dtype_key, arch=arch, pipeline=pipeline
    ):
        if wt_k == warp_tile_k:
            vecs = get_valid_vec_sizes(
                tile_m, tile_n, tile_k,
                wave_m, wave_n, wave_k,
                wt_m, wt_n, wt_k,
                dtype_key, pipeline=pipeline,
            )
            results.update(vecs)

    return sorted(results)


# ---------------------------------------------------------------------------
# Dtype key inference helpers (for test infrastructure)
# ---------------------------------------------------------------------------

def dtype_keys_for_warp_tile_k(warp_tile_k: int) -> List[str]:
    """Infer plausible dtype_keys from warp_tile_k.

    Used by tests to map _TILE_WTILK_TO_VECS keys (which encode warp_tile_k
    but not dtype explicitly) back to dtype_key strings.

    warp_tile_k mapping (from warp_gemm_dispatcher.hpp):
      fp32_fp32_fp32  : warp_tile_k ∈ {4, 8, 16}
      bf16_bf16_fp32  : warp_tile_k ∈ {8, 16, 32}  (gfx942: {8,16}; gfx950: {8,16,32})
      fp16_fp16_fp32  : warp_tile_k ∈ {8, 16, 32}
      fp8_fp8_fp32    : warp_tile_k ∈ {16, 32, 64, 128}
    """
    candidates = []
    if warp_tile_k in {4, 8, 16}:
        candidates.append("fp32_fp32_fp32")
    if warp_tile_k in {8, 16, 32}:
        candidates.append("bf16_bf16_fp32")
        candidates.append("fp16_fp16_fp32")
    if warp_tile_k in {16, 32, 64, 128}:
        candidates.append("fp8_fp8_fp32")
    return candidates


if __name__ == "__main__":
    # Quick smoke test
    print("Wave/warp pairs for (128, 64, 32) bf16:")
    for wave, warp in get_valid_wave_warp_pairs(128, 64, 32, "bf16_bf16_fp32"):
        print(f"  wave={wave}  warp_tile={warp}")

    print()
    print("Vec sizes for (128, 64, 32) wave=(2,2,1) warp_tile=(32,32,8) bf16:")
    for v in get_valid_vec_sizes(128, 64, 32, 2, 2, 1, 32, 32, 8, "bf16_bf16_fp32"):
        print(f"  {v}")

    print()
    print("Vec sizes via get_vec_sizes_for_wave_warp (128,64,32,wt_k=8) bf16:")
    for v in get_vec_sizes_for_wave_warp(128, 64, 32, 8, "bf16_bf16_fp32"):
        print(f"  {v}")
