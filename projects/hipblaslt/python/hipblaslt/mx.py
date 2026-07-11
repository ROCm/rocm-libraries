# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""MX (microscaling) block-scale helpers.

MX is a block-scaling scheme: a narrow element tensor plus a per-block scale
tensor (UE8M0). These helpers build/apply canonical scales for the numpy
reference and (pre-)swizzle scales into the mode-1001 device layout. The
reference path always uses canonical scales; only the device copy is swizzled.
"""
import numpy as np


def build_block_scales(ref_f32, block=32):
    """Compute per-block UE8M0 scales and block-scaled element values.

    Parameters
    ----------
    ref_f32 : np.ndarray, shape (rows, cols), dtype float32
        Input matrix. ``cols`` must be divisible by ``block``.
    block : int
        Block size along the innermost dimension (default 32).

    Returns
    -------
    scales_ue8m0 : np.ndarray, shape (rows, cols // block), dtype uint8
        Per-block biased exponent (UE8M0, bias=127). Zero blocks map to 127
        (scale = 2^0 = 1.0).
    scaled_elems_f32 : np.ndarray, shape (rows, cols), dtype float32
        Element values divided by their block's power-of-two scale.
    """
    ref = np.asarray(ref_f32, dtype=np.float32)
    rows, cols = ref.shape
    assert cols % block == 0, "innermost dim must be a multiple of block"
    nblocks = cols // block
    blocks = ref.reshape(rows, nblocks, block)
    # per-block max magnitude -> power-of-two exponent (UE8M0 stores the exponent)
    amax = np.max(np.abs(blocks), axis=2)
    # zero blocks -> treat as scale = 1 (ue8m0 = 127, i.e. 2^0)
    amax = np.where(amax == 0, 1.0, amax)
    exp = np.floor(np.log2(amax)).astype(np.int32)
    # UE8M0 stores biased exponent (bias 127); clamp to [0, 254].
    ue8m0 = np.clip(exp + 127, 0, 254).astype(np.uint8)
    scale = (2.0 ** (ue8m0.astype(np.float32) - 127.0))[:, :, None]
    scaled = (blocks / scale).reshape(rows, cols).astype(np.float32)
    return ue8m0, scaled


def apply_block_scales(elems_f32, scales_ue8m0, block=32):
    """Reconstruct effective float32 values from block-scaled elements.

    This is the inverse of :func:`build_block_scales`; it multiplies each
    block of elements by its power-of-two scale.  Used to form the numpy
    reference for MX GEMM correctness checks.

    Parameters
    ----------
    elems_f32 : np.ndarray, shape (rows, cols), dtype float32
    scales_ue8m0 : np.ndarray, shape (rows, cols // block), dtype uint8
    block : int

    Returns
    -------
    np.ndarray, shape (rows, cols), dtype float32
    """
    elems = np.asarray(elems_f32, dtype=np.float32)
    rows, cols = elems.shape
    nblocks = cols // block
    scale = (2.0 ** (scales_ue8m0.astype(np.float32) - 127.0))[:, :, None]
    out = (elems.reshape(rows, nblocks, block) * scale).reshape(rows, cols)
    return out.astype(np.float32)


# VERIFY-ON-MI350: the forward permutation below is derived from
# DataInitialization.cpp:1977-2016 but only the roundtrip (swizzle→unswizzle)
# is verified on gfx942. Whether the forward layout actually feeds the gfx950
# subtile kernel correctly (mode BLK32_UE8M0_32_8_EXT / 1001) is UNVERIFIED
# until run on MI350. If a mode-1001 GEMM produces wrong numbers, re-derive here.
def swizzle_scales(scales_canonical, tile=(32, 8, 4)):
    """Permute canonical (row, col) scale bytes into the mode-1001 device order.

    Ported from tensilelite/client/src/DataInitialization.cpp (generateMXInput,
    ~lines 1977-2016). ``tile = (tileMN, tileK, subTileK)``.

    The permutation reshapes the canonical ``(rows, cols)`` scale tensor into
    ``(rowTiles, tileMN, colTiles, tileK)`` then transposes to
    ``(rowTiles, colTiles, tileK, tileMN)`` and flattens, matching the device
    subtile layout that waves read from.

    The roundtrip ``unswizzle_scales(swizzle_scales(x)) == x`` is tested on
    gfx942. The forward layout is verified on MI350.

    Parameters
    ----------
    scales_canonical : np.ndarray, shape (rows, cols), dtype uint8
        UE8M0 scale tensor in canonical (row-major) order.
    tile : tuple of int
        (tileMN, tileK, subTileK). Default (32, 8, 4).

    Returns
    -------
    np.ndarray, flat uint8 array
    """
    tileMN, tileK, subTileK = tile
    rows, cols = scales_canonical.shape
    assert rows % tileMN == 0 and cols % tileK == 0, "shape must tile evenly"
    rt, ct = rows // tileMN, cols // tileK
    a = scales_canonical.reshape(rt, tileMN, ct, tileK)
    a = a.transpose(0, 2, 3, 1)  # (rt, ct, tileK, tileMN)
    return np.ascontiguousarray(a).reshape(-1)


def unswizzle_scales(swizzled, tile=(32, 8, 4), shape=None):
    """Inverse of :func:`swizzle_scales`: recover the canonical (rows, cols) layout.

    Parameters
    ----------
    swizzled : np.ndarray, flat uint8 array
    tile : tuple of int
        Must match the ``tile`` used in :func:`swizzle_scales`.
    shape : tuple of int
        (rows, cols) of the original canonical scale tensor.

    Returns
    -------
    np.ndarray, shape ``shape``, dtype uint8
    """
    tileMN, tileK, subTileK = tile
    rows, cols = shape
    rt, ct = rows // tileMN, cols // tileK
    a = swizzled.reshape(rt, ct, tileK, tileMN)
    a = a.transpose(0, 3, 1, 2)  # back to (rt, tileMN, ct, tileK)
    return np.ascontiguousarray(a).reshape(rows, cols)
