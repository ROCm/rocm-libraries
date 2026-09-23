# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 TDM (Tensor Data Mover) descriptor construction.

The TDM engine copies a rectangular tile of a global tensor into LDS from a
single wave-uniform descriptor, replacing the per-lane address math the
direct-to-LDS path emits. The descriptor is five SGPR groups
``(4, 8, 4, 4, 8) x i32`` handed to ``tensor_load_to_lds``; for the rank<=2
non-gather shape this GEMM needs, groups 2/3/4 are zero and only groups 0 and
1 carry fields.

The bit layout mirrors ``TDM_GROUP0`` / ``TDM_GROUP1`` in CK's
``core/arch/amd_tdm_descriptor.hpp`` exactly. Field placement is the one part
of this file that cannot be derived: a wrong bit gives silent garbage or a
hang, never a compile error, so the tables below are transcribed from the
C++ bitfields and covered by :mod:`tests.core.test_tdm`.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

# ``pad_interval`` is 3 bits and ``pad_amount`` 7 bits in TDM_GROUP1.
TDM_PAD_INTERVAL_MAX = 7
TDM_PAD_AMOUNT_MAX = 127
BYTES_PER_DWORD = 4

# ``tensor_dim1_stride_lo`` is a 16-bit field, so CK's 64-bit assignment drops
# bits [16, 32) of the dim-1 stride. Reproduced verbatim (see module docstring),
# which caps the usable leading dimension.
TDM_MAX_DIM1_STRIDE = 0xFFFF

# (word, lsb, width) for every field this builder sets. Groups 2/3/4 are zero
# for rank<=2 non-gather transfers and have no table.
_GROUP0_FIELDS: Dict[str, Tuple[int, int, int]] = {
    "count": (0, 0, 2),
    "is_restore": (0, 2, 1),
    "is_store": (0, 3, 1),
    "nv": (0, 4, 1),
    "scope_trait": (0, 5, 2),
    "th": (0, 7, 3),
    "gather_index_size": (0, 30, 1),
    "gather_mode": (0, 31, 1),
    "lds_addr": (1, 0, 32),
    "global_addr_lo": (2, 0, 32),
    "global_addr_hi": (3, 0, 25),
    "type": (3, 30, 2),
}

_GROUP1_FIELDS: Dict[str, Tuple[int, int, int]] = {
    "workgroup_mask": (0, 0, 16),
    "data_size": (0, 16, 2),
    "atomic_barrier_enable": (0, 18, 1),
    "iterate_enable": (0, 19, 1),
    "pad_enable": (0, 20, 1),
    "early_timeout": (0, 21, 1),
    "pad_interval": (0, 22, 3),
    "pad_amount": (0, 25, 7),
    "atomic_barrier_address": (1, 0, 16),
    "tensor_dim0_lo": (1, 16, 16),
    "tensor_dim0_hi": (2, 0, 16),
    "tensor_dim1_lo": (2, 16, 16),
    "tensor_dim1_hi": (3, 0, 16),
    "tile_dim0": (3, 16, 16),
    "tile_dim1": (4, 0, 16),
    "tile_dim2": (4, 16, 16),
    "tensor_dim0_stride_lo": (5, 0, 32),
    "tensor_dim0_stride_hi": (6, 0, 16),
    "tensor_dim1_stride_lo": (6, 16, 16),
    "tensor_dim1_stride_hi": (7, 0, 32),
}

# CK's TDM_GROUP0 constructor: ``count = 1`` and ``type = 2`` ("set to 2 for
# spg") are unexplained in the SPG and copied verbatim.
_GROUP0_COUNT = 1
_GROUP0_TYPE = 2


def tdm_data_size_code(elem_bytes: int) -> int:
    """``TDM_GROUP1.data_size`` for an element width in bytes."""
    code = {8: 3, 4: 2, 2: 1, 1: 0}.get(int(elem_bytes))
    if code is None:
        raise ValueError(f"TDM data_size has no encoding for {elem_bytes}-byte elements")
    return code


def encode_tdm_padding(interval_bytes: int, pad_bytes: int) -> Tuple[int, int]:
    """Encode a row-stride pad as ``(pad_interval, pad_amount)``.

    The mover inserts ``pad_bytes`` of gap into the LDS destination after every
    ``interval_bytes`` of payload. Both are counted in dwords, and the interval
    is a log2 biased by one:

        ``pad_interval = log2(interval_bytes / 4) - 1``
        ``pad_amount   = pad_bytes / 4 - 1``

    Setting ``interval_bytes`` to one tile row and ``pad_bytes`` to the ROCKE
    ``lds_k_pad`` gap reproduces the VGPR path's padded row stride exactly, so
    the ``ds_read`` side needs no change.
    """
    interval_bytes = int(interval_bytes)
    pad_bytes = int(pad_bytes)
    if pad_bytes <= 0:
        raise ValueError("encode_tdm_padding needs a positive pad; pad_enable=0 otherwise")
    if interval_bytes % BYTES_PER_DWORD or pad_bytes % BYTES_PER_DWORD:
        raise ValueError(
            f"TDM padding is dword-granular, got interval={interval_bytes}B pad={pad_bytes}B"
        )
    interval_dwords = interval_bytes // BYTES_PER_DWORD
    if interval_dwords < 2 or interval_dwords & (interval_dwords - 1):
        raise ValueError(
            f"TDM pad interval must be a power-of-two dword count >= 2, got {interval_dwords}"
        )
    pad_interval = interval_dwords.bit_length() - 2
    pad_amount = pad_bytes // BYTES_PER_DWORD - 1
    if pad_interval > TDM_PAD_INTERVAL_MAX:
        raise ValueError(
            f"TDM pad_interval {pad_interval} exceeds {TDM_PAD_INTERVAL_MAX} "
            f"(interval {interval_bytes}B too large)"
        )
    if pad_amount > TDM_PAD_AMOUNT_MAX:
        raise ValueError(
            f"TDM pad_amount {pad_amount} exceeds {TDM_PAD_AMOUNT_MAX} "
            f"(pad {pad_bytes}B too large)"
        )
    return pad_interval, pad_amount


def tdm_padding_for_tile(elem_bytes: int, row_elems: int, pad_elems: int):
    """``(pad_enable, pad_interval, pad_amount)`` for a ``row_elems``-wide tile.

    ``pad_elems == 0`` disables padding, which is what the unpadded LDS layout
    wants; the two encoded fields are then don't-care zeros.
    """
    if int(pad_elems) == 0:
        return 0, 0, 0
    interval, amount = encode_tdm_padding(
        int(elem_bytes) * int(row_elems), int(elem_bytes) * int(pad_elems)
    )
    return 1, interval, amount


def _pack(fields: Dict[str, Tuple[int, int, int]], words: int, values: Dict[str, int]):
    """OR ``values`` into a zeroed word vector using a field table."""
    packed = [0] * words
    for name, value in values.items():
        try:
            word, lsb, width = fields[name]
        except KeyError:
            raise KeyError(f"unknown TDM descriptor field {name!r}") from None
        value = int(value)
        if value < 0 or value >> width:
            raise ValueError(
                f"TDM field {name!r} is {width} bits, cannot hold {value} (0x{value:x})"
            )
        packed[word] |= value << lsb
    return packed


def pack_tdm_group0(
    *,
    lds_addr: int,
    global_addr: int,
    gather_index_size: int = 0,
    gather_mode: int = 0,
):
    """Pack ``TDM_GROUP0`` (4 x i32) from concrete integers."""
    global_addr = int(global_addr)
    return _pack(
        _GROUP0_FIELDS,
        4,
        {
            "count": _GROUP0_COUNT,
            "lds_addr": int(lds_addr) & 0xFFFFFFFF,
            "global_addr_lo": global_addr & 0xFFFFFFFF,
            "global_addr_hi": (global_addr >> 32) & 0x1FFFFFF,
            "type": _GROUP0_TYPE,
            "gather_index_size": gather_index_size,
            "gather_mode": gather_mode,
        },
    )


def pack_tdm_group1_2d(
    *,
    elem_bytes: int,
    tensor_dim0: int,
    tensor_dim1: int,
    tile_dim0: int,
    tile_dim1: int,
    dim0_stride: int,
    dim1_stride: int,
    pad_enable: int = 0,
    pad_interval: int = 0,
    pad_amount: int = 0,
    workgroup_mask: int = 0,
    atomic_barrier_enable: int = 0,
    atomic_barrier_address: int = 0,
):
    """Pack ``TDM_GROUP1`` (8 x i32) for a rank-2 non-gather transfer.

    Extents and tile dims are reversed relative to the natural tile shape, so
    ``dim0`` is the contiguous dimension: a ``[block_m, block_k]`` tile of a
    row-major ``MxK`` tensor has ``tile_dim0 = block_k`` and
    ``tile_dim1 = block_m``.

    The strides are *not* reversed, which is the asymmetry that makes this easy
    to get wrong. Stride slot 0 is the pitch from one ``dim0`` row to the next,
    so a row-major ``MxK`` tensor wants ``dim0_stride = K`` and
    ``dim1_stride = 1``, matching the unreversed stride array CK hands the
    descriptor. Measured directly on gfx1250: the observed row step equals the
    slot-0 value exactly, and slots for dim1 have no effect at rank 2. Strides
    and extents are in elements, not bytes.
    """
    dim0_stride = int(dim0_stride)
    dim1_stride = int(dim1_stride)
    if dim1_stride > TDM_MAX_DIM1_STRIDE:
        raise ValueError(
            f"TDM dim1 stride {dim1_stride} exceeds the {TDM_MAX_DIM1_STRIDE}-element "
            "field; the descriptor truncates it to 16 bits"
        )
    tensor_dim0 = int(tensor_dim0)
    tensor_dim1 = int(tensor_dim1)
    return _pack(
        _GROUP1_FIELDS,
        8,
        {
            "workgroup_mask": workgroup_mask,
            "data_size": tdm_data_size_code(elem_bytes),
            "atomic_barrier_enable": atomic_barrier_enable,
            "atomic_barrier_address": atomic_barrier_address,
            "iterate_enable": 0,
            "pad_enable": pad_enable,
            "early_timeout": 0,
            "pad_interval": pad_interval,
            "pad_amount": pad_amount,
            "tensor_dim0_lo": tensor_dim0 & 0xFFFF,
            "tensor_dim0_hi": (tensor_dim0 >> 16) & 0xFFFF,
            "tensor_dim1_lo": tensor_dim1 & 0xFFFF,
            "tensor_dim1_hi": (tensor_dim1 >> 16) & 0xFFFF,
            "tile_dim0": int(tile_dim0),
            "tile_dim1": int(tile_dim1),
            "tile_dim2": 0,
            "tensor_dim0_stride_lo": dim0_stride & 0xFFFFFFFF,
            "tensor_dim0_stride_hi": (dim0_stride >> 32) & 0xFFFF,
            # CK truncates the dim-1 stride to 16 bits here; see TDM_MAX_DIM1_STRIDE.
            "tensor_dim1_stride_lo": dim1_stride & 0xFFFF,
            "tensor_dim1_stride_hi": (dim1_stride >> 32) & 0xFFFFFFFF,
        },
    )


def _insert_words(b, words, static: Dict[int, int], dynamic: Dict[int, object]):
    """Build an ``<n x i32>`` vector from constant and runtime words.

    Every runtime word goes through ``readfirstlane`` first: the descriptor is
    read as SGPRs, so a VGPR-resident word would be silently wrong.
    """
    from rocke.core.ir import I32

    vec = b.zero_vec(I32, words)
    for index in range(words):
        value = dynamic.get(index)
        if value is not None:
            vec = b.vec_insert(vec, b.readfirstlane(value), index)
        elif static.get(index):
            vec = b.vec_insert(vec, b.const_i32(static[index]), index)
    return vec


def build_tdm_descriptor_2d(
    b,
    *,
    global_addr,
    lds_addr,
    elem_bytes: int,
    tensor_dim0,
    tensor_dim1,
    tile_dim0: int,
    tile_dim1: int,
    dim0_stride: int,
    dim1_stride: int,
    pad_enable: int = 0,
    pad_interval: int = 0,
    pad_amount: int = 0,
):
    """Emit the five descriptor groups for a rank-2 non-gather TDM load.

    ``global_addr`` (i64), ``lds_addr`` (i64), the two ``tensor_dim`` extents
    and ``dim1_stride`` may be IR values; everything else is a compile-time
    constant folded into the static word pattern. A runtime ``dim1_stride``
    still has to fit the 16-bit field, so the caller must bound the leading
    dimension by :data:`TDM_MAX_DIM1_STRIDE`. Returns ``(d0, d1, d2, d3, d4)``
    ready for :meth:`IRBuilder.tensor_load_to_lds`.
    """
    from rocke.core.ir import I32

    # Group 0: the two addresses are runtime, the rest is a constant word.
    lds32 = b.trunc(lds_addr, I32)
    g_lo = b.trunc(global_addr, I32)
    g_hi = b.land(
        b.trunc(b.lshr(global_addr, b.const_i64(32)), I32), b.const_i32(0x1FFFFFF)
    )
    word0 = _pack(
        _GROUP0_FIELDS,
        4,
        {"count": _GROUP0_COUNT, "type": _GROUP0_TYPE},
    )
    # ``type`` shares word 3 with global_addr_hi, so that word is merged at
    # runtime rather than taken from the static pattern.
    d0 = _insert_words(
        b,
        4,
        {0: word0[0]},
        {1: lds32, 2: g_lo, 3: b.lor(g_hi, b.const_i32(word0[3]))},
    )

    # Group 1: the tensor extents are runtime (they shrink as the window walks
    # off the end of the tensor and the mover clips the copy), and the row
    # pitch in slot 0 may be too.
    stride0_is_value = not isinstance(dim0_stride, int)
    static1 = pack_tdm_group1_2d(
        elem_bytes=elem_bytes,
        tensor_dim0=0,
        tensor_dim1=0,
        tile_dim0=tile_dim0,
        tile_dim1=tile_dim1,
        dim0_stride=0 if stride0_is_value else dim0_stride,
        dim1_stride=dim1_stride,
        pad_enable=pad_enable,
        pad_interval=pad_interval,
        pad_amount=pad_amount,
    )
    mask16 = b.const_i32(0xFFFF)
    sixteen = b.const_i32(16)
    dim0_lo = b.shl(b.land(tensor_dim0, mask16), sixteen)
    dim0_hi = b.land(b.lshr(tensor_dim0, sixteen), mask16)
    dim1_lo = b.shl(b.land(tensor_dim1, mask16), sixteen)
    dim1_hi = b.land(b.lshr(tensor_dim1, sixteen), mask16)
    runtime1 = {
        1: b.lor(dim0_lo, b.const_i32(static1[1])),
        2: b.lor(b.lor(dim0_hi, dim1_lo), b.const_i32(static1[2])),
        3: b.lor(dim1_hi, b.const_i32(static1[3])),
    }
    if stride0_is_value:
        # Slot 0 is 48 bits (word 5 plus word 6's low half); a row pitch that
        # fits an i32 leaves the high half at its static value.
        runtime1[5] = dim0_stride
    d1 = _insert_words(
        b,
        8,
        {
            index: static1[index]
            for index in (0, 4, 5, 6, 7)
            if index not in runtime1
        },
        runtime1,
    )

    zeros4 = b.zero_vec(I32, 4)
    zeros8 = b.zero_vec(I32, 8)
    return d0, d1, zeros4, zeros4, zeros8
